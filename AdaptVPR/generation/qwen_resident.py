"""Keep BF16 encoders and a bounded DiT prefix on a 48 GiB GPU.

The pinned upstream checkout stays untouched. Only module lifetime and weight
placement change; every block and all four denoising steps still execute.
The remaining blocks use the upstream double-buffered disk offload path.
"""
from types import MethodType


def install_resident_cache(pipe, resident_blocks=32):
    import gc
    import torch
    from lightx2v.models.networks.qwen_image.weights.transformer_weights import QwenImageTransformerAttentionBlock

    runner = pipe.runner
    if not runner.config.get("lazy_load") or not runner.config.get("cpu_offload"):
        raise ValueError("Resident BF16 cache requires the prepared disk-offload pipeline")
    if runner.config.get("dit_quantized") or runner.config.get("feature_caching") != "NoCaching":
        raise ValueError("Resident cache requires unquantized weights and full-step inference")
    if not 0 <= resident_blocks < runner.config["num_layers"]:
        raise ValueError("Resident block count must be smaller than the DiT depth")
    cached = {}

    def cached_loader(name, loader):
        def load():
            if name not in cached:
                module = loader()
                if name == "model" and resident_blocks:
                    install_blocks(module)
                cached[name] = module
            # Upstream removes element 0 after encoding; return a fresh list.
            return list(cached[name]) if name == "encoders" else cached[name]
        return load

    def install_blocks(model):
        manager = model.transformer_infer.offload_manager
        resident = []
        cpu = manager.cpu_buffers[0]
        for index in range(resident_blocks):
            free, _ = torch.cuda.mem_get_info()
            if free < 6 * 2**30:
                raise RuntimeError("Resident BF16 profile has less than 6 GiB GPU headroom; reduce resident blocks")
            block = QwenImageTransformerAttentionBlock(
                index, model.config["task"], "Default", model.config,
                create_cuda_buffer=True, lazy_load=True, lazy_load_path=model.lazy_load_path)
            # Weight containers allocate their CUDA buffers in load(), not in
            # the constructor. Lazy headers supply shapes without CPU payloads.
            block.load({})
            cpu.load_state_dict_from_disk(index)
            block.load_state_dict(cpu.state_dict(), index)
            # The pinned CPU buffer is reused for the next block only after H2D.
            torch.cuda.synchronize()
            resident.append(block)
        model.transformer_infer.resident_blocks = resident
        model.transformer_infer.infer_func = MethodType(hybrid_infer, model.transformer_infer)

    def hybrid_infer(infer, blocks, hidden_states, encoder_hidden_states,
                     temb_img_silu, temb_txt_silu, image_rotary_emb, modulate_index):
        manager = infer.offload_manager
        prefix = len(infer.resident_blocks)
        manager.compute_stream.wait_stream(torch.cuda.current_stream())
        manager.start_prefetch_block(prefix)
        with torch.cuda.stream(manager.compute_stream):
            for index, block in enumerate(infer.resident_blocks):
                infer.block_idx = index
                encoder_hidden_states, hidden_states = infer.infer_block(
                    block=block, hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_img_silu=temb_img_silu, temb_txt_silu=temb_txt_silu,
                    image_rotary_emb=image_rotary_emb, modulate_index=modulate_index)
        manager.swap_cpu_buffers()
        with torch.cuda.stream(manager.init_stream):
            manager.cuda_buffers[0].load_state_dict(manager.cpu_buffers[0].state_dict(), prefix)
        manager.init_stream.synchronize()
        for index in range(prefix, infer.num_blocks):
            infer.block_idx = index
            has_next = index + 1 < infer.num_blocks
            if has_next:
                manager.start_prefetch_block(index + 1)
                manager.swap_cpu_buffers()
                manager.prefetch_weights(index + 1, blocks)
            with torch.cuda.stream(manager.compute_stream):
                encoder_hidden_states, hidden_states = infer.infer_block(
                    block=manager.cuda_buffers[0], hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_img_silu=temb_img_silu, temb_txt_silu=temb_txt_silu,
                    image_rotary_emb=image_rotary_emb, modulate_index=modulate_index)
            if has_next:
                manager.swap_blocks()
        torch.cuda.current_stream().wait_stream(manager.compute_stream)
        return hidden_states

    runner.load_text_encoder = cached_loader("encoders", runner.load_text_encoder)
    runner.load_vae = cached_loader("vae", runner.load_vae)
    runner.load_transformer = cached_loader("model", runner.load_transformer)

    def finish():
        runner.model.scheduler.clear()
        if hasattr(runner, "inputs"):
            del runner.inputs
        runner.input_info = None
        torch.cuda.empty_cache()
        gc.collect()

    runner.end_run = finish
    # Expose module lifetime for diagnostics without retaining per-image states.
    runner.adaptvpr_resident_cache = cached
    return {"version": 1, "profile": "resident_bf16_48g", "resident_dit_blocks": resident_blocks,
            "cache_text_encoder": True, "cache_vae": True, "cache_dit_modules": True,
            "weight_dtype": "BF16", "skipped_denoising_steps": 0}
