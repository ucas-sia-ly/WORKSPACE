"""Explicit bounded GPU smoke: B0-B5 + B5 BF16/FP32, two updates per run.

Runs the actual training entry point, optimizer, AMP and backward paths.
Checkpoint serialization is intercepted to save JSON summaries only; this is
not a training launcher and produces no reusable student weights. No images
are generated and no teacher feature tensors are cached.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_salad import parse_args, train, filter_synthetic_places
from workflow.model import SALADModel
from workflow.training_data import MixedGSVCitiesDataset, coverage_pk_indices


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint",type=Path,required=True)
    parser.add_argument("--backbone-repo",type=Path,required=True)
    parser.add_argument("--diagnostic-dir",type=Path,required=True)
    parser.add_argument("--real-data",type=Path,default=Path("dataset/gsv-cities"))
    parser.add_argument("--output-dir",type=Path,required=True)
    options=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke requires CUDA")
    torch.set_num_threads(2)
    out=options.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    records=json.loads((options.diagnostic_dir/"selected_pairs.json").read_text())["weather_pairs"]
    selected=[next(row for row in records if row["condition"]==weather) for weather in ("night","rain","snow","fog")]
    manifest=out/"manifest.jsonl"
    manifest.write_text("".join(json.dumps(row)+"\n" for row in selected))
    cities=sorted({row["city"] for row in selected})
    dataset=MixedGSVCitiesDataset(options.real_data,manifest,cities=cities,images_per_place=4,
        min_images_per_place=4,synthetic_fraction=0,image_size=(224,224),real_sampling="coverage_pk")
    filter_synthetic_places(dataset)
    if len(dataset)!=4:
        raise ValueError("Smoke requires four distinct eligible places")
    # Choose a seed so every chosen manifest source is in this epoch's K real
    # views. This controls test coverage only; the production sampler is intact.
    seed=None
    for candidate in range(10000):
        if all(any(place.real_paths[i] in place.synthetic_by_source for i in coverage_pk_indices(
            len(place.real_paths),4,seed=candidate,city=place.city,place_id=place.place_id,epoch=0)) for place in dataset.places):
            seed=candidate;break
    if seed is None:
        raise RuntimeError("Could not select a bounded smoke seed")
    (out/"selection.json").write_text(json.dumps(dict(seed=seed,cities=cities,
        statement="Four source views selected into the main K using the ordinary stateless sampler; no sampler bypass."),indent=2))
    cases=[("B0","16",0,0,0,"uniform"),("B1","16",.1,0,0,"uniform"),
           ("B2","16",0,.1,0,"uniform"),("B3","16",0,0,.1,"uniform"),
           ("B4","16",.1,0,.1,"uniform"),("B5","16",.1,0,.1,"verified"),
           ("B5_bf16","bf16",.1,0,.1,"verified"),("B5_fp32","32",.1,0,.1,"verified")]
    summaries=[]
    for name,precision,fw,sw,ow,mode in cases:
        gc.collect();torch.cuda.empty_cache()
        destination=out/name;destination.mkdir()
        command=["--real-data",str(options.real_data),"--synthetic-manifest",str(manifest),
            "--synthetic-places-only","--cities",*cities,"--output-dir",str(destination),
            "--init-checkpoint",str(options.teacher_checkpoint),"--backbone-repo",str(options.backbone_repo),
            "--image-size","224","224","--num-trainable-blocks","1","--epochs","1",
            "--batch-size","2","--images-per-place","4","--min-images-per-place","4",
            "--max-batches-per-epoch","2","--num-workers","0","--real-sampling","coverage_pk",
            "--seed",str(seed),"--synthetic-aux","--synthetic-exposure-prob","1",
            "--synthetic-global-weight",".1","--learning-rate","1e-6","--device","cuda",
            "--precision",precision,"--feature-distill-weight",str(fw),"--score-distill-weight",str(sw),
            "--ot-distill-weight",str(ow),"--pair-confidence-mode",mode,"--profile-training"]
        if fw+sw+ow:
            command += ["--teacher-checkpoint",str(options.teacher_checkpoint),"--teacher-batch-size","1",
                "--pair-calibration",str(options.diagnostic_dir/"calibration.json"),
                "--pair-geometry-cache",str(options.diagnostic_dir/"geometry")]
        args=parse_args(command)
        events,gradients,checkpoints=[],[],[]
        original_forward=SALADModel.forward
        original_backward=torch.Tensor.backward
        original_step=torch.optim.AdamW.step
        def forward(model,images,*a,**kw):
            output=original_forward(model,images,*a,**kw)
            if not model.training:
                assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            elif kw.get("return_aux"):
                for key in ("backbone_local_features","score_logits","log_assignment"):
                    tensor=output[1][key]
                    if tensor.requires_grad:
                        tensor.register_hook(lambda grad,key=key: gradients.append(dict(branch=key,
                            finite=bool(torch.isfinite(grad).all()),abs_sum=float(grad.detach().float().abs().sum()))))
            return output
        def backward(loss,*a,**kw):
            assert bool(torch.isfinite(loss))
            events.append("backward");return original_backward(loss,*a,**kw)
        def step(optimizer,*a,**kw):
            assert all(torch.isfinite(p.grad).all() for group in optimizer.param_groups for p in group["params"] if p.grad is not None)
            events.append("step");return original_step(optimizer,*a,**kw)
        def save(payload,path):
            if path.name=="checkpoint.pt":
                assert payload["global_step"]==2
                assert not any("reliability" in key for key in payload["state_dict"])
                checkpoints.append(dict(global_step=payload["global_step"],metrics=payload["metrics"],
                    training_config=payload["training_config"]))
        with (destination/"train.log").open("w") as log, contextlib.redirect_stdout(log), \
                patch.object(SALADModel,"forward",forward),patch.object(torch.Tensor,"backward",backward), \
                patch.object(torch.optim.AdamW,"step",step),patch("workflow.model.atomic_save_checkpoint",save):
            train(args)
        assert events.count("step")==2, events
        assert events.count("backward")== (4 if name=="B0" else 8),events
        assert all(g["finite"] for g in gradients)
        if fw:
            assert any(g["branch"]=="backbone_local_features" and g["abs_sum"]>0 for g in gradients)
        if sw or ow:
            assert any(g["branch"]=="score_logits" and g["abs_sum"]>0 for g in gradients)
        result=dict(name=name,precision=precision,events=events,local_branch_gradients=gradients,**checkpoints[-1])
        (destination/"checkpoint_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False))
        summaries.append(result)
        (out/"summary.json").write_text(json.dumps(summaries,indent=2,allow_nan=False))
        print(name,json.dumps(result["metrics"],allow_nan=False),flush=True)
    b4=next(r for r in summaries if r["name"]=="B4")["metrics"]["local_distillation"]
    b5=next(r for r in summaries if r["name"]=="B5")["metrics"]["local_distillation"]
    assert b4["positive_a_patches"]==b5["positive_a_patches"]
    print("Complete:",out/"summary.json",flush=True)


if __name__=="__main__":
    main()
