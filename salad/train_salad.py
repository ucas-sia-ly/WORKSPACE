#!/usr/bin/env python3
"""Train or fine-tune DINOv2 + SALAD on real GSV views and accepted AdaptVPR images."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--real-data", type=Path, required=True, help="GSV root containing Images and Dataframes")
    parser.add_argument("--synthetic-manifest", type=Path, help="Accepted AdaptVPR candidates JSONL; omit for real-only")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50, help="Total epochs, including epochs in --resume")
    parser.add_argument("--batch-size", type=int, default=32, help="Places per batch; image batch is places x images-per-place")
    parser.add_argument("--images-per-place", type=int, default=4)
    parser.add_argument("--real-sampling", choices=["random_pk", "coverage_pk"], default="random_pk",
                        help="random_pk: original sampling; coverage_pk: stateless epoch-based coverage of full real pools")
    parser.add_argument("--min-images-per-place", type=int, default=4)
    parser.add_argument("--synthetic-fraction", type=float, default=0.5,
                        help="mix: target synthetic slots; replace: probability of replacing each selected source view")
    parser.add_argument("--synthetic-mode", choices=["mix", "replace"], default="mix",
                        help="replace preserves the original sampler's source identities and substitutes exact-source variants")
    parser.add_argument("--synthetic-aux", action="store_true",
                        help="CCR-VPR Stage 1: all-real main batch plus at most one exact-source synthetic query per place")
    parser.add_argument("--synthetic-exposure-prob", type=float, default=0.5,
                        help="Per-eligible-place auxiliary trigger probability, not a synthetic image fraction")
    parser.add_argument("--synthetic-global-weight", type=float, default=0.1)
    parser.add_argument("--synthetic-global-temperature", type=float, default=0.1)
    parser.add_argument("--feature-distill-weight", type=float, default=0.0)
    parser.add_argument("--score-distill-weight", type=float, default=0.0)
    parser.add_argument("--ot-distill-weight", type=float, default=0.0)
    parser.add_argument("--pair-confidence-mode", choices=["uniform", "verified"], default="uniform",
                        help="Same Stage 2B A correspondences: uniform q=1 or verified soft positive_conf")
    parser.add_argument("--pair-calibration", type=Path,
                        help="Stage 2B calibration JSON for this teacher and preprocessing; required by local losses")
    parser.add_argument("--pair-calibration-index", type=int, default=0)
    parser.add_argument("--pair-geometry-cache", type=Path,
                        help="Optional Stage 2B geometry-only cache; missing pairs use teacher structural evidence")
    parser.add_argument("--ot-min-teacher-keep", type=float, default=1.1920928955078125e-7,
                        help="Exclude numerically negligible teacher cluster mass from conditional OT; default FP32 epsilon")
    parser.add_argument("--synthetic-places-only", action="store_true",
                        help="Train only eligible places with at least one accepted synthetic view")
    parser.add_argument("--cities", nargs="+", help="CSV city names; default all Dataframes/*.csv")
    parser.add_argument("--image-size", nargs=2, type=int, default=[224, 224], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone", choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"], default="dinov2_vitb14",
                        help="Backbone for fresh runs; full checkpoint initialization infers this")
    parser.add_argument("--aggregator", choices=["salad"], default="salad")
    parser.add_argument("--init-policy", choices=["random_aggregator_pretrained_backbone"],
                        default="random_aggregator_pretrained_backbone")
    parser.add_argument("--teacher-checkpoint", type=Path,
                        help="Full trained SALAD checkpoint (positive global_step) for frozen online targets")
    parser.add_argument("--teacher-batch-size", type=int, default=0,
                        help="Online teacher chunk size; 0 tries the full real batch, CUDA OOM halves the chunk automatically")
    parser.add_argument("--num-trainable-blocks", type=int, default=4)
    parser.add_argument("--num-clusters", type=int, default=64)
    parser.add_argument("--cluster-dim", type=int, default=128)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--reliability-ot", action="store_true",
                        help="Predict patch reliability and bias SALAD's dustbin; requires source companions only during training")
    parser.add_argument("--reliability-lambda", type=float,
                        help="Nonnegative dustbin bias strength; fresh heads use 2.0, learned heads retain their checkpoint value")
    parser.add_argument("--reliability-hidden-dim", type=int,
                        help="Reliability head width; fresh heads use 64, learned heads retain their checkpoint width")
    parser.add_argument("--reliability-context", action=argparse.BooleanOptionalAction, default=None,
                        help="Add a small depthwise 3x3 neighborhood branch to the reliability head")
    parser.add_argument("--reliability-detach-features", action=argparse.BooleanOptionalAction, default=None,
                        help="Stop head/auxiliary gradients at DINO features; VPR gradients still train DINO")
    parser.add_argument("--reliability-mode", choices=["learned", "fixed"],
                        help="fixed uses r=.9 for a compute-path ablation; requires zero auxiliary weights")
    parser.add_argument("--reliability-target-cache", type=Path,
                        help="Offline frozen-teacher target cache; requires --no-augment")
    parser.add_argument("--reliability-corruption-weight", type=float, default=0.0,
                        help="Auxiliary known-mask local geometry damage on real views only; metric images stay clean")
    parser.add_argument("--reliability-corruption-max-images", type=int, default=4,
                        help="Maximum extra real-image forwards per batch when corruption supervision is enabled")
    parser.add_argument("--reliability-head-lr", type=float,
                        help="Learning rate for the new head; otherwise --learning-rate")
    parser.add_argument("--reliability-generated-weight", type=float, default=0.0,
                        help="Stage 3 tri-state BCE on fixed-resize generated pairs; requires calibrated teacher evidence")
    parser.add_argument("--reliability-generated-occlusion-weight", type=float, default=0.0,
                        help="Explicit controlled occlusion copies of generated views, separately reported proxy negatives")
    parser.add_argument("--reliability-loss-weight", type=float, default=0.1)
    parser.add_argument("--reliability-real-prior-weight", type=float, default=0.01)
    parser.add_argument("--reliability-coverage-weight", type=float, default=0.1)
    parser.add_argument("--reliability-coverage-floor", type=float, default=0.5,
                        help="Minimum per-image mean reliability encouraged by the auxiliary loss")
    parser.add_argument("--backbone-repo", type=Path, help="Local DINOv2 checkout with hubconf.py (for offline loading)")
    parser.add_argument("--backbone-weights", type=Path, help="Local DINOv2 state_dict; random SALAD is still used")
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=9.5e-9)
    parser.add_argument("--miner-margin", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--precision", choices=["auto", "32", "16", "bf16"], default="auto")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path, help="Resume this entry point's full checkpoint at an epoch boundary")
    initialization.add_argument("--init-checkpoint", type=Path,
                                help="Fine-tune complete SALAD weights (raw, Lightning, or this entry point's checkpoint), with a fresh optimizer")
    parser.add_argument("--save-every", type=int, default=1, help="Keep a numbered checkpoint every N epochs")
    parser.add_argument("--max-batches-per-epoch", type=int, default=0, help="0 means all batches; positive for a small smoke run")
    parser.add_argument("--profile-training", action="store_true",
                        help="Synchronize CUDA and log per-step compute time/allocator peaks; local distillation enables this automatically")
    parser.add_argument("--no-augment", action="store_true", help="Disable training image augmentation")
    parser.add_argument("--check-data", action="store_true", help="Check metadata/manifest without loading DINOv2 or training")
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "images_per_place", "min_images_per_place", "save_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.batch_size < 2:
        parser.error("--batch-size must be >= 2 places for metric-learning negatives")
    if args.num_workers < 0 or args.max_batches_per_epoch < 0:
        parser.error("Worker and batch limits cannot be negative")
    if args.teacher_batch_size < 0:
        parser.error("--teacher-batch-size cannot be negative")
    if args.teacher_batch_size and args.teacher_checkpoint is None:
        parser.error("--teacher-batch-size requires --teacher-checkpoint")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.miner_margin < 0:
        parser.error("Learning rate must be positive; weight decay/miner margin cannot be negative")
    if args.synthetic_places_only and args.synthetic_manifest is None:
        parser.error("--synthetic-places-only requires --synthetic-manifest")
    if args.synthetic_mode == "replace" and args.synthetic_manifest is None:
        parser.error("--synthetic-mode replace requires --synthetic-manifest")
    if not math.isfinite(args.synthetic_exposure_prob) or not 0 <= args.synthetic_exposure_prob <= 1:
        parser.error("--synthetic-exposure-prob must be finite and in [0,1]")
    if not math.isfinite(args.synthetic_global_weight) or args.synthetic_global_weight < 0:
        parser.error("--synthetic-global-weight must be finite and nonnegative")
    if not math.isfinite(args.synthetic_global_temperature) or args.synthetic_global_temperature < 1.1754943508222875e-38:
        parser.error("--synthetic-global-temperature must be finite and positive in FP32")
    if args.synthetic_aux and args.synthetic_mode != "mix":
        parser.error("CCR-VPR auxiliary training requires --synthetic-mode mix; its main batch is all-real")
    for name in ("feature_distill_weight", "score_distill_weight", "ot_distill_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.pair_calibration_index < 0:
        parser.error("--pair-calibration-index cannot be negative")
    if not math.isfinite(args.ot_min_teacher_keep) or not 0 <= args.ot_min_teacher_keep < 1:
        parser.error("--ot-min-teacher-keep must be finite in [0,1)")
    for name in ("reliability_generated_weight", "reliability_generated_occlusion_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
        if getattr(args, name) > 0 and not args.reliability_ot:
            parser.error("Generated reliability supervision requires --reliability-ot")
    if pair_supervision_enabled(args):
        if not args.synthetic_aux or not args.teacher_checkpoint or not args.pair_calibration:
            parser.error("Local distillation requires --synthetic-aux, --teacher-checkpoint and --pair-calibration")
        if args.feature_distill_weight > 0 and args.num_trainable_blocks < 1:
            parser.error("Feature distillation requires at least one trainable backbone block")
    if args.init_checkpoint and args.backbone_weights:
        parser.error("--init-checkpoint contains backbone weights; omit --backbone-weights")
    for name in ("reliability_loss_weight", "reliability_real_prior_weight", "reliability_coverage_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if not math.isfinite(args.reliability_coverage_floor) or not 0 <= args.reliability_coverage_floor <= 1:
        parser.error("--reliability-coverage-floor must be in [0,1]")
    if args.reliability_lambda is not None and (not math.isfinite(args.reliability_lambda) or args.reliability_lambda < 0):
        parser.error("--reliability-lambda must be finite and nonnegative")
    if args.reliability_hidden_dim is not None and args.reliability_hidden_dim < 1:
        parser.error("--reliability-hidden-dim must be positive")
    if args.reliability_head_lr is not None and (not math.isfinite(args.reliability_head_lr) or args.reliability_head_lr <= 0):
        parser.error("--reliability-head-lr must be finite and positive")
    if not math.isfinite(args.reliability_corruption_weight) or args.reliability_corruption_weight < 0:
        parser.error("--reliability-corruption-weight must be finite and nonnegative")
    if args.reliability_corruption_max_images < 1:
        parser.error("--reliability-corruption-max-images must be positive")
    if args.reliability_target_cache and (not args.no_augment or not args.synthetic_manifest):
        parser.error("--reliability-target-cache requires --no-augment and --synthetic-manifest")
    if not args.reliability_ot and any(value is not None for value in (
            args.reliability_lambda, args.reliability_hidden_dim, args.reliability_head_lr,
            args.reliability_context, args.reliability_detach_features, args.reliability_mode,
            args.reliability_target_cache)):
        parser.error("Reliability head options require --reliability-ot")
    if args.reliability_corruption_weight > 0:
        if not args.reliability_ot:
            parser.error("--reliability-corruption-weight requires --reliability-ot")
        if min(args.image_size) < 56:
            parser.error("Local geometry corruption requires at least a 4x4 DINO patch grid")
    return args


def local_distillation_enabled(args):
    return any(getattr(args, name) > 0 for name in
               ("feature_distill_weight", "score_distill_weight", "ot_distill_weight"))


def pair_supervision_enabled(args):
    return (local_distillation_enabled(args) or args.reliability_generated_weight > 0
            or args.reliability_generated_occlusion_weight > 0)


def seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def resolve_model_initialization(args, resume=None):
    """Infer complete-checkpoint dimensions and retain initialization provenance.

    Image resolution and the number of trainable blocks do not change weight
    shapes, so those two requested settings can be applied to pretrained SALAD.
    Resume uses the saved provenance without requiring the source file again.
    """
    from workflow.model import checkpoint_state_and_config, default_model_config, read_checkpoint, validate_model_config

    state = None
    provenance = {"init_checkpoint": None, "init_checkpoint_sha256": None}
    if args.init_checkpoint:
        path = args.init_checkpoint.expanduser().resolve()
        state, config = checkpoint_state_and_config(read_checkpoint(path))
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        provenance = {"init_checkpoint": str(path), "init_checkpoint_sha256": digest.hexdigest()}
    elif resume is not None:
        _, config = checkpoint_state_and_config(resume)
        saved = resume["training_config"]
        provenance = {name: saved.get(name) for name in provenance}
    else:
        config = default_model_config(args.backbone, args.image_size, args.num_trainable_blocks,
                                      args.num_clusters, args.cluster_dim, args.token_dim)
    config = copy.deepcopy(config)
    config["image_size"] = list(args.image_size)
    config["backbone_config"]["num_trainable_blocks"] = args.num_trainable_blocks
    config["backbone_config"]["return_token"] = True
    aggregator = config["agg_config"]
    saved_enabled = aggregator.get("reliability_ot", False)
    if saved_enabled and not args.reliability_ot:
        raise ValueError("This checkpoint contains a learned reliability head; add --reliability-ot")
    if args.reliability_ot:
        for option, default in (("reliability_lambda", 2.0), ("reliability_hidden_dim", 64)):
            requested = getattr(args, option)
            saved = aggregator.get(option, default)
            if saved_enabled and requested is not None and not math.isclose(requested, saved, rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError(f"--{option.replace('_', '-')} must match the learned reliability checkpoint")
            aggregator[option] = saved if requested is None else requested
        aggregator["reliability_ot"] = True
        for option, default in (("reliability_context", False),
                                ("reliability_detach_features", False),
                                ("reliability_mode", "learned")):
            requested = getattr(args, option)
            saved = aggregator.get(option, default)
            if saved_enabled and option != "reliability_detach_features" and requested is not None and requested != saved:
                raise ValueError(f"--{option.replace('_', '-')} must match the learned reliability checkpoint")
            if requested is not None:
                aggregator[option] = requested
        if aggregator.get("reliability_mode", "learned") == "fixed" and any(weight != 0 for weight in (
                args.reliability_loss_weight, args.reliability_real_prior_weight,
                args.reliability_coverage_weight, args.reliability_corruption_weight,
                args.reliability_generated_weight, args.reliability_generated_occlusion_weight)):
            raise ValueError("Fixed reliability mode requires all auxiliary loss weights to be zero")
    validate_model_config(config)
    return config, state, provenance


def filter_synthetic_places(dataset):
    """Restrict eligible places while keeping labels and data statistics aligned."""
    before = len(dataset.places)
    dataset.places = [place for place in dataset.places if place.synthetic_paths]
    for label, place in enumerate(dataset.places):
        place.label = label
    excluded = before - len(dataset.places)
    dataset.summary.update({
        "synthetic_places_only": True,
        "eligible_places_before_synthetic_filter": before,
        "excluded_places_without_synthetic": excluded,
        "excluded_places": dataset.summary["excluded_places"] + excluded,
        "num_places": len(dataset.places),
        "num_real_images": sum(len(place.real_paths) for place in dataset.places),
        "num_synthetic_images": sum(len(place.synthetic_paths) for place in dataset.places),
        "places_with_synthetic": len(dataset.places),
    })


def resolved_training_config(args, model_config, initialization=None, teacher=None, local_supervisor=None):
    dataframes = args.real_data.expanduser().resolve() / "Dataframes"
    metadata = [dataframes / f"{city}.csv" for city in sorted(args.cities)] if args.cities else sorted(dataframes.glob("*.csv"))
    if args.synthetic_manifest:
        metadata.append(args.synthetic_manifest.expanduser().resolve())
    digest = hashlib.sha256()
    for path in metadata:
        digest.update(str(path).encode("utf-8") + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    config = {
        "real_data": str(args.real_data.expanduser().resolve()),
        "synthetic_manifest": str(args.synthetic_manifest.expanduser().resolve()) if args.synthetic_manifest else None,
        "cities": sorted(args.cities) if args.cities else None,
        "images_per_place": args.images_per_place,
        "real_sampling": args.real_sampling,
        "min_images_per_place": args.min_images_per_place,
        "synthetic_fraction": 0.0 if args.synthetic_aux else args.synthetic_fraction,
        "synthetic_mode": args.synthetic_mode,
        "synthetic_places_only": args.synthetic_places_only,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "miner_margin": args.miner_margin,
        "augment": not args.no_augment,
        "max_batches_per_epoch": args.max_batches_per_epoch,
        "metadata_sha256": digest.hexdigest(),
        "seed": args.seed,
        "num_workers": args.num_workers,
        "model_config": model_config,
        **(initialization or {"init_checkpoint": None, "init_checkpoint_sha256": None}),
    }
    if teacher is not None:
        config["teacher"] = {**copy.deepcopy(teacher.provenance), "batch_size": args.teacher_batch_size}
    if local_supervisor is not None:
        config["local_distillation"] = copy.deepcopy(local_supervisor.provenance)
        config["teacher"].update(targets="online_fixed_resize_source_and_generated",
                                 distillation_loss="independent_feature_score_conditional_ot")
    if args.synthetic_aux:
        config["synthetic_aux"] = {
            "version": 1, "exposure_prob": args.synthetic_exposure_prob,
            "global_weight": args.synthetic_global_weight,
            "global_temperature": args.synthetic_global_temperature,
            "loss": "asymmetric_multi_positive_infonce",
            "reference": "detached_current_real_batch",
            "source_policy": "exact_current_batch_source_only",
            "rng": "independent_checkpointed_python_v1",
        }
    if args.reliability_ot:
        config["reliability"] = {
            "loss_weight": args.reliability_loss_weight,
            "real_prior_weight": args.reliability_real_prior_weight,
            "coverage_weight": args.reliability_coverage_weight,
            "coverage_floor": args.reliability_coverage_floor,
            "head_learning_rate": args.reliability_head_lr if args.reliability_head_lr is not None else args.learning_rate,
            "teacher": "detached_local_self_similarity_reciprocal_structure_v1",
        }
        if args.reliability_target_cache:
            config["reliability"].update(
                teacher="offline_frozen_local_self_similarity_v1",
                target_cache=str(args.reliability_target_cache.expanduser().resolve()),
                target_cache_integrity=args.reliability_cache_integrity,
            )
        if args.reliability_corruption_weight > 0:
            config["reliability"]["controlled_corruption"] = {
                "weight": args.reliability_corruption_weight,
                "max_images": args.reliability_corruption_max_images,
                "method": "real_only_local_copy_paste_v1",
                "rng": "independent_per_step_generator_v1",
            }
        if args.reliability_generated_weight > 0 or args.reliability_generated_occlusion_weight > 0:
            config["reliability"]["generated_supervision"] = dict(
                version=1, weight=args.reliability_generated_weight,
                occlusion_weight=args.reliability_generated_occlusion_weight,
                positives="stage2b_A_positive_conf", negatives="explicit_manifest_masks_only",
                unknown="zero_direct_supervision", controlled_occlusion="separate_copy_2x2_patch_rectangle",
                occlusion_rng="independent_per_step_pair_v1")
    return config


def train(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from workflow.metric_loss import multi_similarity_loss
    from workflow.model import (SALADModel, FrozenSALADTeacher, atomic_save_checkpoint,
                                input_preprocessing, load_model_state, profile_teacher_targets, read_checkpoint)
    from workflow.training_data import CoveragePlaceBatchSampler, MixedGSVCitiesDataset, SamplingCoverage, SyntheticAuxSampler
    from workflow.synthetic_losses import asymmetric_multi_positive_infonce

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    resume = read_checkpoint(args.resume) if args.resume else None
    if resume is not None and (resume.get("format_version") != 1 or "optimizer_state_dict" not in resume):
        raise ValueError("--resume requires a full checkpoint produced by train_salad.py")
    model_config, initial_state, initialization = resolve_model_initialization(args, resume)
    dataset = MixedGSVCitiesDataset(
        real_data=args.real_data, synthetic_manifest=args.synthetic_manifest, cities=args.cities,
        images_per_place=args.images_per_place, min_images_per_place=args.min_images_per_place,
        synthetic_fraction=0.0 if args.synthetic_aux else args.synthetic_fraction, image_size=tuple(args.image_size),
        augment=not args.no_augment, synthetic_mode=args.synthetic_mode,
        real_sampling=args.real_sampling, seed=args.seed, return_sampling_metadata=True,
        **({"reliability_pairs": True} if args.reliability_ot else {}),
        **({"reliability_target_cache": args.reliability_target_cache} if args.reliability_target_cache else {}),
    )
    if args.reliability_target_cache:
        args.reliability_cache_integrity = dataset.reliability_target_cache.integrity_digest
    if args.synthetic_places_only:
        filter_synthetic_places(dataset)
    aux_sampler = SyntheticAuxSampler(dataset, args.synthetic_exposure_prob, args.seed) if args.synthetic_aux else None
    if len(dataset) < 2:
        raise ValueError("Need at least two eligible places for metric-learning positives and negatives")
    print(json.dumps(dataset.summary, indent=2, ensure_ascii=False), flush=True)
    if args.check_data:
        print("Data check passed; no model loaded and no training started.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA devices are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
    if device.type == "cpu":
        os.environ["XFORMERS_DISABLED"] = "1"
    precision = ("16" if device.type == "cuda" else "32") if args.precision == "auto" else args.precision
    if precision == "16" and device.type != "cuda":
        raise ValueError("FP16 training requires CUDA; use --precision 32 on CPU")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support BF16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and (args.output_dir / "checkpoint.pt").exists():
        raise FileExistsError("Output directory already has checkpoint.pt; use --resume or a new output directory")

    teacher = (FrozenSALADTeacher(args.teacher_checkpoint, model_config, device=device,
                                 backbone_repo=args.backbone_repo) if args.teacher_checkpoint else None)
    local_supervisor = None
    if pair_supervision_enabled(args):
        from workflow.local_distillation import LocalDistillationSupervisor
        local_supervisor = LocalDistillationSupervisor(teacher, dataset, args)
    training_config = resolved_training_config(args, model_config, initialization, teacher, local_supervisor)
    if resume is not None:
        saved_training_config = {"synthetic_places_only": False, "synthetic_mode": "mix", "init_checkpoint": None,
                                 "init_checkpoint_sha256": None, "real_sampling": "random_pk", **resume["training_config"]}
        if args.real_sampling == "coverage_pk":
            # Image identities and place order are worker-independent. Random
            # augmentations/model numerics need not be identical across workers.
            saved_training_config["num_workers"] = args.num_workers
        if saved_training_config != training_config:
            raise ValueError("Resume data/model/optimizer options must match the saved training_config")
        if resume["dataset_summary"] != dataset.summary:
            raise ValueError("Dataset summary changed since the saved checkpoint")
        if args.epochs != resume["total_epochs"]:
            raise ValueError("--epochs must match the original cosine schedule's total epochs")
        if precision != resume["precision"]:
            raise ValueError("--precision must match the resumed checkpoint")
        if resume["epoch"] >= args.epochs:
            raise ValueError("This checkpoint has already completed all configured epochs")
    full_weights = resume is not None or initial_state is not None
    model = SALADModel(model_config, pretrained_backbone=not full_weights,
                       backbone_repo=args.backbone_repo,
                       backbone_weights=None if full_weights else args.backbone_weights).to(device)
    if resume is not None:
        model.load_state_dict(resume["state_dict"], strict=True)
    elif initial_state is not None:
        load_model_state(model, initial_state, allow_new_reliability=args.reliability_ot)
    parameters = (p for p in model.parameters() if p.requires_grad)
    if args.reliability_ot:
        from workflow.reliability import reliability_losses
        head_parameters = list(model.aggregator.reliability_parameters())
        head_ids = {id(p) for p in head_parameters}
        parameters = [{"params": [p for p in parameters if id(p) not in head_ids]},
                      {"params": head_parameters,
                       "lr": training_config["reliability"]["head_learning_rate"]}]
    optimizer = torch.optim.AdamW(parameters,
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # PyTorch >= 2.1 supports the CUDA scaler API; it is disabled for CPU/BF16.
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and precision == "16")
    generator = torch.Generator().manual_seed(args.seed)
    start_epoch = 0
    global_step = 0
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        start_epoch, global_step = resume["epoch"], resume["global_step"]
        random.setstate(resume["rng_state"]["python"])
        torch.set_rng_state(resume["rng_state"]["torch"])
        generator.set_state(resume["rng_state"]["loader"])
        np_state = resume["rng_state"]["numpy"]
        np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
        if device.type == "cuda" and resume["rng_state"].get("cuda"):
            torch.cuda.set_rng_state_all(resume["rng_state"]["cuda"])
        if aux_sampler is not None:
            if "synthetic_aux" not in resume["rng_state"]:
                raise ValueError("Auxiliary resume checkpoint is missing its independent sampler RNG")
            aux_sampler.load_state_dict(resume["rng_state"]["synthetic_aux"])

    batch_size = min(args.batch_size, len(dataset))
    coverage_state = resume.get("sampling_coverage") if resume is not None else None
    coverage = SamplingCoverage(dataset, since_epoch=start_epoch, state=coverage_state)
    if resume is not None and coverage_state is None:
        print(f"Checkpoint has no historical sampling statistics; coverage accounting starts at epoch {start_epoch + 1}.",
              flush=True)
    place_batch_sampler = None
    loader_options = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda",
                          generator=generator, worker_init_fn=seed_worker)
    if args.real_sampling == "coverage_pk":
        place_batch_sampler = CoveragePlaceBatchSampler(len(dataset), batch_size, args.seed)
        loader = DataLoader(dataset, batch_sampler=place_batch_sampler, **loader_options)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=len(dataset) % batch_size == 1, **loader_options)
    run_config = {**training_config, "epochs": args.epochs, "seed": args.seed,
                  "device": str(device), "precision": precision, "effective_batch_size": batch_size,
                  "init_policy": "pretrained_salad_checkpoint" if initialization["init_checkpoint"] else args.init_policy,
                  "resume": str(args.resume) if args.resume else None,
                  "backbone_repo": str(args.backbone_repo) if args.backbone_repo else None,
                  "backbone_weights": str(args.backbone_weights) if args.backbone_weights else None}
    (args.output_dir / "training_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (args.output_dir / "data_summary.json").write_text(json.dumps(dataset.summary, indent=2), encoding="utf-8")
    print(f"Device={device}, precision={precision}; batch={batch_size} places x {args.images_per_place} images", flush=True)
    log_path = args.output_dir / "training_log.jsonl"
    sampling_log_path = args.output_dir / "sampling_coverage.jsonl"
    for epoch in range(start_epoch, args.epochs):
        started = time.monotonic()
        dataset.set_epoch(epoch)
        if place_batch_sampler is not None:
            place_batch_sampler.set_epoch(epoch)
        coverage.begin_epoch()
        if aux_sampler is not None:
            aux_sampler.begin_epoch()
        model.train()
        loss_sum = 0.0
        real_loss_sum = aux_loss_sum = 0.0
        aux_steps = 0
        real_exposure = synthetic_exposure = steps = 0
        reliability_sums = {}
        teacher_statistics = None
        local_sums = {}
        runtime_sums = {}
        profiling = local_supervisor is not None or args.profile_training
        local_weights = dict(feature=args.feature_distill_weight, score=args.score_distill_weight,
                             ot=args.ot_distill_weight)
        paired_exposure = 0
        for batch_index, batch in enumerate(loader):
            if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                break
            sampling_metadata = batch[-1]
            batch = batch[:-1]
            if args.reliability_ot:
                images, labels, synthetic, paired_data, paired_info = batch
                cached_targets = cached_confidence = None
                companions = None
                pair_valid_cpu = synthetic.flatten() if args.reliability_target_cache else paired_info.flatten()
                pair_valid = pair_valid_cpu.to(device, non_blocking=True)
                is_synthetic = synthetic.flatten().to(device, non_blocking=True)
                if args.reliability_target_cache:
                    cached_targets = paired_data.flatten(0, 1).to(device, non_blocking=True)
                    cached_confidence = paired_info.flatten(0, 1).to(device, non_blocking=True)
                elif args.reliability_loss_weight > 0:
                    # Move only exact-source companions used by synthetic views.
                    companions = paired_data.flatten(0, 1)[pair_valid_cpu].to(device, non_blocking=True)
            else:
                images, labels, synthetic = batch
            images = images.flatten(0, 1).to(device, non_blocking=True)
            labels = labels.flatten().to(device, non_blocking=True)
            if profiling:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if teacher is not None and local_supervisor is None:
                # Stage 2A probes the online real-image target interface only.
                # Consume chunks immediately; no teacher feature or OT loss yet.
                teacher_images = images if not bool(synthetic.any()) else images[~synthetic.flatten().to(device)]
                report = profile_teacher_targets(teacher, teacher_images, args.teacher_batch_size)
                del teacher_images
                if teacher_statistics is None:
                    teacher_statistics = report
                else:
                    for key in ("images", "chunks", "seconds", "oom_retries"):
                        teacher_statistics[key] += report[key]
                    for key in ("largest_chunk", "cuda_baseline_allocated_bytes",
                                "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
                        if report[key] is not None:
                            teacher_statistics[key] = max(teacher_statistics[key], report[key])
            dtype = torch.float16 if precision == "16" else torch.bfloat16
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "32"):
                if args.reliability_ot:
                    descriptors, auxiliary = model(images, return_aux=True)
                    pair_features = None
                    if companions is not None and len(companions):
                        backbone_training = model.backbone.training
                        try:
                            model.backbone.eval()
                            with torch.no_grad():
                                pair_features, _ = model.backbone(companions)
                        finally:
                            model.backbone.train(backbone_training)
                    auxiliary_losses = reliability_losses(
                        auxiliary["reliability_logits"], auxiliary["local_features"],
                        pair_features, pair_valid if args.reliability_loss_weight > 0 else torch.zeros_like(pair_valid), is_synthetic,
                        auxiliary_weight=args.reliability_loss_weight,
                        real_prior_weight=args.reliability_real_prior_weight,
                        coverage_weight=args.reliability_coverage_weight,
                        coverage_floor=args.reliability_coverage_floor,
                        cached_targets=cached_targets, cached_confidence=cached_confidence,
                    )
                    if args.reliability_corruption_weight > 0:
                        from workflow.reliability_corruption import make_local_corruptions, corrupted_patch_loss
                        corruption_generator = torch.Generator(device=device).manual_seed(
                            (args.seed + 2_000_003 + global_step) % (2 ** 63))
                        corrupted, corruption_mask, corruption_indices = make_local_corruptions(
                            images, is_synthetic, max_images=args.reliability_corruption_max_images,
                            generator=corruption_generator)
                        if len(corrupted):
                            backbone_training = model.backbone.training
                            try:
                                model.backbone.eval()
                                with torch.no_grad():
                                    corrupted_features, _ = model.backbone(corrupted)
                            finally:
                                model.backbone.train(backbone_training)
                            corruption_losses = corrupted_patch_loss(
                                model.aggregator.predict_reliability(corrupted_features.detach()), corruption_mask,
                                reference_logits=auxiliary["reliability_logits"][corruption_indices])
                            auxiliary_losses["total"] = (auxiliary_losses["total"]
                                                        + args.reliability_corruption_weight * corruption_losses["corruption_loss"])
                            auxiliary_losses.update(corruption_losses)
                    metric_loss = multi_similarity_loss(descriptors, labels, epsilon=args.miner_margin)
                    loss = metric_loss + auxiliary_losses["total"]
                else:
                    descriptors = model(images)
                    loss = multi_similarity_loss(descriptors, labels, epsilon=args.miner_margin)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch={epoch + 1}, batch={batch_index}")
            loss_value = float(loss.detach())
            scaler.scale(loss).backward()
            if aux_sampler is not None:
                real_loss_sum += float(metric_loss.detach()) if args.reliability_ot else loss_value
                # No retain_graph: free main activations before the auxiliary forward.
                references, reference_labels = descriptors.detach(), labels.detach()
                del loss, descriptors, images
                aux_batch = aux_sampler.sample(sampling_metadata)
                if len(aux_batch["synthetic_images"]):
                    query_images = aux_batch["synthetic_images"].to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "32"):
                        queries = model(query_images)
                    aux_loss = asymmetric_multi_positive_infonce(
                        queries, aux_batch["synthetic_labels"].to(device), references, reference_labels,
                        temperature=args.synthetic_global_temperature,
                        pair_valid=aux_batch["pair_valid"].to(device))
                    weighted_aux_loss = args.synthetic_global_weight * aux_loss
                    if not torch.isfinite(weighted_aux_loss):
                        optimizer.zero_grad(set_to_none=True)
                        raise RuntimeError(f"Nonfinite synthetic loss at epoch={epoch + 1}, batch={batch_index}")
                    scaler.scale(weighted_aux_loss).backward()
                    aux_loss_sum += float(aux_loss.detach())
                    loss_value += float(weighted_aux_loss.detach())
                    aux_steps += 1
                    del queries, aux_loss, weighted_aux_loss, query_images
                if local_supervisor is not None:
                    try:
                        prepared, totals, local_stats = local_supervisor.prepare(aux_batch)
                        local_metrics = local_supervisor.backward(model, prepared, totals, scaler,
                                                                  precision, local_weights, global_step=global_step)
                    except (RuntimeError, ValueError):
                        optimizer.zero_grad(set_to_none=True)
                        raise
                    loss_value += local_metrics["weighted_total"]
                    for key, value in {**local_stats, **local_metrics}.items():
                        local_sums[key] = local_sums.get(key, 0.) + value
                    del prepared, totals, local_stats, local_metrics
                del references, reference_labels, aux_batch
            # One unscale/clip/step/update for real, query, local and head gradients.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0,
                                          error_if_nonfinite=not scaler.is_enabled())
            scaler.step(optimizer)
            scaler.update()
            if profiling:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    runtime_sums["gpu_peak_allocated_bytes"] = max(runtime_sums.get("gpu_peak_allocated_bytes", 0),
                                                                 torch.cuda.max_memory_allocated(device))
                    runtime_sums["gpu_peak_reserved_bytes"] = max(runtime_sums.get("gpu_peak_reserved_bytes", 0),
                                                                torch.cuda.max_memory_reserved(device))
                runtime_sums["step_seconds"] = runtime_sums.get("step_seconds", 0.) + time.perf_counter() - step_started
            coverage.update(sampling_metadata)
            synthetic_count = int(synthetic.sum())
            synthetic_exposure += synthetic_count
            real_exposure += synthetic.numel() - synthetic_count
            loss_sum += loss_value
            if args.reliability_ot:
                values = {"metric_loss": metric_loss.detach(),
                          "spatial_reliability_std": auxiliary["reliability"].flatten(1).std(dim=1, correction=0).mean().detach(),
                          **auxiliary_losses}
                for name, value in values.items():
                    reliability_sums[name] = reliability_sums.get(name, 0.0) + float(value.detach())
                paired_exposure += int(pair_valid_cpu.sum())
            steps += 1
            global_step += 1
        if not steps:
            raise RuntimeError("Training loader produced no batches")
        scheduler.step()
        row = {"epoch": epoch + 1, "global_step": global_step, "batches": steps,
               "loss": loss_sum / steps, "real_exposure": real_exposure,
               "synthetic_exposure": synthetic_exposure,
               "synthetic_fraction_observed": synthetic_exposure / (real_exposure + synthetic_exposure),
               "learning_rate_next_epoch": optimizer.param_groups[0]["lr"],
               "seconds": time.monotonic() - started,
               "sampling": coverage.summary()}
        if teacher_statistics is not None:
            row["teacher"] = {**teacher_statistics,
                              "seconds_per_image": teacher_statistics["seconds"] / teacher_statistics["images"],
                              "mode": "online_targets_only_no_distillation"}
        if aux_sampler is not None:
            row["synthetic_aux"] = {
                **aux_sampler.summary(), "loss_real": real_loss_sum / steps,
                "loss_syn_per_step": aux_loss_sum / steps,
                "loss_syn_nonempty": aux_loss_sum / aux_steps if aux_steps else 0.0,
                "nonempty_batches": aux_steps, "global_weight": args.synthetic_global_weight,
            }
        if local_supervisor is not None:
            counts = ("pairs", "patches", "positive_a_patches", "excluded_b_patches", "geometry_cache_hits",
                      "feature_patches", "score_patches", "ot_patches", "low_teacher_keep_patches",
                      "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes")
            row["local_distillation"] = {k: v if k in counts else v / steps for k, v in local_sums.items()
                                          if not k.startswith(("generated_region_", "occluded_region_"))}
            if args.reliability_ot:
                from workflow.reliability import summarize_reliability_regions
                for prefix, name in (("generated_region_", "generated_reliability"),
                                     ("occluded_region_", "controlled_generated_occlusion")):
                    statistics = {k[len(prefix):]: v for k, v in local_sums.items() if k.startswith(prefix)}
                    if statistics:
                        row[name] = summarize_reliability_regions(statistics)
            for key in ("positive_a", "feature", "score", "ot"):
                row["local_distillation"][key + "_fraction"] = local_sums.get(key + "_patches", 0) / max(1, local_sums.get("patches", 0))
            row["local_distillation"].update(confidence_mode=args.pair_confidence_mode,
                                             positive_policy="stage2b_A_only", weights=local_weights)
        if profiling:
            row["runtime"] = {**runtime_sums, "step_seconds": runtime_sums["step_seconds"] / steps,
                              "timing_scope": "after_main_input_transfer_through_scaler_update"}
        if args.reliability_ot:
            counts = ("positive_patches", "negative_patches", "unknown_patches")
            all_counts = (*counts, "corruption_negative_patches")
            row["reliability"] = {name: value if name in all_counts else value / steps
                                  for name, value in reliability_sums.items()}
            row["reliability"]["paired_synthetic_exposure"] = paired_exposure
            total_patches = sum(row["reliability"][name] for name in counts)
            row["reliability"]["supervised_fraction"] = (
                (row["reliability"]["positive_patches"] + row["reliability"]["negative_patches"])
                / total_patches if total_patches else 0.0)
            row["reliability"]["head_learning_rate_next_epoch"] = optimizer.param_groups[1]["lr"]
        np_state = np.random.get_state()
        checkpoint = {
            "format_version": 1, "state_dict": model.state_dict(), "model_config": model_config,
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "epoch": epoch + 1, "global_step": global_step,
            "total_epochs": args.epochs, "precision": precision, "training_config": training_config,
            "dataset_summary": dataset.summary, "metrics": row,
            "image_preprocessing": input_preprocessing(args.image_size),
            "sampling_coverage": coverage.state_dict(),
            "rng_state": {"python": random.getstate(), "torch": torch.get_rng_state(),
                          "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
                          "loader": generator.get_state(),
                          "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []},
        }
        if aux_sampler is not None:
            checkpoint["rng_state"]["synthetic_aux"] = aux_sampler.state_dict()
        atomic_save_checkpoint(checkpoint, args.output_dir / "checkpoint.pt")
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            atomic_save_checkpoint(checkpoint, args.output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(row) + "\n")
        with sampling_log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"epoch": epoch + 1, "real_sampling": args.real_sampling,
                                  **coverage.summary(), "places": coverage.place_statistics()},
                                 ensure_ascii=False) + "\n")
        print(json.dumps(row), flush=True)
    print(f"Checkpoint: {args.output_dir / 'checkpoint.pt'}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    try:
        train(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
