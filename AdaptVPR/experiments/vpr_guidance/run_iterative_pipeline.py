"""End-to-end example: Using LoRA fine-tuning in an iterative pipeline.

This script demonstrates the complete workflow:
1. Generate baseline data with vanilla IC-Light
2. Train SALAD on real + generated data
3. Evaluate and extract hard cases
4. Fine-tune LoRA on hard cases
5. Generate new data with fine-tuned LoRA
6. Repeat
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_command(cmd: list[str], description: str) -> None:
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with code {result.returncode}")

    print(f"\n✓ {description} completed successfully")


def iterative_training_pipeline(
    gsv_root: Path,
    base_model_path: Path,
    output_root: Path,
    num_rounds: int = 3,
    samples_per_round: int = 1000,
):
    """Run iterative training pipeline.

    Args:
        gsv_root: GSV-Cities root directory
        base_model_path: Path to SD1.5 base model
        output_root: Root directory for outputs
        num_rounds: Number of training rounds
        samples_per_round: Number of samples to generate per round
    """
    output_root.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config = {
        "gsv_root": str(gsv_root),
        "base_model_path": str(base_model_path),
        "output_root": str(output_root),
        "num_rounds": num_rounds,
        "samples_per_round": samples_per_round,
    }
    config_path = output_root / "experiment_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Saved experiment config to {config_path}")

    for round_idx in range(num_rounds):
        print(f"\n{'#'*60}")
        print(f"# ROUND {round_idx + 1} / {num_rounds}")
        print(f"{'#'*60}")

        round_dir = output_root / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # === Step 1: Generate synthetic data ===
        generated_dir = round_dir / "generated"

        if round_idx == 0:
            print("\n[Step 1] Generating baseline data with vanilla IC-Light...")
            generator_note = "vanilla IC-Light"
            lora_checkpoint = None
        else:
            prev_round_dir = output_root / f"round_{round_idx - 1}"
            lora_checkpoint = prev_round_dir / "lora" / "lora_final.safetensors"

            if not lora_checkpoint.exists():
                print(f"Warning: LoRA checkpoint not found: {lora_checkpoint}")
                print("Falling back to vanilla IC-Light")
                lora_checkpoint = None
                generator_note = "vanilla IC-Light (fallback)"
            else:
                print(f"\n[Step 1] Generating data with fine-tuned LoRA from round {round_idx}...")
                generator_note = f"LoRA from round {round_idx - 1}"

        # TODO: Call your generation script here
        # For now, just create a placeholder manifest
        print(f"  Generator: {generator_note}")
        print(f"  Output: {generated_dir}")
        print(f"  Target samples: {samples_per_round}")

        generation_manifest = {
            "round": round_idx,
            "generator": generator_note,
            "lora_checkpoint": str(lora_checkpoint) if lora_checkpoint else None,
            "output_dir": str(generated_dir),
            "target_samples": samples_per_round,
            "note": "In real pipeline, run AdaptVPR generation here with the LoRA checkpoint",
        }
        (round_dir / "generation_manifest.json").write_text(json.dumps(generation_manifest, indent=2))

        # === Step 2: Train SALAD ===
        salad_dir = round_dir / "salad"
        salad_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Step 2] Training SALAD on real + generated data...")
        print(f"  Real data: {gsv_root}")
        print(f"  Synthetic data: {generated_dir}")
        print(f"  Output: {salad_dir}")

        # TODO: Call your SALAD training script here
        # Example:
        # run_command([
        #     "python", "train_salad.py",
        #     "--real-data", str(gsv_root),
        #     "--synthetic-data", str(generated_dir),
        #     "--output", str(salad_dir),
        #     "--epochs", "50",
        # ], f"SALAD training (round {round_idx})")

        salad_manifest = {
            "round": round_idx,
            "real_data": str(gsv_root),
            "synthetic_data": str(generated_dir),
            "checkpoint": str(salad_dir / "salad_checkpoint.pt"),
            "note": "In real pipeline, train SALAD here",
        }
        (round_dir / "salad_manifest.json").write_text(json.dumps(salad_manifest, indent=2))

        # === Step 3: Evaluate SALAD and extract hard cases ===
        eval_dir = round_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Step 3] Evaluating SALAD on validation sets...")
        print(f"  Validation sets: SVOX, Nordland, RobotCar")
        print(f"  Output: {eval_dir}")

        # TODO: Call your SALAD evaluation script here
        # Example:
        # for dataset in ["SVOX", "Nordland", "RobotCar"]:
        #     run_command([
        #         "python", "evaluate_salad.py",
        #         "--checkpoint", str(salad_dir / "salad_checkpoint.pt"),
        #         "--dataset", dataset,
        #         "--output", str(eval_dir / f"{dataset}_results.json"),
        #     ], f"SALAD evaluation on {dataset}")

        # Extract hard cases
        hard_cases_path = round_dir / "hard_cases.json"

        print(f"\n[Step 3b] Extracting hard cases from evaluation results...")
        print(f"  Output: {hard_cases_path}")

        # TODO: In real pipeline, merge hard cases from all validation sets
        # Example:
        # run_command([
        #     "python", "extract_hard_cases.py", "merge",
        #     str(eval_dir / "SVOX_results.json"),
        #     str(eval_dir / "Nordland_results.json"),
        #     str(eval_dir / "RobotCar_results.json"),
        #     "--output", str(hard_cases_path),
        #     "--max-cases", "500",
        # ], "Extracting hard cases")

        # Placeholder
        placeholder_hard_cases = {
            "source": "placeholder",
            "total_error_queries": 0,
            "hard_cases_extracted": 0,
            "error_cases": [],
            "note": "In real pipeline, extract hard cases from SALAD evaluation",
        }
        hard_cases_path.write_text(json.dumps(placeholder_hard_cases, indent=2))

        # === Step 4: Fine-tune LoRA (skip on last round) ===
        if round_idx < num_rounds - 1:
            lora_dir = round_dir / "lora"
            lora_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[Step 4] Fine-tuning LoRA on hard cases...")
            print(f"  Hard cases: {hard_cases_path}")
            print(f"  Output: {lora_dir}")

            # Check if we have hard cases
            hard_cases_data = json.loads(hard_cases_path.read_text())
            num_hard_cases = hard_cases_data.get("hard_cases_extracted", 0)

            if num_hard_cases == 0:
                print("  Warning: No hard cases found, skipping LoRA fine-tuning")
            else:
                # TODO: Call the fine-tuning script
                # Example:
                # run_command([
                #     "python", "finetune_generator.py",
                #     "--hard-cases", str(hard_cases_path),
                #     "--gsv-root", str(gsv_root),
                #     "--base-model", str(base_model_path),
                #     "--output-dir", str(lora_dir),
                #     "--lora-rank", "8",
                #     "--lora-alpha", "8.0",
                #     "--batch-size", "4",
                #     "--learning-rate", "1e-4",
                #     "--num-steps", "1000",
                #     "--lambda-diff", "1.0",
                #     "--lambda-identity", "0.5",
                #     "--lambda-diverse", "0.1",
                # ], f"LoRA fine-tuning (round {round_idx})")

                lora_manifest = {
                    "round": round_idx,
                    "hard_cases": str(hard_cases_path),
                    "num_hard_cases": num_hard_cases,
                    "checkpoint": str(lora_dir / "lora_final.safetensors"),
                    "note": "In real pipeline, fine-tune LoRA here",
                }
                (round_dir / "lora_manifest.json").write_text(json.dumps(lora_manifest, indent=2))

        else:
            print(f"\n[Step 4] Skipping LoRA fine-tuning (last round)")

        # === Step 5: Round summary ===
        print(f"\n{'='*60}")
        print(f"Round {round_idx + 1} summary:")
        print(f"  Generated data: {generated_dir}")
        print(f"  SALAD checkpoint: {salad_dir}")
        print(f"  Hard cases: {hard_cases_path}")
        if round_idx < num_rounds - 1:
            print(f"  LoRA checkpoint: {lora_dir}")
        print(f"{'='*60}")

    # === Final summary ===
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT COMPLETE")
    print(f"{'#'*60}")
    print(f"\nAll rounds completed. Results in: {output_root}")
    print(f"\nTo evaluate final performance:")
    print(f"  1. Compare SALAD from round 0 (baseline) vs round {num_rounds - 1}")
    print(f"  2. Check recall@1/5/10 on validation sets")
    print(f"  3. Analyze hard case trends across rounds")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end iterative training pipeline example"
    )
    parser.add_argument(
        "--gsv-root",
        type=Path,
        required=True,
        help="GSV-Cities root directory",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        required=True,
        help="Path to SD1.5 base model",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory for outputs",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=3,
        help="Number of training rounds",
    )
    parser.add_argument(
        "--samples-per-round",
        type=int,
        default=1000,
        help="Number of samples to generate per round",
    )

    args = parser.parse_args()

    iterative_training_pipeline(
        gsv_root=args.gsv_root,
        base_model_path=args.base_model,
        output_root=args.output_root,
        num_rounds=args.num_rounds,
        samples_per_round=args.samples_per_round,
    )


if __name__ == "__main__":
    main()
