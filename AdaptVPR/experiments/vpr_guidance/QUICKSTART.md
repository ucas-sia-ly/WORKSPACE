# Quick Start Guide

## Prerequisites

1. **Install dependencies:**
```bash
cd AdaptVPR/experiments/vpr_guidance
pip install -r requirements.txt
```

2. **Prepare data:**
- GSV-Cities dataset downloaded
- SD1.5 base model downloaded
- (Optional) Pre-trained SALAD checkpoint

3. **Verify installation:**
```bash
python test_implementation.py
```

## Minimal Working Example

### Step 0: Create a mock hard_cases.json

For testing without SALAD evaluation, create a minimal hard cases file:

```python
import json
from pathlib import Path

# Create mock hard cases
hard_cases = {
    "source": "manual",
    "total_error_queries": 10,
    "hard_cases_extracted": 10,
    "error_cases": [
        {
            "query_id": "Bangkok/12345_0",
            "query_path": "/path/to/generated/Bangkok/12345_0.jpg",
            "source_id": "Bangkok/12345_0.jpg",
            "correct_match_id": "Bangkok/12345_90",
            "wrong_match_id": "Bangkok/67890_0",
            "retrieval_rank": 10,
            "distance_to_wrong": 0.15,
            "distance_to_correct": 0.40,
        },
        # Add more cases...
    ]
}

Path("hard_cases_mock.json").write_text(json.dumps(hard_cases, indent=2))
```

### Step 1: Fine-tune LoRA

```bash
python finetune_generator.py \
  --hard-cases hard_cases_mock.json \
  --gsv-root /path/to/Gsvcities \
  --base-model /path/to/stable-diffusion-v1-5 \
  --output-dir ./outputs/lora_test \
  --lora-rank 8 \
  --batch-size 2 \
  --num-steps 100 \
  --lambda-diff 1.0 \
  --lambda-identity 0.5 \
  --lambda-diverse 0.1
```

Expected output:
```
Using device: cuda
Loading hard cases from hard_cases_mock.json
Loaded 10 hard cases
Found 10 cases with source images
Loading base model from /path/to/stable-diffusion-v1-5
Injecting LoRA (rank=8, alpha=8.0)
  Injected LoRA into down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q (...)
  ...
Starting training for 100 steps...
Training: 100%|████████| 100/100 [02:15<00:00, loss: 0.1234, diff: 0.0876, iden: 0.0358]
Training complete!
Final checkpoint saved to ./outputs/lora_test/lora_final.safetensors
```

### Step 2: Use the fine-tuned LoRA

```python
import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path

# Import our utilities
import sys
sys.path.insert(0, "AdaptVPR/experiments/vpr_guidance")
from lora_utils import inject_lora_into_unet, load_lora_checkpoint

# Load base pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

# Inject and load LoRA
lora_layers = inject_lora_into_unet(
    pipe.unet,
    rank=8,
    alpha=8.0,
)
load_lora_checkpoint(
    lora_layers,
    Path("outputs/lora_test/lora_final.safetensors"),
)

# Now generate with fine-tuned model
# (Use IC-Light's generation script with this modified UNet)
```

### Step 3: Evaluate improvement

After generating new images with the fine-tuned LoRA:

1. Train a fresh SALAD on real + new generated images
2. Evaluate on validation sets (SVOX, Nordland, RobotCar)
3. Compare recall metrics against baseline

## Common Issues

### "No hard cases with source images found"

**Problem:** The `source_id` fields in `hard_cases.json` don't match GSV-Cities paths.

**Solution:**
```python
# Check your GSV-Cities structure
# Expected: Gsvcities/Images/Bangkok/12345_0.jpg

# Update hard_cases.json source_id to match:
"source_id": "Bangkok/12345_0.jpg"  # Relative to Gsvcities/Images/
```

### "CUDA out of memory"

**Problem:** Batch size too large for GPU.

**Solution:**
```bash
# Reduce batch size
--batch-size 1

# Or use gradient accumulation (modify finetune_generator.py)
```

### "DINOv2 not available"

**Problem:** Cannot load DINOv2 from torch.hub.

**Solution:**
```bash
# Use CLIP instead
--identity-model clip
```

### Training loss doesn't decrease

**Problem:** Learning rate too high or loss weights imbalanced.

**Solution:**
```bash
# Try lower learning rate
--learning-rate 5e-5

# Adjust loss weights
--lambda-diff 1.0 --lambda-identity 0.3 --lambda-diverse 0.05
```

## Next Steps

Once you have the basic pipeline working:

1. **Integrate with SALAD evaluation**: Modify your SALAD eval script to output hard_cases.json
2. **Run full iterative training**: Use `run_iterative_pipeline.py` as a template
3. **Experiment with hyperparameters**: Try different LoRA ranks, loss weights, and training steps
4. **Add diversity from pool**: Modify `DiversityLoss` to compare against previous rounds
5. **Conditional LoRA**: Train separate LoRAs for different conditions (night, rain, etc.)

## Getting Help

If you encounter issues:

1. Run `python test_implementation.py` to verify installation
2. Check the full README.md for detailed documentation
3. Inspect loss curves - they should stabilize, not oscillate wildly
4. Visualize generated images at different checkpoints
5. Compare SALAD recall before/after LoRA fine-tuning

## Performance Expectations

With proper tuning, you should see:

- **LoRA training**: 1-2 hours for 1000 steps on 500 hard cases (V100/A100)
- **Memory**: ~20GB GPU for batch_size=4 at 512x512
- **Downstream improvement**: 2-5% recall@1 improvement on hard domains if the approach works
- **First round**: May not show improvement - benefits accumulate over rounds

## Debugging Checklist

Before reporting issues:

- [ ] Can load hard cases successfully
- [ ] Hard cases have valid source_path pointing to actual images
- [ ] Base model loads without errors
- [ ] LoRA injection completes and reports correct parameter counts
- [ ] First training step completes (loss_diff, loss_identity computed)
- [ ] Checkpoints are saved and can be loaded
- [ ] Generated images with fine-tuned LoRA look reasonable (not broken/noisy)
- [ ] New images pass AdaptVPR verifier at similar rates as baseline
