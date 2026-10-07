# VPR-Guided Generator Fine-tuning

This directory implements **Phase 3: Hard-case-driven LoRA fine-tuning** for the IC-Light generator.

## Design Philosophy

Instead of optimizing a frozen teacher SALAD's descriptor loss (which has representation mismatch issues), this approach uses SALAD's **retrieval errors** as a signal to guide generator improvement:

1. SALAD identifies **hard cases** - queries where it made retrieval mistakes
2. For each hard case, we find the original source image
3. We fine-tune the generator's LoRA to produce **better augmentations** for these hard places
4. The generator learns to create more informative training data for difficult cases

## Key Differences from Previous Approach

### What we DON'T do:
- ❌ Use frozen teacher SALAD as a proxy (avoids C.1 representation gap)
- ❌ Optimize per-image cosine similarity (avoids C.2 batch learning mismatch)
- ❌ Flow SALAD gradients through diffusion (avoids complex gradient chains)

### What we DO:
- ✅ Use SALAD only to **identify** which places need better data
- ✅ Optimize three clean objectives:
  - **L_diff**: Preserve diffusion prior (MSE between predicted and true noise)
  - **L_identity**: Maintain place identity via DINO/CLIP features
  - **L_diverse**: Prevent mode collapse with diversity regularization
- ✅ Simple, stable gradients through standard diffusion training

## Architecture

```
Hard Cases (from SALAD eval)
    ↓
Source Images (GSV-Cities)
    ↓
IC-Light + LoRA → Generated Images
    ↓
Three Losses:
  - L_diff (denoising MSE)
  - L_identity (DINO cosine)
  - L_diverse (pairwise distance)
    ↓
LoRA Parameters Update
```

## Files

- **`hard_cases.py`**: Load and manage hard cases from SALAD validation
- **`losses.py`**: Three loss functions (diffusion, identity, diversity)
- **`lora_utils.py`**: LoRA injection, saving/loading, parameter management
- **`finetune_generator.py`**: Main training script

## Prerequisites

You need:
1. A trained SALAD model that has been evaluated on validation sets
2. The SALAD evaluation must produce a `hard_cases.json` file with error cases
3. GSV-Cities source images
4. SD1.5 base model weights
5. (Optional) LPIPS for better diversity loss

Install dependencies:
```bash
pip install torch torchvision diffusers transformers safetensors lpips
```

## Usage

### Step 1: Generate Hard Cases

First, run SALAD evaluation to get retrieval errors:

```python
# In your SALAD evaluation script, save hard cases:
hard_cases = {
    "error_cases": [
        {
            "query_id": "Bangkok/12345_0",
            "query_path": "/path/to/generated/image.jpg",
            "source_id": "Bangkok/12345_0.jpg",  # GSV-Cities path
            "correct_match_id": "Bangkok/12345_90",
            "wrong_match_id": "Bangkok/67890_0",
            "rank": 15,  # Where correct match actually ranked
            "distance_to_wrong": 0.15,
            "distance_to_correct": 0.45,
        },
        # ... more cases
    ]
}
```

Save this to `hard_cases.json`.

### Step 2: Fine-tune LoRA

```bash
python finetune_generator.py \
  --hard-cases /path/to/hard_cases.json \
  --gsv-root /path/to/Gsvcities \
  --base-model /path/to/stable-diffusion-v1-5 \
  --output-dir ./checkpoints/lora_round_1 \
  --lora-rank 8 \
  --lora-alpha 8.0 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --num-steps 1000 \
  --lambda-diff 1.0 \
  --lambda-identity 0.5 \
  --lambda-diverse 0.1 \
  --identity-model dinov2 \
  --seed 42
```

This will:
- Load hard cases and filter to those with GSV-Cities sources
- Inject LoRA into the UNet attention layers
- Train for 1000 steps optimizing the three losses
- Save checkpoints every 200 steps to `--output-dir`

### Step 3: Use Fine-tuned LoRA for Generation

After training, use the LoRA checkpoint to generate new images:

```python
# Load IC-Light with your fine-tuned LoRA
from lora_utils import inject_lora_into_unet, load_lora_checkpoint

# ... load base IC-Light pipeline ...
lora_layers = inject_lora_into_unet(unet, rank=8, alpha=8.0)
load_lora_checkpoint(lora_layers, Path("checkpoints/lora_round_1/lora_final.safetensors"))

# Now generate with the fine-tuned model
# ... standard IC-Light generation ...
```

### Step 4: Iterate

1. Generate new images with the fine-tuned LoRA
2. Train fresh SALAD on real + new generated images
3. Evaluate and extract new hard cases
4. Repeat fine-tuning for another round

## Hyperparameters

### Loss Weights

- **`lambda_diff=1.0`**: Diffusion prior weight
  - Keep this at 1.0 to maintain generation quality
  - Lower → more aggressive fine-tuning, risk of broken samples
  
- **`lambda_identity=0.5`**: Identity preservation weight
  - Higher → generated images stay closer to source semantically
  - Lower → more freedom to change appearance
  - Recommended range: 0.3-0.8

- **`lambda_diverse=0.1`**: Diversity weight
  - Higher → generated images more different from each other
  - Lower → risk of mode collapse
  - Start small (0.05-0.1) and increase if you see collapse

### LoRA Configuration

- **`lora_rank=8`**: Standard rank for diffusion models
  - Lower (4) → faster, less capacity
  - Higher (16-32) → more capacity, risk of overfitting
  
- **`lora_alpha=8.0`**: Scaling factor, typically equals rank
  - Controls the magnitude of LoRA's contribution
  - alpha/rank ratio determines actual update scale

### Training

- **`learning_rate=1e-4`**: Standard for LoRA fine-tuning
  - Too high → unstable, broken images
  - Too low → slow convergence
  
- **`num_steps=1000`**: Sufficient for small-scale fine-tuning
  - Monitor loss curves to decide if more steps needed
  - For 100-500 hard cases, 500-1500 steps is typical

- **`batch_size=4`**: Depends on GPU memory
  - 512x512 images with SD1.5: 4-8 on 24GB GPU
  - Adjust based on your hardware

## Monitoring Training

Key metrics to watch:

1. **`loss_diff`**: Should stabilize around 0.05-0.15
   - Much higher → generator not learning denoising
   - Much lower → might be overfitting

2. **`loss_identity`**: Should decrease steadily
   - Target: < 0.3 (DINO cosine similarity > 0.7)
   - If stuck high → increase lambda_identity

3. **`loss_diverse`**: Negative value, magnitude should be reasonable
   - Too large negative → images very different (good)
   - Close to zero → potential collapse (bad)

4. **Visual inspection**: Save samples every N steps
   - Check if generated images still look realistic
   - Verify they preserve place identity
   - Ensure diversity across samples

## Expected Behavior

After successful fine-tuning:

- Generated images should preserve building geometry and landmarks
- Weather/illumination effects should be realistic and varied
- For the hard cases, new augmentations should create better training signal
- Diversity within the same source should remain high
- Downstream SALAD trained on new data should improve on validation sets

## Troubleshooting

### "No hard cases with source images found"
- Check `--gsv-root` path is correct
- Verify `hard_cases.json` has valid `source_id` fields matching GSV-Cities structure

### Training loss explodes
- Reduce learning rate (try 5e-5)
- Reduce lambda_identity (try 0.2)
- Check for NaN in data (corrupt images)

### Generated images look broken
- lambda_diff too low → increase to 1.0
- Learning rate too high → reduce to 5e-5
- LoRA rank too high → reduce to 4

### Generated images too similar to source (no diversity)
- Increase lambda_diverse (try 0.2-0.5)
- Check if diversity loss is actually being computed
- Use LPIPS instead of simple pixel distance

### LoRA has no effect on generation
- Check LoRA was correctly injected into UNet
- Verify trainable parameter report shows LoRA params
- Increase lora_alpha (try 16.0)

## Design Rationale

### Why three separate losses instead of SALAD gradient?

The previous approach tried to flow SALAD gradients through the full generation pipeline. This had issues:

1. **Representation mismatch**: Frozen teacher SALAD ≠ fresh downstream SALAD
2. **Complex gradient chain**: VAE decode + SALAD + two-pass VJP is fragile
3. **Wrong objective**: Per-image cosine similarity ≠ batch-level metric learning utility

Our approach:
- SALAD only identifies **which places** need better data (discrete signal)
- LoRA learns **how to augment** those places better (via clean losses)
- No assumption that teacher representation matches downstream fresh SALAD

### Why DINO/CLIP for identity instead of SALAD?

DINO and CLIP are:
- Pre-trained on massive diverse data
- Designed for semantic similarity (not VPR-specific)
- Stable across different training runs
- Available in standard libraries

SALAD changes between rounds (random aggregator init), so using it as a loss target would couple rounds too tightly.

### Why diversity loss?

Without diversity regularization, the easiest way to minimize L_diff + L_identity is to generate images very similar to the source. This defeats the purpose of augmentation. Diversity loss ensures the LoRA explores the space of valid augmentations.

## Future Extensions

1. **Add pool diversity**: Track previously generated images and penalize generating similar ones
2. **Condition on hard case type**: Different LoRA behavior for different error patterns
3. **Adaptive lambda**: Adjust loss weights based on validation performance
4. **Multi-round curriculum**: Start with easy cases, progressively add harder ones
5. **Contrastive variant**: Use wrong_match from hard cases as negative examples

## Citation

If you use this implementation, please cite the original AdaptVPR paper and acknowledge the iterative co-training design.
