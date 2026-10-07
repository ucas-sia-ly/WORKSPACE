# Integration TODOs

This document lists components that need to be implemented or connected to complete the full pipeline.

## ✅ Completed Components

- [x] Hard case loading and management (`hard_cases.py`)
- [x] Three loss functions: diffusion, identity, diversity (`losses.py`)
- [x] LoRA injection and checkpoint management (`lora_utils.py`)
- [x] LoRA fine-tuning script (`finetune_generator.py`)
- [x] Hard case extraction utility (`extract_hard_cases.py`)
- [x] Test suite (`test_implementation.py`)
- [x] Documentation (README, QUICKSTART, DESIGN_COMPARISON)

## ⚠️ Missing Components (Need Implementation)

### 1. SALAD Training Script

**File**: `train_salad.py`

**Purpose**: Train a fresh SALAD model on real + synthetic data

**Inputs**:
- GSV-Cities real images
- AdaptVPR generated synthetic images (from manifest)
- Validation sets (SVOX, Nordland, RobotCar)

**Outputs**:
- Trained SALAD checkpoint
- Training logs and metrics

**Interface**:
```python
python train_salad.py \
  --real-data /path/to/Gsvcities \
  --synthetic-manifest /path/to/generation/manifest.json \
  --output-dir ./outputs/salad \
  --epochs 50 \
  --batch-size 32 \
  --backbone dinov2_vitb14 \
  --aggregator salad \
  --init-policy random_aggregator_pretrained_backbone
```

**Key Requirements**:
- Use pretrained DINOv2 backbone (frozen or partially frozen)
- Initialize SALAD aggregator randomly (not from pretrained)
- Train with MultiSimilarityLoss + hard mining
- Track real vs synthetic exposure per epoch
- Save checkpoints at regular intervals

**Reference**: Original SALAD repository: https://github.com/serizba/salad

---

### 2. SALAD Evaluation Script

**File**: `evaluate_salad.py`

**Purpose**: Evaluate trained SALAD on validation sets and extract error cases

**Inputs**:
- Trained SALAD checkpoint
- Validation dataset (SVOX/Nordland/RobotCar)

**Outputs**:
- Recall@1/5/10 metrics
- Error cases JSON (for hard case extraction)
- Visualization of hard cases (optional)

**Interface**:
```python
python evaluate_salad.py \
  --checkpoint ./outputs/salad/checkpoint.pt \
  --dataset SVOX \
  --dataset-root /path/to/SVOX \
  --output ./outputs/evaluation/SVOX_results.json \
  --save-hard-cases
```

**Output Format** (must match `extract_hard_cases.py` expectations):
```json
{
  "dataset": "SVOX",
  "checkpoint": "/path/to/checkpoint.pt",
  "recall": {
    "R@1": 0.847,
    "R@5": 0.923,
    "R@10": 0.956
  },
  "error_queries": [
    {
      "query_id": "Bangkok/12345_0",
      "query_path": "/path/to/query.jpg",
      "ground_truth": "Bangkok/12345_90",
      "predicted": "Bangkok/67890_0",
      "rank": 15,
      "distance_pred": 0.12,
      "distance_gt": 0.45
    },
    ...
  ]
}
```

---

### 3. IC-Light Generation with LoRA

**File**: `generate_with_lora.py`

**Purpose**: Generate synthetic images using IC-Light + fine-tuned LoRA

**Inputs**:
- GSV-Cities source images
- Generation prompts (from AdaptVPR planning)
- LoRA checkpoint (optional, None = vanilla IC-Light)

**Outputs**:
- Generated images
- Generation manifest (paths, prompts, verification results)

**Interface**:
```python
python generate_with_lora.py \
  --sources /path/to/sources.txt \
  --prompts /path/to/prompts.jsonl \
  --lora-checkpoint ./outputs/lora/lora_final.safetensors \
  --output-dir ./outputs/generated \
  --base-model /path/to/sd15 \
  --iclight-checkpoint /path/to/iclight_sd15_fc.safetensors \
  --num-inference-steps 25 \
  --highres-denoise 0.30
```

**Key Requirements**:
- Load base IC-Light pipeline
- Inject LoRA if provided, else use vanilla
- Run AdaptVPR verification on generated images
- Save only accepted images (passed=True, eligible_for_training=True)
- Create manifest compatible with SALAD training

**Integration Point**: Can reuse existing `AdaptVPR/run.py` generation logic, just need to inject LoRA before generation

---

### 4. Integration with AdaptVPR Generation Pipeline

**File**: Modify `AdaptVPR/adapters/iclight_sd15_fc.py` or create wrapper

**Purpose**: Allow AdaptVPR generation to use a LoRA checkpoint

**Current**: AdaptVPR loads vanilla IC-Light, no LoRA support

**Needed**: Add LoRA loading before generation

**Approach 1: Environment Variable**
```python
# In iclight_sd15_fc.py, after loading UNet:
lora_checkpoint = os.getenv("ADAPTVPR_LORA_CHECKPOINT")
if lora_checkpoint:
    from experiments.vpr_guidance.lora_utils import inject_lora_into_unet, load_lora_checkpoint
    lora_layers = inject_lora_into_unet(unet, rank=8, alpha=8.0)
    load_lora_checkpoint(lora_layers, Path(lora_checkpoint))
    print(f"Loaded LoRA from {lora_checkpoint}")
```

**Approach 2: Separate Adapter**
```python
# Create iclight_sd15_fc_lora.py
# Same as iclight_sd15_fc.py but with LoRA injection
# Set different port (8003) and environment variable for LoRA path
```

**Approach 3: API Parameter**
```python
# Modify GenerateRequest in iclight_sd15_fc.py
class GenerateRequest(BaseModel):
    image_path: str
    prompt: str
    # ... existing fields ...
    lora_checkpoint: Optional[str] = None  # New field
```

---

## 🔌 Integration Points

### Connection 1: AdaptVPR → SALAD Training

**From**: `AdaptVPR/run.py` output (generation manifest + images)

**To**: `train_salad.py` input (real + synthetic data)

**Bridge**: Need to parse AdaptVPR manifest and create dataset

```python
# In train_salad.py
def load_synthetic_data(manifest_path: Path):
    """Load synthetic images from AdaptVPR manifest."""
    records = load_jsonl(manifest_path.parent / "records")
    
    synthetic_images = []
    for record in records:
        if record.get("passed") and record.get("eligible_for_training"):
            source_path = record["source_path"]
            generated_path = record["output_path"]
            synthetic_images.append({
                "source": source_path,
                "generated": generated_path,
                "route": record["route"],
                "prompt": record.get("prompt"),
            })
    
    return synthetic_images
```

### Connection 2: SALAD Evaluation → Hard Cases

**From**: `evaluate_salad.py` output (error cases JSON)

**To**: `extract_hard_cases.py` input

**Bridge**: Already implemented in `extract_hard_cases.py`, just need matching format

### Connection 3: Hard Cases → LoRA Training

**From**: `extract_hard_cases.py` output (hard_cases.json)

**To**: `finetune_generator.py` input

**Bridge**: Already implemented, just need valid source_path mapping

```python
# In finetune_generator.py, improve source path resolution
def resolve_source_path(case: HardCase, gsv_root: Path) -> Path | None:
    """Find source image in GSV-Cities."""
    source_id = case.source_id or case.query_id
    
    # Try multiple patterns
    patterns = [
        gsv_root / "Images" / source_id,
        gsv_root / "Images" / source_id.replace("_", "/"),
        # Add more patterns as needed
    ]
    
    for path in patterns:
        if path.exists():
            return path
    
    return None
```

### Connection 4: LoRA → Generation

**From**: `finetune_generator.py` output (LoRA checkpoint)

**To**: AdaptVPR generation with LoRA loaded

**Bridge**: See "Integration with AdaptVPR Generation Pipeline" above

---

## 📋 Complete Workflow Checklist

To run the full iterative pipeline:

### Round 0: Baseline

- [ ] Generate synthetic data with vanilla IC-Light
  - `python AdaptVPR/run.py ...` (existing)
  
- [ ] Train SALAD on real + synthetic
  - `python train_salad.py ...` (**TODO**)
  
- [ ] Evaluate SALAD on validation sets
  - `python evaluate_salad.py --dataset SVOX ...` (**TODO**)
  - `python evaluate_salad.py --dataset Nordland ...` (**TODO**)
  - `python evaluate_salad.py --dataset RobotCar ...` (**TODO**)
  
- [ ] Extract and merge hard cases
  - `python extract_hard_cases.py merge ...` (✅ implemented)

### Round 1: First LoRA

- [ ] Fine-tune LoRA on hard cases
  - `python finetune_generator.py ...` (✅ implemented)
  
- [ ] Generate synthetic data with LoRA
  - `python AdaptVPR/run.py ... --lora-checkpoint ...` (**TODO**: add LoRA support)
  
- [ ] Train SALAD on real + new synthetic
  - `python train_salad.py ...` (**TODO**)
  
- [ ] Evaluate and extract hard cases
  - Same as Round 0

### Round 2+: Iteration

- Repeat Round 1 steps with updated LoRA

---

## 🎯 Priority Implementation Order

### P0 (Critical Path)
1. **SALAD training script** - Without this, can't evaluate if the approach works
2. **SALAD evaluation script** - Need hard cases to fine-tune LoRA
3. **LoRA support in AdaptVPR generation** - Need to use fine-tuned LoRA

### P1 (Important)
4. **Source path resolution** - Better heuristics for finding GSV source from query_id
5. **Manifest compatibility** - Ensure AdaptVPR output works with SALAD input
6. **Checkpoint management** - Clean interface for saving/loading between rounds

### P2 (Nice to Have)
7. **Visualization tools** - Plot loss curves, visualize hard cases, compare images
8. **Hyperparameter tuning** - Grid search for lambda weights, LoRA rank
9. **Multi-GPU support** - Distribute training across GPUs

---

## 🧪 Testing Strategy

After implementing each component:

1. **Unit test**: Does it run without errors on minimal input?
2. **Smoke test**: Does it produce reasonable output on small dataset?
3. **Integration test**: Does output from A work as input to B?
4. **Full pipeline test**: Can you run one complete round end-to-end?

For the full pipeline test:
- Use 10 source images
- Generate 10 synthetic images (Round 0)
- Train SALAD for 5 epochs (sanity check)
- Evaluate on 50 validation queries (mock dataset)
- Extract ~5 hard cases
- Fine-tune LoRA for 50 steps
- Generate 10 new synthetic images (Round 1)
- Compare SALAD_0 vs SALAD_1 recall

If all steps complete and SALAD_1 ≈ SALAD_0, that's acceptable for initial test. Real improvement needs full-scale runs.

---

## 📝 Documentation TODOs

- [ ] Add example SALAD evaluation output format to docs
- [ ] Document GSV-Cities path resolution logic
- [ ] Add troubleshooting section for common integration issues
- [ ] Create diagram of complete data flow
- [ ] Write comparison section: vanilla vs LoRA generation quality

---

## 🤝 External Dependencies

Components that depend on external code:

1. **SALAD implementation**: Need to adapt from https://github.com/serizba/salad
2. **VPR datasets**: SVOX, Nordland, RobotCar loaders
3. **GSV-Cities loader**: Parse official dataframes
4. **IC-Light weights**: Download from HuggingFace
5. **DINOv2 weights**: Load from torch.hub or local

Make sure all external dependencies are documented in main README.
