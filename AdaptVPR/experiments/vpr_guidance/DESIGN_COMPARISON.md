# Design Comparison: New vs. Previous Approach

## Previous Approach (What We Diagnosed)

### Architecture
```
Source Image
    ↓
IC-Light + LoRA
    ↓
Noisy Latent z_t
    ↓
[Pass A: no_grad]
    UNet → x0_pred → VAE decode → Image → Frozen SALAD Teacher
    ↓
    Compute ∇(L_vpr + L_keep) w.r.t. x0
    ↓
    Detach gradient
    ↓
[Pass B: with grad]
    UNet → x0_train → VJP proxy loss
    ↓
    Combined: λ_diff * L_diff + L_proxy
    ↓
LoRA Parameters Update
```

### Loss Components
1. **L_diff**: MSE(noise_pred, noise) - ✓ Good
2. **L_vpr**: 1 - cos(SALAD_teacher(generated), SALAD_teacher(source)) - ✗ Problematic
3. **L_keep**: L1(generated_image, target_image) - ✗ Unnecessary constraint

### Issues Identified

#### C.1 — Teacher/Fresh SALAD Representation Gap (Critical)
- **Problem**: Frozen teacher SALAD has pretrained aggregator
- **Downstream**: Fresh SALAD has random aggregator
- **Impact**: Optimizing teacher cosine similarity ≠ helping fresh SALAD learn
- **Evidence**: Teacher and fresh descriptors can be uncorrelated

#### C.2 — Per-Image vs. Batch Metric Learning (Critical)
- **Problem**: LoRA optimizes individual image similarity to source
- **Reality**: SALAD learns from batch-level hard pairs (MultiSimilarityLoss)
- **Impact**: Images good for teacher may be useless for batch metric learning

#### D.1 — Bilevel Objective Mismatch (Fundamental)
- **True goal**: max Performance(SALAD trained on G_LoRA(sources), test_domain)
- **Actual**: min 1 - cos(SALAD_teacher(G_LoRA(source)), SALAD_teacher(source))
- **Gap**: One-level surrogate for two-level optimization

#### Implementation Complexity
- Two-pass VJP with detached gradients
- VAE gradient through decode (correct but complex)
- SALAD input gradient patching
- High coupling between teacher and generator

---

## New Approach (This Implementation)

### Architecture
```
SALAD Validation → Hard Cases (discrete signal)
    ↓
Hard Case Sources (GSV-Cities)
    ↓
IC-Light + LoRA
    ↓
Standard Diffusion Training with Three Losses:
    1. L_diff: preserve diffusion prior
    2. L_identity: maintain place identity (DINO/CLIP)
    3. L_diverse: prevent collapse
    ↓
LoRA Parameters Update
    ↓
Generate New Images → Train Fresh SALAD → Evaluate → Repeat
```

### Loss Components
1. **L_diff**: MSE(noise_pred, noise)
   - Same as before, good for maintaining generation quality
   
2. **L_identity**: 1 - cos(DINO(generated), DINO(source))
   - Uses stable pretrained features (DINO or CLIP)
   - Not tied to any specific SALAD instance
   - Ensures semantic place identity preservation
   
3. **L_diverse**: -mean_pairwise_distance(generated_images)
   - Prevents mode collapse
   - Can compare against previous generation pool
   - No SALAD gradient needed

### How It Solves Previous Issues

#### Solves C.1 (Representation Gap)
- ✓ **No frozen teacher as loss target**
- ✓ Uses DINO/CLIP: stable across SALAD training runs
- ✓ SALAD only identifies which places need better data (discrete)
- ✓ LoRA learns general identity preservation, not teacher-specific features

#### Solves C.2 (Batch Learning)
- ✓ **No per-image optimization**
- ✓ Hard cases tell us "which places" not "what descriptor"
- ✓ Diversity loss encourages varied augmentations
- ✓ Downstream SALAD picks informative pairs from the pool naturally

#### Solves D.1 (Bilevel Objective)
- ✓ **Iterative co-training** approximates bilevel optimization
- ✓ Round k: LoRA → Generated → Train SALAD_k → Validate → Extract hard cases
- ✓ Round k+1: Use SALAD_k errors to guide LoRA_k+1
- ✓ Objective is implicit: improve worst-case SALAD performance

#### Implementation Simplicity
- ✓ No two-pass VJP
- ✓ No SALAD gradient flow
- ✓ Standard diffusion training loop
- ✓ Clean separation: SALAD evaluates, LoRA improves

---

## Key Philosophical Difference

### Previous: "Generate images that the teacher SALAD likes"
- Optimize: cos(SALAD_teacher(gen), SALAD_teacher(src))
- Risk: Teacher's preferences ≠ fresh SALAD's needs
- Failure mode: Reward hacking, teacher overfitting

### New: "Generate better augmentations for hard places"
- Optimize: identity + diversity on hard case sources
- Signal: SALAD tells us which places are hard (not what to generate)
- Benefit: Decoupled from any specific SALAD representation

---

## Analogy

### Previous Approach
Like a student (LoRA) trying to write essays (generated images) that maximize a specific teacher's (frozen SALAD) grading score. But the final exam is graded by a different teacher (fresh SALAD) with different preferences.

### New Approach
Like a tutor (SALAD) identifying which topics (places) the student (LoRA) struggles with, and the student practicing those topics with general study methods (identity + diversity losses). No attempt to game the tutor's grading rubric.

---

## When Would Each Approach Work?

### Previous Approach Could Work If:
1. Teacher SALAD and fresh SALAD have similar representations
   - Requires: both use same pretrained aggregator
   - Reality: fresh SALAD uses random aggregator init
   
2. Per-image cosine similarity correlates with batch utility
   - Requires: SALAD learns from independent samples
   - Reality: SALAD uses hard-pair mining in batches
   
3. Computational resources allow full gradient chain
   - Two-pass VJP is correct but expensive

### New Approach Works When:
1. Hard cases are identifiable from validation
   - Only need: SALAD evaluation on val sets
   - Robust: doesn't depend on teacher representation
   
2. Identity + diversity are good proxies for augmentation quality
   - DINO/CLIP capture place identity
   - Diversity prevents collapse
   - Both are well-validated in literature
   
3. Iterative improvement is acceptable
   - Benefits accumulate over rounds
   - Each round is independent and interpretable

---

## Experimental Validation Strategy

To empirically compare:

### Baseline (A): Real data only
- Train SALAD on GSV-Cities
- Evaluate on SVOX/Nordland/RobotCar

### Previous Approach (B): Teacher-guided LoRA
- Train with frozen SALAD teacher
- Optimize L_diff + L_vpr + L_keep
- Generate synthetic data
- Train fresh SALAD on real + synthetic
- Evaluate

### New Approach (C): Hard-case-guided LoRA
- Round 0: Generate with vanilla IC-Light
- Train SALAD_0, extract hard cases
- Round 1: Fine-tune LoRA on hard cases
- Generate with LoRA_1
- Train SALAD_1, evaluate

### Expected Outcomes

If **C.1, C.2, D.1 are real issues**:
- B ≈ A (no improvement or even regression)
- C > A (iterative improvement works)
- C > B (new approach solves the mismatch)

If **previous approach was actually correct**:
- B > A (teacher proxy works)
- C ≈ A (no benefit from hard case signal)
- B ≥ C (direct optimization better than iterative)

---

## Computational Cost Comparison

### Previous Approach (per training step)
- UNet forward (Pass A): no_grad
- VAE decode: small grad for x0_leaf
- SALAD forward: no grad on params, yes grad on input
- VJP computation: autograd.grad
- UNet forward (Pass B): full grad
- Combined backward: LoRA parameters
- **Total**: ~2x UNet forward + 1x VAE + 1x SALAD per step

### New Approach (per training step)
- UNet forward: full grad
- VAE decode: no grad (inference only)
- DINO/CLIP forward: no grad (frozen feature extractor)
- Combined loss backward: LoRA parameters
- **Total**: 1x UNet forward + 1x DINO per step

### Memory
- Previous: Need to store intermediate activations for VJP
- New: Standard diffusion training memory footprint

### Training Time (estimate for 1000 steps, 512x512)
- Previous: ~3-4 hours (V100)
- New: ~1-2 hours (V100)

---

## Recommendations

### Use Previous Approach If:
- You have a pretrained SALAD with fixed aggregator (not random init)
- You want to optimize for that specific SALAD instance
- You need single-round optimization (no iteration)
- Computational resources are abundant

### Use New Approach If:
- Fresh SALAD uses random aggregator initialization
- You want a simpler, more interpretable pipeline
- Iterative improvement over rounds is acceptable
- You want to decouple generator training from VPR model specifics
- You have validation sets for hard case extraction

---

## Future Hybrid Approach

Could combine both:

1. Use new approach for **which places** to augment (hard cases)
2. Use previous approach's **contrastive signal** (correct vs. wrong matches)
3. Replace frozen teacher with **current round's SALAD**
4. Keep **identity + diversity** as regularizers

```python
# Hybrid loss
L = λ_diff * L_diff + \
    λ_identity * L_identity(gen, source) + \
    λ_diverse * L_diverse(gen, pool) + \
    λ_contrastive * (
        distance(SALAD_current(gen), SALAD_current(wrong_match)) -
        distance(SALAD_current(gen), SALAD_current(correct_match))
    )
```

This would require SALAD gradient but avoid teacher/fresh mismatch by using the live SALAD from the current round.

---

## Conclusion

The new approach trades:
- **Gives up**: Direct optimization of SALAD descriptor similarity
- **Gains**: Robustness to representation mismatch, simpler implementation, interpretability

The bet is that **general identity + diversity** on **SALAD-identified hard cases** is a better proxy for downstream VPR utility than **teacher SALAD cosine similarity** on all training data.

Empirical validation (A/B/C experiment) will tell us which hypothesis is correct.
