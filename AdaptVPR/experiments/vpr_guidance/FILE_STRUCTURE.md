# File Structure Overview

## 📁 Created Files (14 total)

### Core Implementation (5 files)

1. **`hard_cases.py`** (117 lines)
   - Load and manage hard cases from SALAD validation
   - Filter cases with available source images
   - Save summaries for inspection
   - **Status**: ✅ Complete

2. **`losses.py`** (246 lines)
   - `DiffusionLoss`: MSE for denoising
   - `IdentityLoss`: DINO/CLIP for place identity
   - `DiversityLoss`: Pairwise distance to prevent collapse
   - **Status**: ✅ Complete

3. **`lora_utils.py`** (298 lines)
   - LoRA layer implementation
   - Inject LoRA into UNet
   - Save/load checkpoints (safetensors)
   - Parameter management and reporting
   - **Status**: ✅ Complete

4. **`finetune_generator.py`** (346 lines)
   - Main training script for LoRA
   - Loads hard cases, trains on source images
   - Optimizes L_diff + L_identity + L_diverse
   - Saves checkpoints periodically
   - **Status**: ✅ Complete

5. **`__init__.py`** (27 lines)
   - Package initialization
   - Exports main classes and functions
   - **Status**: ✅ Complete

### Utilities (2 files)

6. **`extract_hard_cases.py`** (165 lines)
   - Extract hard cases from SALAD evaluation results
   - Merge hard cases from multiple datasets
   - Two commands: `extract` and `merge`
   - **Status**: ✅ Complete

7. **`run_iterative_pipeline.py`** (260 lines)
   - End-to-end example script
   - Demonstrates full iterative training workflow
   - Orchestrates all components (with TODOs for missing parts)
   - **Status**: ✅ Complete (template)

### Testing (1 file)

8. **`test_implementation.py`** (237 lines)
   - Unit tests for all components
   - Hard case loading test
   - Loss computation tests
   - LoRA injection/save/load tests
   - Training step smoke test
   - **Status**: ✅ Complete

### Documentation (6 files)

9. **`README.md`** (545 lines)
   - Complete documentation of the approach
   - Design philosophy and differences from previous
   - Usage instructions
   - Hyperparameter guide
   - Troubleshooting section
   - **Status**: ✅ Complete

10. **`QUICKSTART.md`** (226 lines)
    - Minimal working example
    - Step-by-step quick start guide
    - Common issues and solutions
    - Debugging checklist
    - **Status**: ✅ Complete

11. **`DESIGN_COMPARISON.md`** (447 lines)
    - Detailed comparison: new vs. previous approach
    - Explains why previous approach had issues
    - Shows how new approach solves them
    - When to use which approach
    - **Status**: ✅ Complete

12. **`INTEGRATION_TODOS.md`** (420 lines)
    - Lists what still needs to be implemented
    - SALAD training script (P0)
    - SALAD evaluation script (P0)
    - LoRA support in IC-Light adapter (P0)
    - Complete workflow checklist
    - **Status**: ✅ Complete

13. **`IMPLEMENTATION_SUMMARY.md`** (534 lines)
    - High-level overview of what was implemented
    - Core design thinking
    - What you still need to do
    - Experimental validation strategy
    - **Status**: ✅ Complete (this document)

### Dependencies (1 file)

14. **`requirements.txt`** (10 lines)
    - Python package dependencies
    - torch, diffusers, transformers, etc.
    - **Status**: ✅ Complete

---

## 📊 Statistics

- **Total lines of code**: ~2,800 lines
- **Python files**: 8
- **Documentation files**: 6
- **Implementation status**: Core complete, integration needed
- **Test coverage**: All core components tested

---

## 🎯 What's Done vs. What's Needed

### ✅ Fully Implemented (Ready to Use)

1. Hard case loading and filtering
2. Three loss functions (diffusion, identity, diversity)
3. LoRA injection into UNet
4. LoRA checkpoint save/load (safetensors)
5. Fine-tuning training loop
6. Hard case extraction from SALAD eval
7. Test suite for all components
8. Complete documentation

### ⚠️ Integration Needed (You Must Implement)

1. **SALAD training script** - Train fresh SALAD on real+synthetic
2. **SALAD evaluation script** - Evaluate and output error cases
3. **LoRA support in IC-Light** - Load LoRA checkpoint before generation

### 📦 External Dependencies (Already Exist)

1. AdaptVPR generation pipeline ✓
2. IC-Light adapter ✓
3. GSV-Cities dataset ✓
4. Validation datasets (SVOX, Nordland, RobotCar) ✓

---

## 🚀 How to Use This Implementation

### Step 1: Verify Installation
```bash
cd AdaptVPR/experiments/vpr_guidance
pip install -r requirements.txt
python test_implementation.py
```

### Step 2: Read Documentation
- Start with `IMPLEMENTATION_SUMMARY.md` (this file)
- Read `README.md` for detailed design
- Check `QUICKSTART.md` for quick example
- Review `INTEGRATION_TODOS.md` for what's needed

### Step 3: Implement Missing Pieces
Priority order:
1. SALAD training script (can adapt from SALAD repo)
2. SALAD evaluation script (output hard_cases.json format)
3. Add LoRA support to IC-Light adapter (5-10 lines)

### Step 4: Run End-to-End Test
```bash
# Use run_iterative_pipeline.py as template
# Fill in TODOs with actual script calls
python run_iterative_pipeline.py \
  --gsv-root /path/to/Gsvcities \
  --base-model /path/to/sd15 \
  --output-root ./outputs/test \
  --num-rounds 2 \
  --samples-per-round 100
```

### Step 5: Full-Scale Experiment
Once small-scale test works:
- Increase to 1000+ samples per round
- Run 3-5 rounds
- Compare baseline vs LoRA-guided generation
- Analyze recall@1/5/10 on validation sets

---

## 🎓 Key Design Decisions

### 1. No SALAD Gradient
**Decision**: Don't flow SALAD gradients through diffusion

**Reason**: Avoids teacher/fresh representation mismatch (C.1)

**Tradeoff**: Can't directly optimize SALAD loss, but gain robustness

### 2. DINO/CLIP for Identity
**Decision**: Use DINO or CLIP instead of SALAD for identity loss

**Reason**: Stable across training runs, not tied to specific SALAD instance

**Tradeoff**: Not VPR-specific, but that's actually a feature

### 3. Hard Cases as Discrete Signal
**Decision**: SALAD identifies "which places", not "what to generate"

**Reason**: Avoids per-image optimization mismatch (C.2)

**Tradeoff**: Need validation sets and SALAD evaluation

### 4. Iterative Co-training
**Decision**: Round-by-round improvement, not single-shot

**Reason**: Approximates bilevel optimization (D.1)

**Tradeoff**: Takes multiple rounds, but each is interpretable

### 5. Simple Losses
**Decision**: Standard diffusion + frozen feature extractors

**Reason**: Stable, well-understood, easy to tune

**Tradeoff**: Less direct than SALAD gradient, but more robust

---

## 📈 Expected Results

### If Method Works:
- Round 0 (vanilla): R@1 = 85%
- Round 1 (first LoRA): R@1 = 86-87%
- Round 2 (second LoRA): R@1 = 87-88%
- Diminishing returns after 3-4 rounds

### If Method Doesn't Work:
- All rounds similar: R@1 ≈ 85%
- Means: hard case signal not informative, or loss design wrong

### If Method Harms:
- LoRA rounds worse: R@1 < 85%
- Means: LoRA breaking generation quality, check lambda_diff

---

## 🔧 Debugging Guide

If results are bad:

1. **Check LoRA is updating**
   - Run test_implementation.py
   - Verify gradient flow in finetune_generator.py
   - Check loss_diff and loss_identity are reasonable

2. **Check generated images look good**
   - Visualize samples from LoRA-generated pool
   - Compare to vanilla IC-Light
   - Should be realistic, diverse, preserve identity

3. **Check hard cases are meaningful**
   - Inspect hard_cases.json
   - Are these actually hard for VPR?
   - Do they have valid source images?

4. **Check SALAD training is fair**
   - Same hyperparameters for all rounds
   - Same real/synthetic mix ratio
   - Same random seed for aggregator init

5. **Compare A/B/C baselines**
   - A (real only) should be worst
   - B (vanilla IC-Light) should beat A
   - C (LoRA) should beat B (if method works)

---

## 📞 Support

For issues or questions:

1. Check relevant documentation file
2. Run test suite to isolate problem
3. Review INTEGRATION_TODOS.md for missing pieces
4. Compare against DESIGN_COMPARISON.md for conceptual clarity

---

## 🎉 Success Criteria

You'll know the implementation is working when:

- [ ] `test_implementation.py` passes all tests
- [ ] `finetune_generator.py` trains without errors
- [ ] Loss curves are stable (not exploding/NaN)
- [ ] Generated images look realistic
- [ ] LoRA checkpoint can be loaded and used
- [ ] Integration with AdaptVPR generation works
- [ ] SALAD training on LoRA-generated data completes
- [ ] Round 1 SALAD recall ≥ Round 0 SALAD recall
- [ ] Hard cases decrease or change over rounds

---

## 📝 Citation

If this implementation helps your research:

```bibtex
@misc{adaptvpr_lora_guidance,
  title={Hard-Case-Driven LoRA Fine-tuning for VPR Data Generation},
  author={Your Name},
  year={2026},
  note={Built on AdaptVPR framework}
}
```

---

## License

This implementation follows the same license as AdaptVPR.

---

**End of Implementation** - All core components ready. Integration points documented. Good luck with experiments! 🚀
