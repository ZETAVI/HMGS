# Task Completion Checklist

## Before Committing Code

### 1. Verify Training Still Works
```bash
# Quick smoke test on a small scene
python launch.py --config configs/tnt/barn.yaml --gpu 0 --train --eval tag=test
# Should run without errors for at least 100 iterations
```

### 2. Check Integration Points
If you modified:
- **GS branch** → Ensure SDF branch can still receive depth/normal maps
- **SDF branch** → Ensure GS branch can still query SDF values
- **Config structure** → Update both branch configs consistently
- **Data loading** → Verify both branches load same images/cameras

### 3. Preserve Scene Normalization
- Ensure `given_scale` and `given_center` are passed to both branches
- Check that coordinate systems remain aligned

### 4. Gradient Flow
- Verify `.detach()` is used when passing tensors between branches
- No unexpected gradient flow that could break dual optimization

## Code Quality Checks

### 1. No Hardcoded Paths
```bash
# Search for absolute paths
grep -r "/home/" --include="*.py" .
grep -r "/data/" --include="*.py" . | grep -v "# data/"
```

### 2. Config Parameters in YAML
- New hyperparameters should be in config files, not hardcoded
- Use `self.config.<section>.<param>` pattern

### 3. Imports Clean
- No unused imports
- Grouped by: stdlib → third-party → local
- No circular imports

### 4. CUDA Memory Management
- Use `torch.cuda.empty_cache()` after validation
- Check for memory leaks in long training runs
- Verify tensors moved to correct device (`.cuda()`, `.to(device)`)

## Testing Checklist

### 1. Run Training on Test Scene
```bash
# Small test to verify end-to-end pipeline
python launch.py --config configs/tnt/barn.yaml --gpu 0 --train tag=verify
# Monitor for first few iterations, check both branches update
```

### 2. Verify Outputs Created
Check that both output directories are populated:
```bash
ls -lh output/verify/
ls -lh exp/barn/verify/
```

### 3. Rendering Still Works
```bash
# If model was trained:
python render.py -m output/<model_path>
# Should generate images without errors
```

### 4. Metrics Computation
```bash
python metrics.py -m output/<model_path>
# Should output PSNR, SSIM, LPIPS values
```

## Before Pushing to Remote

### 1. Git Status Clean
```bash
git status
# No untracked files that should be committed
# No tracked files in .gitignore (like data/, output/, exp/)
```

### 2. Update .gitignore if Needed
- New output directories
- New cache/temp files
- IDE-specific files

### 3. Commit Message Conventions
```bash
# Use descriptive messages:
git commit -m "Add depth-guided sampling threshold parameter to config"
# Not: git commit -m "fix"
```

### 4. Verify Submodules
```bash
# If you modified CUDA extensions:
git submodule status
# Ensure submodules are at correct commits
```

## Documentation Updates

### 1. Update Copilot Instructions if Architecture Changed
If you modified:
- Key integration points
- Training workflow
- Config structure
- Output locations

Update `.github/copilot-instructions.md`

### 2. Update README for User-Facing Changes
- New command-line arguments
- New dataset support
- Installation requirements changes

### 3. Update Config Examples
If you added new parameters:
- Update at least one example config in `configs/`
- Document parameter meaning in comments

## Performance Verification

### 1. No Significant Slowdown
- Training should maintain similar FPS to baseline
- Check with `nvidia-smi` that GPU utilization is high

### 2. Memory Usage Reasonable
```bash
# Monitor during training:
watch -n 1 nvidia-smi
# Should not exceed available GPU memory
```

### 3. Numerical Stability
- Check TensorBoard logs for NaN/Inf values
- Verify loss curves are smooth, not erratic

## Cleanup Before Final Commit

### 1. Remove Debug Code
```bash
# Search for debug statements
grep -r "import pdb" --include="*.py" .
grep -r "print(" --include="*.py" . | grep -v "logger"
```

### 2. Remove Commented Code
- Clean up large blocks of commented-out code
- Keep only meaningful comments

### 3. Format Consistency
- Consistent indentation (4 spaces for Python)
- No trailing whitespace
- Unix line endings (LF, not CRLF)

## Final Checklist

- [ ] Training works end-to-end
- [ ] Both GS and SDF branches update correctly
- [ ] No hardcoded paths
- [ ] Config parameters properly used
- [ ] Outputs created in correct locations
- [ ] No memory leaks or CUDA errors
- [ ] Git status clean (no unintended files)
- [ ] Commit message descriptive
- [ ] Documentation updated if needed
- [ ] No debug code left behind
- [ ] Performance remains acceptable

## If Modifying CUDA Extensions

- [ ] Rebuilt extensions: `pip install -e gaussian_splatting/submodules/diff-gaussian-rasterization`
- [ ] Tested on target CUDA version (11.6/11.8)
- [ ] Verified no compilation warnings
- [ ] Tested backward pass (if gradients affected)

## If Adding New Model/Dataset Support

- [ ] Config template created in `configs/`
- [ ] Data loader tested with sample data
- [ ] Registration decorator used (`@register('name')`)
- [ ] Example usage documented
- [ ] Tested with both GS and SDF branches
