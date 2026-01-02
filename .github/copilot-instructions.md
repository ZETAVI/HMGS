# GSDF: 3D Gaussian Splatting meets Signed Distance Fields

## Project Overview

GSDF is a **dual-branch neural rendering architecture** combining:
- **3D Gaussian Splatting (3DGS)** - explicit rendering for high-quality visualization (PSNR ~29.38, real-time FPS)
- **Neural Signed Distance Fields (SDF)** - implicit representation for accurate geometry reconstruction

The two branches **mutually guide each other** during training to overcome the traditional rendering quality vs. geometry accuracy tradeoff.

**Critical Insight**: The project has TWO independent training modes:
1. `train.py` - Standalone Scaffold-GS training (generates pretrained models)
2. `launch.py` - **Main method**: Dual-branch joint training via PyTorch Lightning

## Tech Stack

**Core**: Python 3.7.13, PyTorch 1.12.1, CUDA 11.6, PyTorch Lightning 1.9.5

**Critical CUDA Extensions** (must be built before first run):
```bash
pip install gaussian_splatting/submodules/diff-gaussian-rasterization
pip install gaussian_splatting/submodules/simple-knn
```

**Key Libraries**: nerfacc (0.3.3), OmegaConf (2.2.3), PyMCubes, wandb, tensorboard

**Environment**: `conda activate gsdf` (see `environment.yml`)

## Architecture

### Dual-Branch Structure

```
HMGS/
├── gaussian_splatting/   # GS branch (Scaffold-GS based, explicit rendering)
│   ├── scene/           # GaussianModel, Scene, cameras
│   ├── gaussian_renderer/ # CUDA rasterization
│   └── utils/           # GS-specific utilities
├── instant_nsr/         # SDF branch (Instant-NSR based, implicit reconstruction)
│   ├── models/          # SDF geometry, NeuS model (factory pattern via @register)
│   ├── systems/         # PyTorch Lightning training (NeuSSystem)
│   └── datasets/        # Data loaders (COLMAP, DTU, etc.)
├── configs/             # YAML configs per dataset/scene
├── output/              # GS branch outputs (.gitignored)
└── exp/                 # SDF branch outputs (.gitignored)
```

### Critical Integration Point

**File**: [instant_nsr/systems/neus.py](instant_nsr/systems/neus.py#L463-L650) (`NeuSSystem.training_step()`)

This method orchestrates dual-branch training:
1. Receives batch from SDF data loader
2. Extracts same image index for GS branch: `viewpoint_cam = self.scene.getTrainCameras()[batch['used_index']]`
3. Uses same pixel coordinates: `yy = batch['used_y']`, `xx = batch['used_x']`
4. GS renders depth/normal: `render_pkg = gaussian_renderer.render(..., out_depth=True, return_normal=True)`
5. **Always detach** GS outputs before passing to SDF: `picked_gs_depth_dt = picked_gs_depth.detach()`
6. Depth guides SDF ray sampling: `out = self(batch, picked_gs_depth_dt, use_depth_guide=True)`
7. Joint loss: RGB + depth alignment + normal consistency

**Mutual Guidance Mechanisms**:
- **GS → SDF**: Depth-guided ray sampling (activated after 15k iters when `self.current_epoch_set > start_step`)
- **SDF → GS**: Geometry-aware density control (SDF depth/normal supervise Gaussian placement)
- **Bidirectional**: Joint depth and normal supervision aligns both representations

## Training Workflows

### Main Training Command
```bash
python launch.py --config configs/tnt/barn.yaml --gpu 0 --train tag=my_experiment
```

**Key Parameters**:
- `--config`: Scene config path (YAML)
- `--gpu`: GPU ID (`-1` auto-selects idle GPU via `nvidia-smi`)
- `--train/--eval/--test/--validate`: Mode selection
- `tag=<name>`: Experiment identifier (appended to `config.trial_name`)
- `--resume <path>`: Resume from checkpoint
- `--resume_weights_only`: Restore only weights, not training state

### Training Phases
1. **Pre-training (0-15k iters)**: GS trains alone in `pretrain_gs()` ([neus.py:308-377](instant_nsr/systems/neus.py#L308-L377))
2. **Joint training (15k+)**: Both branches train with mutual guidance
3. **Loss weight adjustment**: After 15k, `normal_w` and `depth_w` reduced by 10x ([neus.py:479-481](instant_nsr/systems/neus.py#L479-L481))

### GPU Auto-Selection Pattern
All entry scripts ([train.py](train.py#L15-L18), [launch.py](launch.py#L15-L18), [render.py](render.py)) use:
```python
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
```

### Monitoring
```bash
# TensorBoard logs for both branches
tensorboard --logdir runs/

# Outputs:
# - GS: output/${tag}/
# - SDF: exp/${scene_name}/${trial_name}/
```

## Data Preparation

### Directory Structure
```
data/
├── {dataset_name}/
│   └── {scene_name}/
│       ├── images/          # Input photos
│       └── sparse/0/        # COLMAP SfM output (cameras.bin, points3D.bin)
```

**Supported Datasets**: MipNeRF360, Tanks&Temples, DTU, DeepBlending

### Custom Data with COLMAP
```bash
mkdir -p data/my_dataset/my_scene/images
# 1. Place images in data/my_dataset/my_scene/images/
# 2. Run COLMAP SfM (outside this repo):
colmap feature_extractor --database_path database.db --image_path images/
colmap exhaustive_matcher --database_path database.db
colmap mapper --database_path database.db --image_path images/ --output_path sparse/
# 3. Copy sparse/0/ to data/my_dataset/my_scene/sparse/0/
```

## Configuration System

YAML configs in `configs/{dataset}/{scene}.yaml` control both branches via OmegaConf.

**Critical Parameters**:
```yaml
model:
  if_gaussian: true              # Enable GS branch (set to false for SDF-only)
  gs_sampling: true              # Use depth-guided sampling
  num_samples_per_ray: 1024      # SDF ray samples
  using_pretrain: false          # Load pretrained Scaffold-GS
  using_pretrain_path: output/barn  # Path if using_pretrain=true
  
dataset:
  neuralangelo_scale: 3.14       # Scene normalization (must match for both branches!)
  neuralangelo_center: [0,0,0]   # Scene center (must match!)
  
system:
  loss:
    normal_w: 0.01               # Normal loss weight (reduced 10x after 15k)
    depth_w: 0.01                # Depth loss weight (reduced 10x after 15k)
```

**Access in Code**: `self.config.model.if_gaussian`, `self.config.system.loss.normal_w` (OmegaConf dot notation)

**Factory Pattern**: Models/systems use `@register` decorator:
```python
# In instant_nsr/models/geometry.py
@register('volume-sdf-sg')
class VolumeSDF(BaseImplicitGeometry):
    ...

# In config YAML
model:
  geometry:
    name: volume-sdf-sg  # ← Looked up in models['volume-sdf-sg']
```

Access via: `instant_nsr.models.make('volume-sdf-sg', config)` or `instant_nsr.systems.make('neus-system', config)`

## Code Patterns & Conventions

### Naming
- **Classes**: `PascalCase` (GaussianModel, NeuSSystem, BaseImplicitGeometry)
- **Functions**: `snake_case` (training_step, prefilter_voxel, get_rays)
- **Private attributes**: `_anchor`, `_offset`, `_scaling` (GS learnable parameters)

### Import Organization
```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import torch
import numpy as np
from tqdm import tqdm

# Local (GS and SDF imports often mixed in neus.py)
from gaussian_splatting.scene import Scene, GaussianModel
import instant_nsr.models
import instant_nsr.systems
```

### Cross-Branch Tensor Passing
**Always detach** to avoid gradient conflicts ([neus.py:492](instant_nsr/systems/neus.py#L492)):
```python
picked_gs_depth_dt = picked_gs_depth.detach()  # Detach GS output
out = self(batch, picked_gs_depth_dt, use_depth_guide=True)  # Pass to SDF
```

### Model Registration Pattern
Used throughout `instant_nsr/` for factory pattern:
```python
from instant_nsr.models import register

@register('volume-sdf-sg')
class VolumeSDF(BaseImplicitGeometry):
    ...
```

Access via: `instant_nsr.models.make('volume-sdf-sg', config)` or `instant_nsr.systems.make('neus-system', config)`

## Development Workflows

### Adding New GS Features
1. Modify [gaussian_splatting/scene/gaussian_model.py](gaussian_splatting/scene/gaussian_model.py) (`GaussianModel`)
2. Update `__init__()` to add learnable parameters (follow `_anchor`, `_offset` pattern)
3. Add parameters to optimizer in `training_setup()`
4. Update rendering in [gaussian_renderer/\_\_init\_\_.py](gaussian_splatting/gaussian_renderer/__init__.py) (`render()`) if needed

### Adding New SDF Features
1. Create subclass in [instant_nsr/models/geometry.py](instant_nsr/models/geometry.py)
2. Inherit from `BaseImplicitGeometry`
3. Register with `@register('your-name')`
4. Update config to use new geometry: `model.geometry.name: your-name`

### Modifying Training Loop
**File**: [instant_nsr/systems/neus.py](instant_nsr/systems/neus.py#L463) (`NeuSSystem.training_step()`)
- **Caution**: This coordinates both branches!
- Test with both `if_gaussian=true` and `if_gaussian=false`
- Verify gradient flow doesn't leak between branches (use `.detach()`)

### Tuning Hyperparameters
1. Copy existing config: `cp configs/tnt/barn.yaml configs/tnt/my_scene.yaml`
2. Edit parameters (see "Configuration System" above)
3. Run: `python launch.py --config configs/tnt/my_scene.yaml --gpu 0 --train tag=tuning`

## Evaluation & Rendering

### Manual Post-Training
```bash
# Render test views
python render.py -m <path_to_trained_model>

# Compute metrics (PSNR, SSIM, LPIPS)
python metrics.py -m <path_to_trained_model>
```

### Mesh Extraction (Geometry Evaluation)
- Marching cubes in [instant_nsr/models/geometry.py](instant_nsr/models/geometry.py) (`MarchingCubeHelper`)
- For Chamfer Distance, see: https://github.com/hbb1/2d-gaussian-splatting

## Common Pitfalls & Solutions

### 1. Different Camera Representations
**Issue**: GS uses `Camera` class, SDF uses rays
**Solution**: [neus.py](instant_nsr/systems/neus.py#L523-L526) handles conversion:
```python
viewpoint_cam = self.scene.getTrainCameras()[batch['used_index']]  # GS Camera
# batch['rays'] used for SDF
```

### 2. Scene Normalization Mismatch
**Issue**: Branches have different coordinate systems
**Solution**: Use same `given_scale` and `given_center` ([neus.py:197-200](instant_nsr/systems/neus.py#L197-L200)):
```python
self.scene = Scene(self.lp, self.gaussians, 
                   given_scale=self.config.dataset.neuralangelo_scale,
                   given_center=self.config.dataset.neuralangelo_center)
```

### 3. Gradient Conflicts
**Issue**: Gradients flow between branches unexpectedly
**Solution**: Always detach tensors passed between branches:
```python
picked_gs_depth_dt = picked_gs_depth.detach()
```

### 4. CUDA OOM
**Issue**: Out of memory during training
**Solution**: 
- Reduce `model.train_num_rays` in config (default: 256)
- Lower `dataset.img_downscale` for smaller images
- Use `torch.cuda.empty_cache()` after validation

### 5. Pretrained Model Loading
**Issue**: Checkpoint mismatch errors when `using_pretrain=true`
**Solution**: Ensure path points to valid Scaffold-GS checkpoint ([neus.py:229-236](instant_nsr/systems/neus.py#L229-L236))

### 6. Background Color Inconsistency
**Issue**: Rendering looks different between train/eval
**Solution**: GS uses random background during training, white during eval:
```python
# Training (neus.py:464)
random_background = torch.rand(3).cuda()
# Eval
background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
```

## Performance Targets

- **Rendering**: ~30 FPS on single GPU (after training)
- **Training**: ~15k iters pre-training + 30k joint training
- **Memory**: <16GB GPU for most scenes
- **Quality**: PSNR ~29.38 on MipNeRF360 scenes

## References

- **GS Branch**: Based on [Scaffold-GS](https://github.com/city-super/Scaffold-GS)
- **SDF Branch**: Based on [Instant-NSR](https://github.com/bennyguo/instant-nsr-pl)
- **License**: Follow [3D-GS License](https://github.com/graphdeco-inria/gaussian-splatting)
- **Paper**: [arXiv:2403.16964](https://arxiv.org/abs/2403.16964)

