# Code Style and Conventions

## Naming Conventions

### Classes
- **PascalCase**: `GaussianModel`, `NeuSSystem`, `BaseImplicitGeometry`
- Descriptive names indicating purpose

### Functions/Methods
- **snake_case**: `training_step`, `prefilter_voxel`, `get_rays`
- Verb-based names for actions

### Variables
- **snake_case**: `gs_depth`, `picked_gs_depth_dt`, `current_epoch_set`
- Private attributes prefixed with underscore: `_anchor`, `_offset`, `_scaling`

### Constants
- **UPPER_SNAKE_CASE**: `TENSORBOARD_FOUND` (rare, typically in module scope)

### Files/Directories
- **snake_case**: `gaussian_model.py`, `training_step`, `instant_nsr/`
- Config files: descriptive names like `barn.yaml`, `bicycle.yaml`

## Import Organization

Group imports in three sections (standard → third-party → local):

```python
# Standard library
import os
import sys
import json
from pathlib import Path

# Third-party
import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl

# Local modules
from gaussian_splatting.scene import Scene, GaussianModel
import instant_nsr.models
import instant_nsr.systems
from instant_nsr.utils.misc import load_config
```

**Note**: Gaussian and SDF imports often mixed in joint training files (e.g., `instant_nsr/systems/neus.py`)

## Code Structure Patterns

### Dual-Branch Architecture
- **GS branch**: Uses custom training loops with `tqdm` progress bars
- **SDF branch**: Uses PyTorch Lightning's `training_step()` pattern
- **Integration**: Both trained in `instant_nsr/systems/neus.py:NeuSSystem.training_step()`

### Configuration Access
Configs are OmegaConf objects accessed via dot notation:
```python
self.config.model.if_gaussian
self.config.system.loss.normal_w
self.config.dataset.neuralangelo_scale
```

### GPU Auto-Selection Pattern
All entry scripts (`train.py`, `launch.py`, `render.py`) use:
```python
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
```

## Type Hints and Documentation

### Type Hints
- **Sparse usage**: Not consistently used throughout the codebase
- Present in some function signatures: `def __init__(self, args: ModelParams, gaussians: GaussianModel, ...)`
- Tensor types not explicitly annotated

### Docstrings
- **Minimal**: Most functions lack comprehensive docstrings
- Class-level comments sometimes present:
  ```python
  class NeuSSystem(BaseSystem):
      """
      Two ways to print to console:
      1. self.print: correctly handle progress bar
      2. rank_zero_info: use the logging module
      """
  ```
- Critical sections have inline comments explaining algorithm details

## Loss Functions and Weights

### Loss Weight Access
```python
# From config
self.config.system.loss.normal_w
self.config.system.loss.depth_w

# Typical pattern: reduce weights after warm-up
if self.current_epoch_set > 15000:
    self.config.system.loss.normal_w = self.config.system.loss.normal_w / 10
```

### Loss Composition
```python
# GS branch
loss_gaussian = l1_loss + ssim_loss + normal_loss + depth_loss

# SDF branch
loss_sdf = rgb_loss + eikonal_loss + curvature_loss

# Combined (in joint training)
total_loss = loss_gaussian + loss_sdf + mutual_loss
```

## File Organization

### Project Structure
```
HMGS/
├── gaussian_splatting/      # GS-branch code (Scaffold-GS based)
│   ├── scene/              # Scene representation, cameras, Gaussians
│   ├── utils/              # GS-specific utilities
│   ├── gaussian_renderer/  # CUDA rasterization
│   └── arguments/          # Argument parsers
├── instant_nsr/            # SDF-branch code (Instant-NSR based)
│   ├── models/             # SDF geometry, texture, NeRF models
│   ├── systems/            # PyTorch Lightning training systems
│   ├── datasets/           # Data loaders
│   └── utils/              # SDF-specific utilities
├── configs/                # YAML config files per dataset/scene
├── output/                 # GS branch experiment outputs
└── exp/                    # SDF branch experiment outputs
```

### Output Locations
- **GS branch**: `output/{scene_name}/{tag}/`
  - Point clouds, checkpoints, logs
- **SDF branch**: `exp/{scene_name}/{trial_name}/`
  - Checkpoints in `ckpt/`, configs in `config/`, TensorBoard in parent
- **Rendering results**: Saved in model output directories

## Registration Patterns

### Model Registration
Uses decorator-based registration:
```python
from instant_nsr.models import register

@register('volume-sdf-sg')
class VolumeSDF(BaseImplicitGeometry):
    ...
```

Accessed via:
```python
import instant_nsr.models
model = instant_nsr.models.make('volume-sdf-sg', config)
```

## Common Code Patterns

### Detached Tensors for Cross-Branch Communication
```python
# Avoid gradient flow between branches
picked_gs_depth_dt = picked_gs_depth.detach()
out = self(batch, picked_gs_depth_dt, use_depth_guide=True)
```

### CUDA Event Timing
```python
iter_start = torch.cuda.Event(enable_timing=True)
iter_end = torch.cuda.Event(enable_timing=True)
iter_start.record()
# ... operations ...
iter_end.record()
torch.cuda.synchronize()
elapsed = iter_start.elapsed_time(iter_end)
```

### Scene Normalization
Both branches must use same normalization:
```python
Scene(args, gaussians, given_scale=config.dataset.neuralangelo_scale,
      given_center=config.dataset.neuralangelo_center)
```

## Testing and Validation

### No Explicit Test Suite
- No `tests/` directory or unit tests
- Validation via rendering on test views during training
- Metrics computed automatically at specified iterations

### Quality Assurance
- Visual inspection of rendered images
- Quantitative metrics: PSNR, SSIM, LPIPS
- Geometry evaluation: Chamfer Distance (external tools)

## Development Guidelines

1. **Maintain dual-branch compatibility**: Changes to GS or SDF should preserve integration points
2. **Respect config hierarchy**: Use YAML configs, avoid hardcoded parameters
3. **Preserve scene normalization**: Ensure both branches see same coordinate system
4. **Detach gradients** when passing data between branches
5. **Follow existing patterns**: Use registration for new models, match logging conventions
6. **Document critical sections**: Add comments for non-obvious algorithmic choices
