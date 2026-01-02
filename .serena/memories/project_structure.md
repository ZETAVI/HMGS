# Project Structure

## Root Directory Layout

```
HMGS/
├── .github/              # GitHub-specific files (Actions, Copilot instructions)
├── .serena/              # Serena memory files (project context for AI agents)
├── .vscode/              # VS Code workspace settings
├── assets/               # Paper figures, visualizations
├── configs/              # YAML configuration files per dataset/scene
├── gaussian_splatting/   # GS-branch implementation (Scaffold-GS based)
├── instant_nsr/          # SDF-branch implementation (Instant-NSR based)
├── papers/               # Research papers (.gitignored)
├── data/                 # Input datasets (.gitignored)
├── output/               # GS-branch outputs (.gitignored)
├── exp/                  # SDF-branch outputs (.gitignored)
├── runs/                 # TensorBoard logs (.gitignored)
│
├── __init__.py           # Package marker
├── environment.yml       # Conda environment specification
├── launch.py             # Main training launcher (PyTorch Lightning)
├── train.py              # Alternative training script (GS-focused)
├── train.sh              # Bash wrapper for training
├── render.py             # Post-training rendering script
├── metrics.py            # Evaluation metrics computation
├── LICENSE.md            # License information
└── README.md             # Project documentation
```

## GS Branch (`gaussian_splatting/`)

Implements explicit 3D Gaussian Splatting (based on Scaffold-GS):

```
gaussian_splatting/
├── __init__.py
├── arguments/            # Command-line argument parsers
│   └── __init__.py       # ModelParams, PipelineParams, OptimizationParams
├── scene/                # Scene representation and Gaussian models
│   ├── __init__.py       # Scene class (loads data, manages cameras)
│   ├── cameras.py        # Camera class for rendering
│   ├── colmap_loader.py  # COLMAP data parsing
│   ├── dataset_readers.py # Dataset loading utilities
│   └── gaussian_model.py # GaussianModel (anchor-offset structure)
├── gaussian_renderer/    # CUDA-accelerated rendering
│   ├── __init__.py       # render(), prefilter_voxel() functions
│   └── network_gui.py    # (Optional) GUI server interface
├── lpipsPyTorch/         # LPIPS perceptual loss
│   ├── __init__.py
│   └── modules/          # LPIPS network definitions
├── utils/                # GS-specific utilities
│   ├── loss_utils.py     # L1, SSIM losses
│   ├── image_utils.py    # Image processing, PSNR
│   ├── graphics_utils.py # Rotation, covariance utilities
│   ├── camera_utils.py   # Camera transformations
│   ├── general_utils.py  # Misc utilities (safe_state, learning rate)
│   └── ...               # Additional utilities
└── submodules/           # CUDA extensions (must be built)
    ├── diff-gaussian-rasterization/  # Tile-based rasterizer
    └── simple-knn/                   # K-nearest neighbors
```

**Key Files:**
- `scene/gaussian_model.py`: Core Gaussian primitive representation
- `gaussian_renderer/__init__.py`: Rendering pipeline with CUDA backend
- `arguments/__init__.py`: All hyperparameters for GS branch

## SDF Branch (`instant_nsr/`)

Implements implicit SDF representation (based on Instant-NSR):

```
instant_nsr/
├── __init__.py
├── models/               # Neural network models
│   ├── __init__.py       # Model factory (make, register decorators)
│   ├── base.py           # BaseModel class
│   ├── geometry.py       # SDF geometry models (VolumeSDF, etc.)
│   ├── texture.py        # Texture/radiance field models
│   ├── neus.py           # NeuS model (SDF + volume rendering)
│   ├── nerf.py           # NeRF baseline model
│   ├── network_utils.py  # MLP, encoding utilities
│   ├── ray_utils.py      # Ray generation, sampling
│   └── utils.py          # Model utilities
├── systems/              # PyTorch Lightning training systems
│   ├── __init__.py       # System factory
│   ├── base.py           # BaseSystem (Lightning module)
│   ├── neus.py           # NeuSSystem (dual-branch training!)
│   ├── nerf.py           # NeRF training system
│   ├── criterions.py     # Loss functions (PSNR, etc.)
│   └── utils.py          # Training utilities
├── datasets/             # Data loaders
│   ├── __init__.py       # Dataset factory
│   ├── colmap.py         # COLMAP dataset loader
│   ├── blender.py        # Blender synthetic dataset
│   ├── dtu.py            # DTU dataset
│   ├── colmap_utils.py   # COLMAP parsing utilities
│   └── utils.py          # Dataset utilities
└── utils/                # SDF-specific utilities
    ├── callbacks.py      # PyTorch Lightning callbacks
    ├── misc.py           # Config loading, misc utilities
    └── ...               # Additional utilities
```

**Key Files:**
- `systems/neus.py`: **CRITICAL** - orchestrates dual-branch training
- `models/geometry.py`: SDF network definitions
- `models/neus.py`: NeuS model (SDF → density → volume rendering)
- `datasets/colmap.py`: Loads same data as GS branch

## Configuration System (`configs/`)

Hierarchical YAML configs per dataset and scene:

```
configs/
├── mipnerf360/       # MipNeRF360 dataset scenes
│   ├── bicycle.yaml
│   ├── bonsai.yaml
│   ├── counter.yaml
│   └── ...
├── tnt/              # Tanks & Temples scenes
│   ├── barn.yaml
│   └── truck.yaml
├── dtu/              # DTU dataset scenes
│   ├── scan24.yaml
│   └── ...
└── db/               # Deep Blending scenes
    ├── db_drjohnson.yaml
    └── db_playroom.yaml
```

**Config Structure:**
- `name`: Scene name
- `tag`: Experiment identifier
- `dataset`: Data loading parameters (paths, normalization)
- `model`: Both GS and SDF model parameters
  - `model.if_gaussian`: Enable/disable GS branch
  - `model.gs_sampling`: Enable depth-guided sampling
  - `model.geometry`: SDF network architecture
  - `model.variance`: Volume rendering variance
- `system.loss`: Loss weights (normal, depth, eikonal, etc.)

## Data Flow

```
Input Data (COLMAP)
    ↓
configs/{dataset}/{scene}.yaml
    ↓
launch.py (entry point)
    ↓
    ├──> instant_nsr.datasets (loads images/cameras)
    └──> instant_nsr.systems (SDF training)
         └──> NeuSSystem.__init__()
              ├──> gaussian_splatting.Scene (GS scene setup)
              └──> gaussian_splatting.GaussianModel (Gaussian primitives)
                   ↓
         NeuSSystem.training_step() [DUAL-BRANCH TRAINING]
              ├──> GS forward: render depth/normal
              └──> SDF forward: volume rendering with depth guidance
                   ↓
         Mutual guidance & joint loss
                   ↓
         Outputs:
         ├──> output/{tag}/ (GS checkpoints, logs)
         └──> exp/{scene}/{trial}/ (SDF checkpoints, logs)
```

## Critical Integration Points

### 1. Joint Training Loop
**File**: `instant_nsr/systems/neus.py:NeuSSystem.training_step()`
- Receives batch from SDF data loader
- Extracts same image index for GS branch
- Renders GS depth → guides SDF ray sampling
- Computes joint loss (RGB + depth + normal)

### 2. Scene Initialization
**Files**: 
- `gaussian_splatting/scene/__init__.py:Scene.__init__()`
- `instant_nsr/systems/neus.py:NeuSSystem.__init__()`

Both branches must see same `given_scale` and `given_center` for coordinate alignment.

### 3. Data Loading
**Files**:
- `instant_nsr/datasets/colmap.py` (SDF branch)
- `gaussian_splatting/scene/dataset_readers.py` (GS branch)

Both load from `data/{dataset}/{scene}/sparse/0/` (COLMAP output).

### 4. Rendering Pipeline
**File**: `gaussian_splatting/gaussian_renderer/__init__.py:render()`
- Called with `out_depth=True, return_normal=True` during training
- Outputs used to guide SDF branch

## Output Directory Structure

### GS Branch: `output/{tag}/`
```
output/{tag}/
├── point_cloud/
│   └── iteration_{N}/
│       └── point_cloud.ply
├── cfg_args
├── cameras.json
└── (TensorBoard logs)
```

### SDF Branch: `exp/{scene}/{trial}/`
```
exp/{scene}/{trial}/
├── ckpt/               # PyTorch Lightning checkpoints
│   ├── epoch=X-step=Y.ckpt
│   └── last.ckpt
├── config/             # Saved config files
│   └── parsed.yaml
├── save/               # Saved outputs (meshes, etc.)
└── (parent: TensorBoard logs)
```

## Entry Points

1. **Main Training**: `launch.py` (recommended)
   - Uses PyTorch Lightning for SDF branch
   - Orchestrates dual-branch training
   - Handles config parsing, logging, checkpointing

2. **Alternative**: `train.py` (legacy, GS-focused)
   - Custom training loop for GS branch
   - Less integrated with SDF branch

3. **Rendering**: `render.py`
   - Post-training novel view synthesis
   - Outputs RGB, depth, normal images

4. **Evaluation**: `metrics.py`
   - Computes PSNR, SSIM, LPIPS on rendered images

## Development Hot Spots

**Adding new GS features**: Start in `gaussian_splatting/scene/gaussian_model.py`
**Adding new SDF features**: Start in `instant_nsr/models/` or `instant_nsr/systems/`
**Modifying training**: Edit `instant_nsr/systems/neus.py:NeuSSystem.training_step()`
**Changing data loading**: Edit `instant_nsr/datasets/colmap.py` and `gaussian_splatting/scene/dataset_readers.py`
**Tuning hyperparameters**: Edit YAML configs in `configs/`
