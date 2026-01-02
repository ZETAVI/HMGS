# Tech Stack

## Core Languages
- **Python 3.7.13** - primary language for all code
- **C++/CUDA** - custom CUDA kernels in submodules for rasterization

## Deep Learning Framework
- **PyTorch 1.12.1** - main DL framework
- **PyTorch Lightning 1.9.5** - training orchestration for SDF branch
- **CUDA 11.6** - GPU acceleration

## Key Libraries

### Neural Rendering
- `nerfacc==0.3.3` - accelerated NeRF/volume rendering utilities
- `torch_efficient_distloss` - distortion loss for NeRF
- Custom CUDA extensions (must be built):
  - `gaussian_splatting/submodules/diff-gaussian-rasterization` - tile-based Gaussian rasterizer
  - `gaussian_splatting/submodules/simple-knn` - KNN for Gaussian initialization

### Geometry Processing
- `PyMCubes==0.1.4` - marching cubes for mesh extraction from SDF
- `pyransac3d` - RANSAC for 3D geometry fitting
- `plyfile==0.8.1` - PLY file I/O for point clouds/meshes

### Configuration & Logging
- `omegaconf==2.2.3` - hierarchical configuration management (YAML configs)
- `wandb` - experiment tracking
- `tensorboard` - training visualization

### Utilities
- `opencv-python` - image processing
- `matplotlib` - visualization
- `imageio`, `imageio-ffmpeg` - video/image I/O
- `scipy` - scientific computing
- `einops` - tensor operations
- `tqdm` - progress bars

## System Requirements
- **OS**: Ubuntu 22.04 (Linux preferred)
- **Compiler**: GCC 11.4.0
- **GPU**: CUDA-capable (NVIDIA)
- **Memory**: Sufficient for dual-branch training (scenes vary, typically 16GB+ GPU)

## Environment
- Conda environment named `gsdf`
- Dependencies managed via `environment.yml`
