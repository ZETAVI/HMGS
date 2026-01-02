# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GSDF (3D Gaussian Splatting meets SDF) is a NeurIPS 2024 paper implementation that combines explicit 3D Gaussian Splatting (3DGS) with implicit Signed Distance Fields (SDF) for improved neural rendering and 3D reconstruction from multiview images. The dual-branch architecture mutually guides both representations during training to achieve:

- **High-quality rendering** via 3DGS (PSNR ~29.38, real-time fps)
- **Accurate geometry reconstruction** via SDF (reduced floaters, precise surfaces)

### Core Innovation

Three mutual guidance mechanisms connect the branches:
1. **GS → SDF**: Depth-guided ray sampling accelerates SDF convergence
2. **SDF → GS**: Geometry-aware density control (growing/pruning Gaussians near surfaces)
3. **Bidirectional**: Mutual geometry supervision aligns depth and normal maps

## Development Environment

### Setup
```bash
# Clone with submodules (includes diff-gaussian-rasterization)
git clone https://github.com/city-super/GSDF.git --recursive
cd GSDF

# Create conda environment
conda env create --file environment.yml
conda activate gsdf

Requirements: Ubuntu 22.04, CUDA 11.8, GCC 11.4.0 (similar configs should work)

Data Preparation

mkdir data
# Organize as: data/dataset_name/scene_name/{images/, sparse/0/}

Supported datasets:
- MipNeRF360 (bicycle, bonsai, counter, garden, kitchen, room, stump)
- Tanks&Temples / Deep Blending
- DTU
- Custom data (process with COLMAP for SfM points/poses)

Training

Single Scene Training

bash ./train.sh

Key parameters in train.sh:
- exp_dir: Experiment output directory
- config: Path to config file (in configs/)
- gpu: GPU ID ('-1' for auto-select most idle)
- train/eval: Mode selection
- tag: Experiment identifier

Output locations:
- GS branch logs: outputs/${tag}/
- SDF branch logs: exp/scenename/${tag}/

Training Process

The train.py orchestrates dual-branch training:
1. GS Branch: Based on Scaffold-GS, handles Gaussian primitives with tile-based rasterization
2. SDF Branch: Based on Instant-NSR, uses multi-resolution hash encoding (16 layers, 2^5 to 2^11)
3. Mutual Guidance: Applied at each iteration via launch.py coordination

Typical training iterations: 30k-500k depending on scene complexity

Code Architecture

High-Level Structure

GSDF/
├── train.py              # Main training orchestrator (dual-branch coordination)
├── launch.py             # Training launcher with config management
├── render.py             # Rendering pipeline (post-training visualization)
├── metrics.py            # Evaluation metrics (PSNR, SSIM, LPIPS)
│
├── gaussian_splatting/   # GS Branch (explicit representation)
│   ├── scene/           # Scene representation, cameras, Gaussian models
│   ├── utils/           # 3DGS-specific utilities (loss, graphics, system)
│   └── arguments/       # Command-line argument parsers
│
├── instant_nsr/          # SDF Branch (implicit representation)
│   ├── models/          # Neural SDF models with hash encoding
│   ├── datasets/        # Data loaders for multiview images
│   ├── systems/         # Training systems (volume rendering, Eikonal loss)
│   └── utils/           # SDF-specific utilities
│
└── configs/             # Configuration files per dataset/scene

Key Dual-Branch Integration Points

In train.py:
- Alternating optimization between GS and SDF branches
- Depth map exchange (GS renders depth → guides SDF ray sampling)
- SDF query at Gaussian locations → density control (grow/prune)
- Joint loss: L = L_gs + L_sdf + L_mutual where L_mutual = λ_d·L_depth + λ_n·L_normal

GS Branch (gaussian_splatting/):
- Inherits from Scaffold-GS architecture
- Modified density control adds SDF proximity check: ε_g = ∇_g + φ_g·μ(s) and ε_p = σ_a - φ_p(1-μ(s))
- Outputs: RGB images, depth maps, normal maps (from Gaussian covariance)

SDF Branch (instant_nsr/):
- Uses Instant-NGP style multi-resolution hash grids
- Volume rendering with adaptive sampling range [D - k|s|, D + k|s|] based on GS depth
- Eikonal loss L_eik and curvature loss L_curv for regularization

Configuration System

Configs in configs/ follow hierarchy:
- Base configs for datasets (e.g., mipnerf360.yaml)
- Scene-specific overrides
- Key parameters:
- model.geometry.grad_type: SDF gradient computation method
- model.variance.init_val: Initial variance for volume rendering
- loss.lambda_eikonal, loss.lambda_curvature: Regularization weights
- GS parameters passed via train.sh (iterations, position_lr, etc.)

Evaluation

Rendering Evaluation

Automatic (during training completion):
- Renders test views
- Computes PSNR, SSIM, LPIPS
- Estimates FPS via torch.cuda.synchronize() timing
- Saves results to log directories

Manual (post-training):
python render.py -m <path_to_trained_model>  # Generate novel views
python metrics.py -m <path_to_trained_model>  # Compute metrics

Geometry Reconstruction Evaluation

For Chamfer Distance on extracted meshes, refer to https://github.com/hbb1/2d-gaussian-splatting.

Development Workflow

Adding New Scenes

1. Process images with COLMAP → obtain SfM data
2. Place in data/dataset_name/scene_name/
3. Create/modify config in configs/
4. Update train.sh with paths and scene name
5. Run training

Modifying Mutual Guidance

Key locations:
- Depth-guided sampling: instant_nsr/models/ - modify ray sampling logic based on depth input
- Density control: gaussian_splatting/scene/gaussian_model.py - adjust grow/prune criteria using SDF queries
- Mutual loss: train.py - tune λ_d, λ_n weights for depth/normal alignment

Debugging Tips

- GS branch uses gaussian_splatting/utils/loss_utils.py for losses
- SDF branch logs via pytorch_lightning in instant_nsr/systems/
- Check both log directories for branch-specific issues
- Visualize depth/normal maps from both branches for alignment verification

Technical Background (from paper)

3D Gaussian Splatting (3DGS)

- Primitive: Anisotropic 3D Gaussians with position μ, covariance Σ, opacity α, SH colors
- Rendering: Tile-based rasterization with α-blending (real-time)
- Limitation: Fuzzy geometry, floaters, no explicit surface

Signed Distance Field (SDF)

- Representation: Continuous function s(x) = signed distance to nearest surface
- Rendering: Volume rendering with density σ(x) = sigmoid(-k·s(x))
- Limitation: Slow training, computationally expensive

Multi-resolution Hash Encoding (from Instant-NGP)

- 16 levels, resolution 2^5 to 2^11
- Feature dim 4 per level
- Accelerates SDF training from hours to minutes

Loss Functions

GS Branch: L_gs = λ₁·L₁ + (1-λ₁)·L_SSIM + λ_vol·L_vol
SDF Branch: L_sdf = L₁ + λ_eik·L_eik + λ_curv·L_curv
Mutual: L_mutual = λ_d·L_depth + λ_n·L_normal

Common Issues

1. CUDA OOM: Reduce batch size in config or use lower resolution
2. Slow SDF convergence: Ensure GS branch provides reasonable depth maps early
3. Floaters in rendering: Increase λ_vol or adjust pruning threshold in density control
4. Poor geometry: Tune Eikonal/curvature loss weights or mutual supervision weights

Citation

@article{yu2024gsdf,
title={Gsdf: 3dgs meets sdf for improved rendering and reconstruction},
author={Yu, Mulin and Lu, Tao and Xu, Linning and Jiang, Lihan and Xiangli, Yuanbo and Dai, Bo},
journal={arXiv preprint arXiv:2403.16964},
year={2024}
}

Acknowledgments

- GS Branch: Built on https://github.com/city-super/Scaffold-GS
- SDF Branch: Built on https://github.com/bennyguo/instant-nsr-pl
- Follows https://github.com/graphdeco-inria/gaussian-splatting