# GSDF Project Overview

## Purpose
GSDF (3D Gaussian Splatting meets SDF) is a **dual-branch neural rendering architecture** combining:
- **3D Gaussian Splatting (3DGS)** - explicit rendering for high-quality, real-time visualization
- **Neural Signed Distance Fields (SDF)** - implicit representation for accurate geometry reconstruction

The two branches **mutually guide each other** during training to achieve both:
- High rendering quality (PSNR ~29.38, real-time FPS)
- Precise 3D surface reconstruction (reduced floaters, sharp geometry)

## Core Innovation
Three mutual guidance mechanisms connect the branches:
1. **GS → SDF**: Gaussian depth guides SDF ray sampling (accelerates convergence)
2. **SDF → GS**: SDF proximity controls Gaussian density (grow/prune near surfaces)
3. **Bidirectional**: Joint depth and normal supervision aligns both representations

## Use Cases
- **Input**: Multi-view images of 3D scenes (processed with COLMAP for camera poses)
- **Output**: 
  - High-quality novel view synthesis (rendering)
  - Accurate 3D mesh reconstruction (geometry)
- **Datasets**: MipNeRF360, Tanks&Temples, DTU, Deep Blending, or custom COLMAP data

## Academic Context
- NeurIPS 2024 paper
- Built on Scaffold-GS (rendering) + Instant-NSR (reconstruction)
- Addresses the traditional tradeoff between rendering quality and geometry accuracy
