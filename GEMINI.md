# GSDF Project Context & Serena Agent Manual

## 1. Serena MCP Tool Usage Guidelines

As an intelligent agent operating in this environment, you must adhere to the following protocols to ensure efficient and accurate code manipulation.

### Core Philosophy: "Symbolic First"
Do **not** read entire files unless absolutely necessary. Instead, rely on step-by-step information acquisition using symbolic tools.

1.  **Exploration**:
    *   Use `get_symbols_overview` to understand a file's structure.
    *   Use `find_symbol` to locate specific classes, methods, or functions.
    *   Use `find_referencing_symbols` to understand how symbols are used across the codebase.
    *   Use `search_for_pattern` only when symbolic tools are insufficient or for non-code files.

2.  **Reading**:
    *   Only read the *bodies* of symbols you need to modify or understand in detail (e.g., `find_symbol(..., include_body=True)`).
    *   Avoid `read_file` for large source files; it is token-inefficient.

3.  **Editing**:
    *   **Symbolic Editing (Preferred)**: Use `replace_symbol_body` to modify entire functions or classes. Use `insert_after_symbol` or `insert_before_symbol` to add new code.
    *   **File-Based Editing**: Use `replace_content` with **regex** for small, targeted changes (e.g., changing a few lines within a function) or when symbolic tools are not applicable.

## 2. Project Overview
**GSDF** (3DGS Meets SDF) is a dual-branch architecture combining **3D Gaussian Splatting (3DGS)** and **Neural Signed Distance Fields (SDF)**.
- **Goal**: Improve rendering quality (fewer artifacts) and geometric reconstruction (better surface detail) via mutual guidance.
- **Architecture**:
    - **GS Branch**: Based on [Scaffold-GS](https://github.com/city-super/Scaffold-GS).
    - **SDF Branch**: Based on [Instant-NSR](https://github.com/bennyguo/instant-nsr-pl) (NeuS-based).
- **Paper**: [arXiv:2403.16964](https://arxiv.org/abs/2403.16964)

## 3. Project Structure & Architecture Analysis

### Architectural Pattern: "Host-Parasite" Controller
The project uses **Instant-NSR** (PyTorch Lightning) as the master controller (`NeuSSystem`) which instantiates and manages the **Scaffold-GS** model (`GaussianModel`) internally.

### Core Controller: `NeuSSystem` (`instant_nsr/systems/neus.py`)
This class orchestrates the joint training loop and mutual guidance.
- **Initialization**: Initializes both `NeuS` (implicit) and `GaussianModel` (explicit).
- **`pretrain_gs()`**: Runs a pure Gaussian Splatting phase (~15k steps) to establish initial geometry.
- **`training_step()`**: The joint training heartbeat.

### Mutual Guidance Mechanism
The key innovation lies in how the two branches supervise each other within the `training_step`:

1.  **SDF $\to$ GS Guidance**:
    - **Loss Supervision**: The GS branch uses the SDF's predicted depth and normals as pseudo-ground-truth.
        - `L1(GS_depth, SDF_depth)`
        - `CosSim(GS_normal, SDF_normal)`
    - **Structural Supervision**: The `adjust_anchor` method uses SDF values to decide where to grow or prune Gaussians (pruning those far from the zero-level set).

2.  **GS $\to$ SDF Guidance**:
    - **Sampling Guidance**: The SDF ray sampler uses the GS rendered depth map (`picked_gs_depth`) to skip empty space and focus sampling near the surface (`use_depth_guide=True`).
    - **Loss Supervision**: The SDF branch uses GS depth and normals as consistency targets.

### Key Modules
- **`gaussian_splatting/`** (GS Branch):
    - `scene/`: Data loaders (`colmap_loader.py`) and `GaussianModel`.
    - `gaussian_renderer/`: Differentiable rasterizer interface.
    - `submodules/`: C++/CUDA extensions (`diff-gaussian-rasterization`).
- **`instant_nsr/`** (SDF Branch):
    - `models/`: Neural networks (`geometry.py`, `neus.py`).
    - `systems/`: Training loops and loss functions (`systems/neus.py`).
    - `datasets/`: Data loading logic.

### Generated Directories (Git-Ignored)
- **`data/`**: User-provided datasets (Colmap format).
- **`exp/`**: SDF-branch experiment outputs (logs, checkpoints).
- **`output/`**: GS-branch experiment outputs.
- **`runs/`**: TensorBoard logs.

## 4. Technology Stack
- **Languages**: Python 3.7.13, C++/CUDA 11.6+.
- **Frameworks**: PyTorch 1.12.1, PyTorch Lightning 1.9.5.
- **Libraries**: `nerfacc` (NeRF accel), `plyfile`, `PyMCubes` (Meshing), `wandb`, `omegaconf`.
- **Build**: Conda (`environment.yml`), CMake (for submodules).

## 5. Setup & Installation
1.  **Clone**: `git clone https://github.com/city-super/GSDF.git --recursive`
2.  **Env**: `conda env create --file environment.yml && conda activate gsdf`

## 6. Usage Workflows

### Training
**Option A: Shell Script (Recommended)**
Modify `train.sh` variables (`exp_dir`, `config`, `gpu`) and run:
```bash
bash ./train.sh
```

**Option B: Direct Launch**
```bash
python launch.py \
    --config configs/tnt/barn.yaml \
    --gpu 0 \
    --train --eval \
    --exp_dir ./exp \
    tag=my_experiment
```

### Rendering & Evaluation
```bash
python render.py -m <path_to_trained_model_dir>
python metrics.py -m <path_to_trained_model_dir>
```

## 7. Development Conventions

### Code Style
- **Python**: `snake_case` for modules/functions/vars, `PascalCase` for classes.
- **Config**: Heavily relied upon. Use YAML in `configs/` instead of hardcoding.
- **Compatibility**: Ensure changes work for both GS and SDF branches.

### Configuration (`configs/`)
- Use `${variable}` syntax for interpolation.
- Example: `model.geometry.radius` often references `model.radius`.

### Submodules
- Custom CUDA kernels reside in `gaussian_splatting/submodules`.
- If you modify C++ code, ensure you rebuild the extension.

## 8. Task Completion Checklist
- [ ] **Config**: specific parameters moved to YAML, not hardcoded.
- [ ] **Structure**: GS code in `gaussian_splatting/`, SDF code in `instant_nsr/`.
- [ ] **Testing**: Run `launch.py` with `--train --eval` to verify no regressions.
- [ ] **Cleanup**: Remove debug prints and temporary files.
- [ ] **Output**: Verify logs/checkpoints appear in `exp/` or `output/`.
