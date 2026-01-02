# Suggested Commands

## Environment Setup

### Initial Installation
```bash
# Clone with submodules (critical for CUDA extensions!)
git clone https://github.com/city-super/GSDF.git --recursive
cd GSDF

# Create conda environment
conda env create --file environment.yml
conda activate gsdf
```

### Verify CUDA Extensions Built
```bash
# These are installed via pip in environment.yml but verify:
python -c "import diff_gaussian_rasterization; print('Rasterizer OK')"
python -c "import simple_knn; print('KNN OK')"
```

## Training

### Main Training Command
```bash
# Edit train.sh first, then:
bash ./train.sh

# Or run directly:
python launch.py \
    --exp_dir ./exp \
    --config configs/tnt/barn.yaml \
    --gpu 0 \
    --train \
    --eval \
    tag=my_experiment
```

**Key Parameters:**
- `--config`: Path to scene config file
- `--gpu`: GPU ID (use `-1` for auto-select idle GPU)
- `--train/--eval/--test/--validate`: Mode selection
- `tag=<name>`: Experiment identifier (appended to CLI args)
- `--resume <path>`: Resume from checkpoint
- `--resume_weights_only`: Restore only weights, not training state

### Monitor Training
```bash
# TensorBoard for both branches
tensorboard --logdir runs/

# Logs are written to:
# - GS branch: output/${tag}/
# - SDF branch: exp/${scene_name}/${trial_name}/
```

## Evaluation

### Automatic (during training)
- Quality metrics computed automatically at test iterations
- FPS estimated via CUDA synchronization timing
- Results saved to log directories

### Manual Rendering & Metrics
```bash
# Render test views
python render.py -m <path_to_trained_model>

# Compute metrics (PSNR, SSIM, LPIPS)
python metrics.py -m <path_to_trained_model>
```

## Data Preparation

### Setup Data Directory
```bash
mkdir data

# Organize as:
# data/
# └── dataset_name/
#     └── scene_name/
#         ├── images/        # Input photos
#         └── sparse/0/      # COLMAP SfM output
```

### Process Custom Data with COLMAP
```bash
# Example COLMAP SfM pipeline (run outside this repo):
colmap feature_extractor --database_path database.db --image_path images/
colmap exhaustive_matcher --database_path database.db
colmap mapper --database_path database.db --image_path images/ --output_path sparse/
```

## Development Utilities

### Check GPU Availability
```bash
# The code auto-selects GPU with lowest memory usage via:
nvidia-smi -q -d Memory |grep -A4 GPU|grep Used
```

### Git Operations
```bash
# Update submodules (if changed)
git submodule update --init --recursive

# Check status
git status
git diff
```

### File Search
```bash
# Find config files
find configs/ -name "*.yaml"

# Search for patterns in code
grep -r "def training_step" instant_nsr/

# List directory structure
tree -L 2 gaussian_splatting/
```

### Python Debugging
```bash
# Run with anomaly detection
python launch.py --config <config> --gpu 0 --train tag=debug \
    --debug_from 0 --detect_anomaly
```

## Common Tasks

### Train a New Scene
1. Process images with COLMAP → `sparse/0/`
2. Place in `data/dataset_name/scene_name/`
3. Create config in `configs/dataset_name/scene_name.yaml` (copy existing)
4. Update `train.sh` with new config path
5. Run `bash train.sh`

### Resume Interrupted Training
```bash
python launch.py --config <config> --gpu 0 --train \
    --resume exp/${scene}/${trial}/ckpt/last.ckpt \
    tag=resumed
```

### Extract Mesh from SDF
```bash
# Marching cubes extraction (code in instant_nsr/models/geometry.py)
# Typically called during validation/testing in the training loop
# Manual extraction requires loading checkpoint and running isosurface extraction
```

## Linux System Commands

### Disk Space
```bash
df -h          # Check disk usage
du -sh data/*  # Check data directory size
```

### Process Management
```bash
ps aux | grep python   # Find running Python processes
nvidia-smi            # Monitor GPU usage
htop                  # Interactive process viewer
```

### File Operations
```bash
ls -lah               # List files with details
cp -r source dest     # Copy directories
mv old new            # Move/rename
rm -rf dir/           # Remove directory (careful!)
```
