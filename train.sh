#!/bin/bash

# ==========================================
# 批量训练脚本 / Batch Training Script
# ==========================================

# 默认全局变量 / Default Global Variables
gpu=0
exp_dir="./exp"

# 1. 解析命令行参数 (允许运行时覆盖 GPU 设置)
# 使用方法: bash train.sh --gpu 0
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -g|--gpu) gpu="$2"; shift ;;
        -l|--logdir) exp_dir="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Using GPU: $gpu"
echo "Experiment Directory: $exp_dir"

# ==========================================
# 2. 定义场景和分辨率数组 / Define Scenes & Resolutions
# ==========================================

# 场景名称列表 (对应 configs/mipnerf360/ 下的 yaml 文件名)
# 你可以在这里添加更多的场景
scenes=(
    'drjohnson'
    'playroom'
)

# 对应的渲染分辨率 (1, 2, 4 等) - 索引必须与 scenes 一一对应
resolutions=(
    -1
    -1
)

# 检查数组长度是否一致
if [ "${#scenes[@]}" -ne "${#resolutions[@]}" ]; then
    echo "Error: The number of scenes and resolutions must match!"
    exit 1
fi

# ==========================================
# 3. 循环遍历并执行 / Loop Execution
# ==========================================

for ((i=0; i<${#scenes[@]}; i++)); do
    scene="${scenes[$i]}"
    res="${resolutions[$i]}"
    
    echo "----------------------------------------------------------------"
    echo "Processing Scene [$((i+1))/${#scenes[@]}]: ${scene} (Resolution: ${res})"
    echo "----------------------------------------------------------------"

    # 定义变量
    config="configs/mipnerf360/${scene}.yaml"
    tag="ORIGINAL_DB/${scene}"
    output_dir="output/${tag}"
    log_dir="${output_dir}/logger"

    # 创建日志目录
    mkdir -p "${log_dir}"

    # --------------------------------------
    # A. 训练 / Training (launch.py)
    # --------------------------------------
    echo "[Step 1/3] Starting Training..."
    
    # 构建并打印命令，方便调试
    cmd_train="python launch.py --exp_dir ${exp_dir} --config ${config} --gpu ${gpu} --train --eval tag=${tag}"
    echo "Running: $cmd_train"
    
    python -u launch.py \
        --exp_dir "${exp_dir}" \
        --config "${config}" \
        --gpu "${gpu}" \
        --train \
        --eval \
        tag="${tag}" \
        2>&1 | tee "${log_dir}/training.log"
    
    # 检查上一步的退出状态 (使用 PIPESTATUS 获取管道前一个命令也就是 python 的状态)
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error found in training ${scene}, skipping render/metrics..."
        continue
    fi

    # --------------------------------------
    # B. 渲染 / Rendering (render.py)
    # --------------------------------------
    echo "[Step 2/3] Starting Rendering..."

    # 构建并打印命令，方便调试
    cmd_render="python render.py -m ${output_dir} --config ${config} --skip_train --iteration -1 --resolution ${res}"
    echo "Running: $cmd_render"

    # -m 指定模型输出路径, --resolution 指定降采样倍率
    python -u render.py \
        -m "${output_dir}" \
        --config "${config}" \
        --skip_train \
        --iteration -1 \
        --resolution "${res}" \
        2>&1 | tee "${log_dir}/render.log"

    # --------------------------------------
    # C. 评估 / Metrics (metrics.py)
    # --------------------------------------
    echo "[Step 3/3] Calculating Metrics..."

    # 构建并打印命令，方便调试
    cmd_metrics="python metrics.py -m ${output_dir}"
    echo "Running: $cmd_metrics"

    python -u metrics.py \
        -m "${output_dir}" \
        2>&1 | tee "${log_dir}/metrics.log"

    echo "Done processing ${scene}."
    echo ""
done
