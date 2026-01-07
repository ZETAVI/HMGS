import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss
from plyfile import PlyData, PlyElement
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import time

import instant_nsr.models
from instant_nsr.models.utils import cleanup
from instant_nsr.models.ray_utils import get_rays
import instant_nsr.systems
from instant_nsr.systems.base import BaseSystem
from instant_nsr.systems.criterions import PSNR, binary_cross_entropy
from instant_nsr.utils.loss_utils import l1_loss, ssim
from gaussian_splatting import gaussian_renderer
import sys
from gaussian_splatting.scene import Scene, GaussianModel
from gaussian_splatting.utils.general_utils import safe_state
import uuid
from gaussian_splatting.utils.image_utils import psnr, error_map
from gaussian_splatting.utils.visualize_utils import apply_depth_colormap
from gaussian_splatting.utils.depth_utils import depth_to_normal
from argparse import ArgumentParser, Namespace
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams
import os
import numpy as np
from random import randint
from tqdm import tqdm
from gaussian_splatting.scene.cameras import Camera
import torchvision
# from gaussian_splatting.utils.misc import config_to_primitive
import random
from pathlib import Path
from gaussian_splatting.lpipsPyTorch import lpips
import json
from os import makedirs
from PIL import Image
import torchvision.transforms.functional as tf
from instant_nsr.systems import register

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = gaussian_renderer.prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask,out_depth=True,return_normal=True)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    depth_gs = render_pkg["depth_hand"]
                    depth_gs=depth_gs/depth_gs.max()
                    depth_gs_map = apply_depth_colormap(render_pkg["depth_hand"][...,None], render_pkg["accumulation"][...,None], near_plane=None, far_plane=None)
                    normal_gs = render_pkg["gs_normal"]
                    normal_gs_normal=(F.normalize(normal_gs, p=2, dim=0)+1)/2

                    normal = render_pkg["normal"]
                    normal = torch.nn.functional.normalize(normal, p=2, dim=0)
                    # transform to world space
                    c2w = (viewpoint.world_view_transform.T).inverse()
                    normal2 = c2w[:3, :3] @ normal.reshape(3, -1)
                    normal = normal2.reshape(3, *normal.shape[1:])
                    normal = (normal + 1.) / 2.

                    depth_black = render_pkg["depth_map"]
                    depth_normal, _ = depth_to_normal(viewpoint, depth_black)
                    depth_normal = (depth_normal + 1.) / 2.
                    depth_normal = depth_normal.permute(2, 0, 1)

                    depth_map = apply_depth_colormap(depth_black.permute(1, 2, 0), render_pkg["accumulation"].permute(1, 2, 0), near_plane=None, far_plane=None)
                    depth_map = depth_map.permute(2, 0, 1)  # HWC -> CHW

                    accumlated_alpha = render_pkg["accumulation"]
                    colored_accum_alpha = apply_depth_colormap(accumlated_alpha.permute(1, 2, 0), None, near_plane=0.0, far_plane=1.0)
                    colored_accum_alpha = colored_accum_alpha.permute(2, 0, 1)


                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    error_image = error_map(image, gt_image)

                    if tb_writer and (idx < 38):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/normal_GSDF".format(viewpoint.image_name), normal_gs_normal[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth_GSDF_black".format(viewpoint.image_name), depth_gs[None], global_step=iteration)

                        tb_writer.add_images(config['name'] + "_view_{}/normal_gof".format(viewpoint.image_name), normal[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth_normal_gof".format(viewpoint.image_name), depth_normal[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth_map".format(viewpoint.image_name), depth_map[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/accumulated_alpha".format(viewpoint.image_name), colored_accum_alpha[None], global_step=iteration)

                        tb_writer.add_images(config['name'] + "_view_{}/errormap_GSDF".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/errormap".format(viewpoint.image_name), error_image[None], global_step=iteration)
                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
          
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})
                
        if tb_writer:
            tb_writer.add_scalar(config['name'] + '/total_points', scene.gaussians.get_anchor.shape[0], iteration)
            
        torch.cuda.empty_cache()

def neus_rendering_report(neus_system, dataset_name, iteration, testing_iterations, tb_writer=None, logger=None):
    """
    NeuS分支的场景渲染报告函数

    与GS分支的training_report对应，专门用于NeuS分支的渲染可视化
    在指定的iteration对test和train视角进行NeuS完整场景渲染

    Args:
        neus_system: NeuSSystem实例（self）
        dataset_name: 数据集名称
        current_step: 当前训练步数
        tb_writer: TensorBoard SummaryWriter（self.tb_writer，与GS共享）
        logger: 日志信息记录器 (self.loggger)

    功能：
        1. 对test和train视角进行NeuS完整场景渲染
        2. 可视化NeuS的各种输出（RGB、depth、normal、opacity、background等）
        3. 记录到TensorBoard (output目录，与GS分支共享)
        4. 计算并记录评估指标（L1、PSNR）

    与GS分支的区别：
        - 使用NeuS模型渲染（体渲染+SDF）
        - 渲染速度较慢，限制视角数量
        - 输出更丰富（foreground/background分离、opacity等）
        - 使用不同的命名前缀（NeuS_xxx）便于在tensorboard中区分
    """
    # 检查tb_writer
    if tb_writer is None:
        if logger:
            logger.warning("TensorBoard writer not available, skipping NeuS rendering report")
        return

    # if logger:
    #     logger.info(f"\n{'='*60}")
    #     logger.info(f"[STEP {iteration}] NeuS Branch Rendering Report")
    #     logger.info(f"{'='*60}")

    torch.cuda.empty_cache()

    if iteration in testing_iterations:

        # 配置要渲染的视角（NeuS渲染慢，只渲染少量视角）
        validation_configs = (
            {
                'name': 'test',
                # 'cameras': neus_system.scene.getTestCameras()[:8],  # 限制为前8个test视角
                'cameras': neus_system.scene.getTestCameras(),
                'log_prefix': f'{dataset_name}_NeuS_test'
            },
            {
                'name': 'train',
                'cameras': [neus_system.scene.getTrainCameras()[idx % len(neus_system.scene.getTrainCameras())]
                        for idx in range(5, min(20, len(neus_system.scene.getTrainCameras())), 5)],  # 每隔5个取一个，最多3个
                'log_prefix': f'{dataset_name}_NeuS/train'
            }
        )

        for config in validation_configs:
            if not config['cameras'] or len(config['cameras']) == 0:
                continue

            l1_test = 0.0
            psnr_test = 0.0
            render_count = 0

            if logger:
                logger.info(f"\nRendering {config['name']} views ({len(config['cameras'])} views)...")

            for idx, viewpoint in enumerate(config['cameras']):
                try:
                    # 1. 准备数据 - 使用与preprocess_data相同的方式获取rays
                    W, H = neus_system.dataset.img_wh

                    # 获取相机姿态
                    # 从viewpoint的Camera对象获取对应的索引
                    cam_index = viewpoint.uid
                    c2w = neus_system.dataset.all_c2w[cam_index]

                    # 获取directions
                    if neus_system.dataset.directions.ndim == 3:  # (H, W, 3)
                        directions = neus_system.dataset.directions
                    elif neus_system.dataset.directions.ndim == 4:  # (N, H, W, 3)
                        directions = neus_system.dataset.directions[cam_index]

                    # 使用get_rays生成全图的rays
                    rays_o, rays_d = get_rays(directions, c2w)
                    rays = torch.cat([rays_o, rays_d], dim=-1).reshape(-1, 6)

                    # 准备batch
                    batch = {
                        'rays': rays.to(torch.device("cuda")),
                        'rgb': neus_system.dataset.all_images[cam_index].reshape(-1, 3).to(torch.device("cuda")),
                        'index': torch.tensor([cam_index], device="cuda")
                    }

                    # 2. NeuS渲染（不使用GS depth guide，纯NeuS渲染）
                    with torch.no_grad():
                        neus_system.model.eval()  # 确保评估模式
                        out = neus_system(batch, gs_depth=None, use_depth_guide=False)
                        neus_system.model.train()  # 恢复训练模式

                    # 跳过无效渲染
                    if 'zero_samples' in out and out['zero_samples']:
                        if logger:
                            logger.warning(f"  View {idx} ({viewpoint.image_name}): zero samples, skipped")
                        continue

                    # 3. 提取和处理渲染结果
                    # RGB渲染（完整 = 前景 + 背景）
                    comp_rgb_full = out['comp_rgb_full'].view(H, W, 3).permute(2, 0, 1)  # CHW
                    comp_rgb_full = torch.clamp(comp_rgb_full, 0.0, 1.0)

                    # 前景RGB
                    comp_rgb = out['comp_rgb'].view(H, W, 3).permute(2, 0, 1)
                    comp_rgb = torch.clamp(comp_rgb, 0.0, 1.0)

                    # 前景深度图
                    depth_fg = out['depth'].view(H, W, 1)
                    # 不需要归一化，因为apply_depth_colormap会自动处理
                    # depth_normalized = depth / (depth.max() + 1e-10)  # 归一化
                    # depth_normalized_vis = depth_normalized.permute(2, 0, 1).repeat(3, 1, 1)  # CHW, 转为3通道

                    # 前景不透明度
                    opacity_fg = out['opacity'].view(H, W, 1)      # [H, W, 1]

                    # # 深度图着色
                    # depth_colored = apply_depth_colormap(
                    #     depth,
                    #     None,
                    #     near_plane=None,
                    #     far_plane=None
                    # ).permute(2, 0, 1)  # CHW

                    # 法线图
                    comp_normal = out['comp_normal'].view(H, W, 3).permute(2, 0, 1)
                    # 法线从[-1,1]归一化到[0,1]用于可视化
                    comp_normal_vis = (comp_normal + 1.0) / 2.0
                    comp_normal_vis = torch.clamp(comp_normal_vis, 0.0, 1.0)

                    # # 不透明度/Alpha通道
                    # opacity = out['opacity'].view(H, W, 1)
                    # # 着色opacity以便更好可视化
                    # opacity_colored = apply_depth_colormap(
                    #     opacity,
                    #     None,
                    #     near_plane=0.0,
                    #     far_plane=1.0
                    # ).permute(2, 0, 1)

                    # 背景（如果有learned_background）
                    has_background = neus_system.config.model.learned_background and 'comp_rgb_bg' in out
                    if has_background:
                        comp_rgb_bg = out['comp_rgb_bg'].view(H, W, 3).permute(2, 0, 1)
                        comp_rgb_bg = torch.clamp(comp_rgb_bg, 0.0, 1.0)

                        # 背景深度和不透明度
                        depth_bg = out['depth_bg'].view(H, W, 1)       # [H, W, 1]
                        opacity_bg = out['opacity_bg'].view(H, W, 1)   # [H, W, 1]

                    # ========== 3.2 合成整体depth和opacity（数值层面）==========
                    # Opacity合成（Alpha Compositing）
                    # opacity_full = opacity_fg + (1 - opacity_fg) * opacity_bg
                    opacity_full = opacity_fg + (1.0 - opacity_fg) * opacity_bg  # [H, W, 1]
                    
                    # Depth合成（加权平均，权重为对最终图像的贡献度）
                    # 前景贡献权重: w_fg = opacity_fg
                    # 背景贡献权重: w_bg = (1 - opacity_fg) * opacity_bg
                    # depth_full = (w_fg * depth_fg + w_bg * depth_bg) / (w_fg + w_bg)
                    weight_fg = opacity_fg
                    weight_bg = (1.0 - opacity_fg) * opacity_bg
                    total_weight = weight_fg + weight_bg

                    # 加权平均深度（避免除零）
                    depth_full = (weight_fg * depth_fg + weight_bg * depth_bg) / (total_weight + 1e-10)

                    # 对于完全透明的像素（total_weight ≈ 0），depth无意义，设置为前景深度
                    mask_invalid = (total_weight < 1e-6)
                    depth_full = torch.where(mask_invalid, depth_fg, depth_full)  # [H, W, 1]

                    # ========== 3.3 统一进行颜色映射（转为CHW格式）==========
                    # 深度着色（apply_depth_colormap支持CPU tensor）
                    depth_fg_colored = apply_depth_colormap(
                        depth_fg, None, near_plane=None, far_plane=None
                    ).permute(2, 0, 1)  # CHW

                    depth_bg_colored = apply_depth_colormap(
                        depth_bg, None, near_plane=None, far_plane=None
                    ).permute(2, 0, 1)  # CHW

                    depth_full_colored = apply_depth_colormap(
                        depth_full, None, near_plane=None, far_plane=None
                    ).permute(2, 0, 1)  # CHW

                    # 不透明度着色
                    opacity_fg_colored = apply_depth_colormap(
                        opacity_fg, None, near_plane=0.0, far_plane=1.0
                    ).permute(2, 0, 1)  # CHW

                    opacity_bg_colored = apply_depth_colormap(
                        opacity_bg, None, near_plane=0.0, far_plane=1.0
                    ).permute(2, 0, 1)  # CHW

                    opacity_full_colored = apply_depth_colormap(
                        opacity_full, None, near_plane=0.0, far_plane=1.0
                    ).permute(2, 0, 1)  # CHW


                    # Ground truth
                    # gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_image = viewpoint.original_image.to(torch.device("cpu"))
                    gt_image = torch.clamp(gt_image, 0.0, 1.0)

                    # 误差图（完整渲染 vs GT）
                    error_image = error_map(comp_rgb_full, gt_image)

                    # 4. 记录到TensorBoard（使用tb_writer，与GS分支共享）
                    view_name = viewpoint.image_name

                    # 主要渲染输出（按重要性排序，便于在tensorboard中查看）
                    tb_writer.add_images(
                        f'{config["log_prefix"]}_view_{view_name}/1_render_full',
                        comp_rgb_full[None],
                        global_step=iteration
                    )
                    tb_writer.add_images(
                        f'{config["log_prefix"]}_view_{view_name}/2_errormap',
                        error_image[None],
                        global_step=iteration
                    )
                    tb_writer.add_images(
                        f'{config["log_prefix"]}_view_{view_name}/3_depth_full',
                        depth_full_colored[None],
                        global_step=iteration
                    )
                    tb_writer.add_images(
                        f'{config["log_prefix"]}_view_{view_name}/4_normal',
                        comp_normal_vis[None],
                        global_step=iteration
                    )
                    tb_writer.add_images(
                        f'{config["log_prefix"]}_view_{view_name}/5_opacity_full',
                        opacity_full_colored[None],
                        global_step=iteration
                    )


                    # 前景/背景分离
                    tb_writer.add_images(
                        f'{config["log_prefix"]}_view_{view_name}/6_render_foreground',
                        comp_rgb[None],
                        global_step=iteration
                    )
                    if has_background:
                        tb_writer.add_images(
                            f'{config["log_prefix"]}_view_{view_name}/7_render_background',
                            comp_rgb_bg[None],
                            global_step=iteration
                    )
                    tb_writer.add_images(
                        f'{config["log_prefix"]}_view_{view_name}/depth_foreground',
                        depth_fg_colored[None],
                        global_step=iteration
                    )
                    tb_writer.add_images(
                        f'{config["log_prefix"]}_view_{view_name}/depth_background',
                        depth_bg_colored[None],
                        global_step=iteration
                    )
                    tb_writer.add_images(
                        f'{config["log_prefix"]}_view_{view_name}/opacity_foreground',
                        opacity_fg_colored[None],
                        global_step=iteration
                    )
                    tb_writer.add_images(
                        f'{config["log_prefix"]}_view_{view_name}/opacity_background',
                        opacity_bg_colored[None],
                        global_step=iteration
                    )

                    

                    # Ground truth（只在第一次记录）
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(
                            f'{config["log_prefix"]}_view_{view_name}/0_ground_truth',
                            gt_image[None],
                            global_step=iteration
                        )

                    # 5. 计算指标
                    l1_test += l1_loss(comp_rgb_full, gt_image).mean().double()
                    psnr_test += psnr(comp_rgb_full, gt_image).mean().double()
                    render_count += 1

                    if logger and (idx % 2 == 0 or idx == len(config['cameras']) - 1):
                        logger.info(f"  Rendered view {idx+1}/{len(config['cameras'])} ({view_name})")

                except Exception as e:
                    if logger:
                        logger.error(f"  Error rendering view {idx}: {str(e)}")
                    import traceback
                    if logger:
                        logger.error(traceback.format_exc())
                    continue

            # 6. 平均指标并记录
            if render_count > 0:
                psnr_test /= render_count
                l1_test /= render_count

                if logger:
                    logger.info(f"\n[STEP {iteration}] NeuS {config['name']}: L1 {l1_test:.6f}, PSNR {psnr_test:.4f}")

                # 记录标量指标（与GS分支类似）
                tb_writer.add_scalar(f'{config["log_prefix"]}/l1_loss', l1_test, iteration)
                tb_writer.add_scalar(f'{config["log_prefix"]}/psnr', psnr_test, iteration)

        # # 记录NeuS特有的参数
        # if 'inv_s' in out:
        #     inv_s_value = out['inv_s'].item() if hasattr(out['inv_s'], 'item') else float(out['inv_s'])
        #     tb_writer.add_scalar(f'{dataset_name}_NeuS/params/inv_s', inv_s_value, current_step)

        torch.cuda.empty_cache()

        if logger:
            logger.info(f"{'='*60}\n")

def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 

    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger


@register('neus-system')
class NeuSSystem(BaseSystem):
    """
    NeuS系统集成了Scaffold-GS和Instant-NSR用于神经表面重建。
    该系统结合了两种方法：
    1. Scaffold-GS：用于新视图合成的高斯溅射方法
    2. Instant-NSR：使用有符号距离函数(SDF)的神经表面重建方法
    
    系统支持：
    - 两个模型的联合训练与相互引导
    - 使用Scaffold-GS预测进行深度引导的光线采样
    - 高斯基元的几何感知密度控制
    - 预训练模型加载或从头训练
    - Tensorboard日志记录和可视化
    
    参数：
        config: 包含模型、数据集和训练参数的配置对象
    
    属性：
        current_epoch_set (int): 当前训练迭代计数器
        pretrain_step (int): 预训练Scaffold-GS的迭代次数（默认：15000）
        geometry_awared_control (bool): 是否使用几何感知密度控制
        gaussians (GaussianModel): Scaffold-GS模型
        scene (Scene): 用于相机和高斯基元的场景管理器
        progress_bar (tqdm): 训练进度条
        tb_writer (SummaryWriter): 用于日志记录的Tensorboard写入器
    
    训练流程：
        1. 预训练：单独训练Scaffold-GS `pretrain_step`次迭代
        2. 联合训练：使用以下方式训练两个模型：
           - Scaffold-GS为Instant-NSR提供深度/法线引导
           - Instant-NSR为Scaffold-GS的密度控制提供SDF
           - 通过深度和法线一致性损失进行相互监督
    
    关键特性：
        - 基于复杂度的动态光线采样
        - 自适应损失权重（迭代15000后法线/深度损失降低）
        - 使用预测的SDF进行几何感知的高斯致密化
        - 带预热期的多阶段训练

    Two ways to print to console:
        1. self.print: correctly handle progress bar 正确处理进度条
        2. rank_zero_info: use the logging module 使用logging模块
    """
    

    
    def __init__(self, config):
        """
        GS分支的初始化 和 NeuS隐式分支的初始化        
        初始化NeuS系统，设置预训练参数并配置Scaffold-GS模型。

        Sub-Component: GaussianModel and Scene (from gaussian_splatting)
       
        """
        # 触发父类NEuS模型的初始化
        super().__init__(config)

        # 初始化预训练和几何感知控制参数
        self.current_epoch_set = 0  # 当前训练迭代计数
        self.pretrain_step = 15000  # 预训练步数
        self.geometry_awared_control = False  # 几何感知控制标志（即3dgs是否会利用SDF的信息）
        
        # 防止loss权重多次减少
        self.loss_weight_reduced = False

        # 如果启用高斯分支
        if self.config.model.if_gaussian:
            # 设置参数解析器
            parser = ArgumentParser(description="Training script parameters")
            parser.source_path = config.dataset.root_dir
            print(parser.source_path)
            
            # 创建模型、优化和管线参数对象
            lp = ModelParams(parser)
            op = OptimizationParams(parser)
            pp = PipelineParams(parser)
            
            # 添加训练相关参数
            parser.add_argument('--ip', type=str, default="127.0.0.1")
            parser.add_argument('--port', type=int, default=6009)
            parser.add_argument('--debug_from', type=int, default=-1)
            parser.add_argument('--detect_anomaly', action='store_true', default=False)
            # default_steps = [100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000] + [15000 + 2500 * i for i in range(0, 11)]
            # [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000]
            default_steps = [5000 * (i+1) for i in range(0, 9)]
            parser.add_argument("--test_iterations", nargs="+", type=int, default=default_steps)
            # parser.add_argument("--test_iterations", nargs="+", type=int, default=[self.pretrain_step, self.pretrain_step+15000, self.pretrain_step+30000, self.pretrain_step+100000])
            parser.add_argument("--save_iterations", nargs="+", type=int, default=[self.pretrain_step, self.pretrain_step+15000, self.pretrain_step+30000, self.pretrain_step+100000])
            parser.add_argument("--quiet", action="store_true")
            parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
            parser.add_argument("--start_checkpoint", type=str, default=None)
            parser.add_argument('--if_merging', action='store_true', help="if using merging operator") 
            parser.add_argument('--config', required=True, help='path to config file')
            parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
            # parser.add_argument('--normal_w', type=float, default=0.01, help='weight of normal loss')
            # parser.add_argument('--depth_w', type=float, default=0.01, help='weight of depth loss')
            # parser.add_argument('--growing_weight', type=float, default=0.0002, help='weight of growing operator')
            parser.add_argument('tag', default='test')
            parser.add_argument('--add', type=int, default=0)
            parser.add_argument('--exp_dir', default='./exp')
            group = parser.add_mutually_exclusive_group(required=True)
            group.add_argument('--train', action='store_true')
            
            # 设置输出路径
            out_path = "output/" + config.tag
            fake_input = ["--source_path", config.dataset.root_dir, "--model_path", out_path]
            fake_input.extend(sys.argv[1:])
            args = parser.parse_args(fake_input)
            
            # 创建输出目录和日志器
            os.makedirs(args.model_path, exist_ok=True)
            print(f'model_path: {args.model_path}')
            self.loggger = get_logger(args.model_path)
            self.loggger.info(f'args: {args}')
            
            # 准备输出和Tensorboard日志器
            self.tb_writer = self.prepare_output_and_logger(lp.extract(args), op.extract(args), pp.extract(args))
            safe_state(args.quiet)
            # Start GUI server, configure and run training
            # network_gui.init(args.ip, args.port)
            
            # 设置异常检测
            torch.autograd.set_detect_anomaly(args.detect_anomaly)

            # 3DGS 分支其实也会共用img_downscale配置
            args.resolution = config.dataset.img_downscale
            self.args = args
            
            # 设置背景颜色
            bg_color = [1, 1, 1] if lp.extract(args).white_background else [0, 0, 0]
            self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            
            # 提取参数
            self.op = op.extract(args)
            self.piplin = pp.extract(args)
            self.lp = lp.extract(args)
            self.saving_iterations = args.save_iterations
            self.testing_iterations = args.test_iterations


            # NeuS专用的渲染报告间隔（比GS稀疏，因为NeuS渲染慢）
            # 自动从GS的test_iterations生成：取每2个
            # self.neus_render_iterations = [self.testing_iterations[i]
            #                               for i in range(0, len(self.testing_iterations), 2)]
            # 如果为空，使用默认值
            # if not self.neus_render_iterations:
            self.neus_render_iterations = [25000, 45000]
            
            # 创建高斯模型
            self.gaussians = GaussianModel(self.lp.feat_dim, self.lp.n_offsets, self.lp.voxel_size, 
                                          self.lp.update_depth, self.lp.update_init_factor, 
                                          self.lp.update_hierachy_factor, self.lp.use_feat_bank, 
                                          self.lp.use_tcnn)
            self.ema_loss_for_log = 0.0
            self.dataset_size=0
            self.last_iteration_time=0

            # 直接使用预训练的结果 Using a pretrained Scaffold-GS
            if self.config.model.using_pretrain:
                # 实例化场景，主要于三维场景几何有关，负责维护相机位置和高斯点
                self.scene = Scene(self.lp, self.gaussians, load_iteration=15000, shuffle=False, 
                                  if_pretrain=self.config.model.using_pretrain,
                                  pretrain_path=self.config.model.using_pretrain_path,
                                  given_scale=self.config.dataset.neuralangelo_scale,
                                  given_center=self.config.dataset.neuralangelo_center)
                self.gaussians.training_setup(self.op)
                self.gaussians.update_learning_rate(15000)
                self.progress_bar = tqdm(range(15000, self.op.iterations), desc="Training progress")
                self.viewpoint_stack = self.scene.getTrainCameras().copy()
                self.viewpoint_candidate = self.scene.getTrainCameras().copy()
            # 从头先训练scaffold Pretrain Scaffold-GS from scratch.
            else:
                self.progress_bar = tqdm(range(0, self.op.iterations), desc="Training progress")               
                # 实例化场景，主要与三维场景几何有关，负责维护相机位置和高斯点
                self.scene = Scene(self.lp, self.gaussians, 
                                   shuffle=False,given_scale=self.config.dataset.neuralangelo_scale,
                                   given_center=self.config.dataset.neuralangelo_center)
                
                self.gaussians.training_setup(self.op)
                self.viewpoint_stack = self.scene.getTrainCameras().copy()
                self.viewpoint_candidate = self.scene.getTrainCameras().copy()
                # pretrain scaffold gs
                self.pretrain_gs()
          
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays

    def forward(self, batch, gs_depth=None, use_depth_guide=False):
        """
        调用instant-nsr模型的前向传递函数, 进行SDF的预测和渲染.
        """
        return self.model(batch['rays'], gs_depth, use_depth_guide)
    
    def preprocess_data(self, batch, stage):
        """
        预处理batch中的数据, 选取图片/像素点, 构建光线并准备RGB值/前景掩码.
        """
        
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            # 如果batch中没有指定训练的图像索引，则随机选择一个图像索引
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
                
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)

        if stage in ['train']:
            # 随机选点，在图片上随机采样train_num_rays个像素点
            c2w = self.dataset.all_c2w[index]
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            # 构建光线：根据相机位姿和方向计算光线的起点和方向
            rays_o, rays_d = get_rays(directions, c2w)

            # Ground Truth 像素颜色：直接从显存中的全量数据 (self.dataset.all_images) 读取 RGB 值
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            # 每个训练图的前景掩码（提前处理好存储在显存中）
            # todo 这里是否可以优化，不直接放在显存中
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)

        else:
            # Ensure indices are on the same device as the indexed tensors
            index_c2w = index.to(self.dataset.all_c2w.device)
            c2w = self.dataset.all_c2w[index_c2w][0]

            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                index_dir = index.to(self.dataset.directions.device)
                directions = self.dataset.directions[index_dir][0] 
            rays_o, rays_d = get_rays(directions, c2w)
            
            index_img = index.to(self.dataset.all_images.device)
            rgb = self.dataset.all_images[index_img].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            
            index_mask = index.to(self.dataset.all_fg_masks.device)
            fg_mask = self.dataset.all_fg_masks[index_mask].view(-1).to(self.rank)


        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        if self.dataset.apply_mask:
            # 前景掩码应用于RGB值
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
        if stage in ['train']:
            batch.update({
                'rays': rays,
                'rgb': rgb,
                'fg_mask': fg_mask,
                'used_index': index,
                'used_y': y,
                'used_x': x,
            }) 
        else:
            batch.update({
                'rays': rays,
                'rgb': rgb,
                'fg_mask': fg_mask,
            })

    #Only training Scaffold-GS for the first 'pretrain_step' iterations
    def pretrain_gs(self):
        datasetname=self.args.source_path.split('/')[-1]
        for iteration in range(0, self.pretrain_step + 1): 
            iter_start = torch.cuda.Event(enable_timing = True)
            iter_end = torch.cuda.Event(enable_timing = True)
            iter_start.record()
            self.gaussians.update_learning_rate(iteration)

            if not self.viewpoint_stack:
                self.viewpoint_stack = self.scene.getTrainCameras().copy()
                self.dataset_size = len(self.viewpoint_stack)
                
            viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))
            random_background = torch.rand(3).cuda()
            # voxel_visible_mask = gaussian_renderer.prefilter_voxel(viewpoint_cam, self.gaussians, self.piplin, self.background)
            voxel_visible_mask = gaussian_renderer.prefilter_voxel(viewpoint_cam, self.gaussians, self.piplin, random_background)

            retain_grad = (iteration < self.op.update_until and iteration >= 0)

            time2=time.time()

            # render_pkg = gaussian_renderer.render(viewpoint_cam, self.gaussians, self.piplin, self.background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
            render_pkg = gaussian_renderer.render(viewpoint_cam, self.gaussians, self.piplin, random_background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)

            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
            time3=time.time()
            time_2=time3-time2
            self.tb_writer.add_scalar(f'{datasetname}'+'/pure_gs_forward', time_2, iteration)
            if iteration <= self.op.iterations:
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                scaling_reg = scaling.prod(dim=1).mean()
                loss_gaussian= (1.0 - self.op.lambda_dssim) * Ll1 + self.op.lambda_dssim * (1.0 - ssim(image, gt_image)) + 0.01*scaling_reg
                loss_gaussian.backward()

                time4=time.time()
                time_3=time4-time3
                self.tb_writer.add_scalar(f'{datasetname}'+'/pure_gs_backward', time_3, iteration)
                iter_end.record()

                with torch.no_grad():
                    self.ema_loss_for_log = 0.4 * loss_gaussian.item() + 0.6 * self.ema_loss_for_log
                    if iteration % 10 == 0:
                        self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                        self.progress_bar.update(10)
                    if iteration == self.op.iterations:
                        self.progress_bar.close()
                    training_report(self.tb_writer, self.args.source_path.split('/')[-1], iteration, Ll1, loss_gaussian, l1_loss, iter_start.elapsed_time(iter_end), self.args.test_iterations, self.scene, gaussian_renderer.render, (self.piplin, self.background), None, self.loggger)
                    if (iteration in self.saving_iterations):
                        self.loggger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                        self.scene.save(iteration)

                    if iteration < self.op.update_until and iteration > self.op.start_stat:
                        self.gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                        
                        # densification
                        if iteration > self.op.update_from and iteration % 100 ==0: # opt.update_intern_interval == 0:
                            self.gaussians.adjust_anchor(check_interval=self.op.update_interval, extent=self.scene.cameras_extent, success_threshold=self.op.success_threshold, grad_threshold=self.op.densify_grad_threshold, min_opacity=self.op.min_opacity)

                    # Optimizer step
                    if iteration < self.op.iterations:
                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none = True)

                    if (iteration in self.args.checkpoint_iterations):
                        # if 'debug' not in scene.model_path:
                        self.loggger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                        torch.save((self.gaussians.capture(), iteration), self.scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    # vector similarity
    def cos_similarity_loss(self, a, b):
        return 1.0-((a*b).sum(dim=-1) / (a.norm(dim=-1)*b.norm(dim=-1)+1e-8)).abs().mean()

    # Training step for both Scaffold-GS and Instant-nsr
    def training_step(self, batch, batch_idx):
        """
        这里是 GSDF 项目的核心：双分支训练(Scaffold-GS和Instant-NSR)

        明确每一训练步数下, 如何渲染/两个branch对齐/计算loss联合优化
        
        :param self: 说明
        :param batch: 说明
        :param batch_idx: 说明
        """
        random_background = torch.rand(3).cuda()
        datasetname=self.args.source_path.split('/')[-1]
        time1=time.time()

        if self.last_iteration_time!=0:
            time_5=time1-self.last_iteration_time
            self.tb_writer.add_scalar(f'{datasetname}'+'/time_5', time_5, self.current_epoch_set)

        self.current_epoch_set=self.current_epoch_set+1

        #inite loss of gs
        loss_gaussian=0

        # Reducing the normal and depth loss weight in the later iterations
        # 在训练的后期迭代中减少法线和深度损失的权重
        if self.current_epoch_set > 15000 and not self.loss_weight_reduced:
        # if self.current_epoch_set > (self.op.iterations-15000)/2:
            self.config.system.loss.normal_w = self.config.system.loss.normal_w/10
            self.config.system.loss.depth_w = self.config.system.loss.depth_w/10
            self.loss_weight_reduced = True

        # --- GS 分支（Gaussian Splatting）---
        # Training for Scaffold-GS
        if self.config.model.if_gaussian:
            # 计算GS当前全部累加的迭代数
            current_epoch_gs = self.current_epoch_set + self.pretrain_step
            iter_start = torch.cuda.Event(enable_timing = True)
            iter_end = torch.cuda.Event(enable_timing = True)
            iter_start.record()
            self.gaussians.update_learning_rate(current_epoch_gs)

            # Get the same image index as Instant-nsr
            # 1. 获取相机视角
            viewpoint_cam = self.scene.getTrainCameras()[batch['used_index']]

            # Get the same pixel indexes as Instant-nsr
            yy = batch['used_y']
            xx = batch['used_x']

            ## Forward of Scaffold-GS
            # 2. filter 3D Gaussians out of frumstum.
            # voxel_visible_mask = gaussian_renderer.prefilter_voxel(viewpoint_cam, self.gaussians, self.piplin, self.background)
            voxel_visible_mask = gaussian_renderer.prefilter_voxel(viewpoint_cam, self.gaussians, self.piplin, random_background)

            # Determine whether to retain gradients for updating Gaussians
            retain_grad = (current_epoch_gs < self.op.update_until and current_epoch_gs >= 0)

            # 3. Gaussian 渲染（生成 RGB + Depth + Normal）
            render_pkg = gaussian_renderer.render(viewpoint_cam, self.gaussians, self.piplin, random_background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, out_depth=True, return_normal=True, radius=self.config.model.radius)

            # render_pkg = gaussian_renderer.render(viewpoint_cam, self.gaussians, self.piplin, self.background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, out_depth=True, return_normal=True, radius=self.config.model.radius)
            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity_gs, gs_depth_hand,gs_normal = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"], render_pkg["depth_hand"], render_pkg["gs_normal"]

            # 4. 提取深度图（用于指导 SDF）
            gs_depth = gs_depth_hand.mean(dim=0,keepdim=True).permute(1, 2, 0)
            picked_gs_depth = gs_depth[yy,xx]   # 选择与 SDF 相同的像素
            # 5. 提取法线图（用于指导 SDF）
            gs_normal = gs_normal.permute(1, 2, 0)
            picked_gs_normal = gs_normal[yy,xx]

            time2=time.time()
            time_1=time2-time1
            
            self.tb_writer.add_scalar(f'{datasetname}'+'/time_1', time_1, self.current_epoch_set)

        # --- SDF 分支（Neural SDF）---
        # Using predicted depth of Scaffold-GS to guide the ray sampling of Instant-nsr after warm-up of Instant-nsr.
        # 4. Detach GS 深度（因为这个gsdepth后传入到sdf网络中进行位置的采样，为了避免后续sdf的梯度更新泄漏到gs中，需要detach）
        picked_gs_depth_dt = picked_gs_depth.detach()

        # 在Instant-nsr的预热期内（5K步）不使用深度引导采样
        # if self.current_epoch_set > self.config.model.geometry.xyz_encoding_config.start_step and self.current_epoch_set%500>100:
        if self.current_epoch_set > self.config.model.geometry.xyz_encoding_config.start_step:
            # 调用Instant-nsr的前向传播，即NeusSystem的forward函数
            out = self(batch, picked_gs_depth_dt, use_depth_guide=True) 
        else:
            out = self(batch, picked_gs_depth_dt, use_depth_guide=False)
        time3=time.time()
        time_2=time3-time2
        self.tb_writer.add_scalar(f'{datasetname}'+'/time_2', time_2, self.current_epoch_set)

        # If all the sampled pixels are belong to background, skiping this training iteration. 
        if out['zero_samples']==True:
            return None
        
        # loss of Instant-NSR
        loss = 0.

        # predicted normal and depth of Scaffold-GS, taken as GT of the Instant-NSR side
        # out['rays_valid'][...,0]可能是一个mask，表示哪些光线是有效的，然后在gs的预测中选择对应的深度和法线
        fixed_picked_gs_normal = picked_gs_normal[out['rays_valid'][...,0]].detach()
        fixed_picked_gs_depth = picked_gs_depth[out['rays_valid'][...,0]].detach()

        # The depth loss for the Instant-nsr.
        diff_neus = torch.abs(out['depth'][out['rays_valid'][...,0]] - fixed_picked_gs_depth)

        # Filter out the huge depth differents, which could be the impact of background.
        # 选择性地过滤掉巨大的深度差异，这些差异可能是背景的影响。
        # 具体来说，根据当前场景的半径来设定一个阈值(depth_ratio)，超过这个阈值的深度差异将被视为异常值并被忽略。
        if self.current_epoch_set > self.config.model.geometry.xyz_encoding_config.start_step:
            depth_ratio = 10.0
        else:
            depth_ratio = 2.0
        diff_neus[diff_neus > self.config.model.radius/depth_ratio] = 0
        diff_neus_count = (diff_neus>0).sum()
        loss_depth_L1 = diff_neus.sum() / (diff_neus_count+1e-8)
        # normalzied the depth loss by the frontground size.
        # 这里的loss_depth_L1 只会影响到Instant-nsr的训练
        loss += loss_depth_L1 * self.C(self.config.system.loss.depth_w)/self.config.model.radius
        self.log('train/loss_depth_L1_neus', float(loss_depth_L1/self.config.model.radius))

        # The normal loss for the Instant-nsr is only taken into account after the warmup period.
        # Instant-nsr的法线损失仅在预热期后才被考虑。 同样，法线损失也只会影响Instant-nsr的训练
        if self.current_epoch_set > self.config.model.geometry.xyz_encoding_config.start_step:
            normal_diff = self.cos_similarity_loss(fixed_picked_gs_normal,out['comp_normal'][out['rays_valid'][...,0]])
            loss +=  normal_diff * self.config.system.loss.normal_w
        else:
            normal_diff = self.cos_similarity_loss(fixed_picked_gs_normal,out['comp_normal'][out['rays_valid'][...,0]])
            loss +=  normal_diff * 0.0
        self.log('train/normal_loss_neus', normal_diff)

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        # RGB L1 loss
        # nens branch 的 RGB L1 损失，与GT RGB 进行对齐
        loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)        

        # Eikonal loss for SDF regularization
        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)

        # Curvature loss, Note that the curvature loss weight is adaptived to the training iteration.
        # 曲率损失，让SDF更平滑
        if self.C(self.config.system.loss.lambda_smoothing)>0:
            loss_smoothing = out['smoothing'].abs().mean()
            self.log('train/loss_smoothing', loss_smoothing)
            
            loss+=  loss_smoothing * self.C(self.config.system.loss.lambda_smoothing)
          
        # todo 这里的是什么约束
        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        # 输出lambda系数
        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        # log
        self.log('train/inv_s', out['inv_s'], prog_bar=True)
        
        time4=time.time()
        time_3=time4-time3
        self.tb_writer.add_scalar(f'{datasetname}'+'/time_3', time_3, self.current_epoch_set)

        # Calculate the loss of Scaffold-GS
        if self.config.model.if_gaussian:
            if current_epoch_gs <= self.op.iterations:
                gt_image = viewpoint_cam.original_image.cuda()

                # RGB L1 loss
                Ll1 = l1_loss(image, gt_image)
                self.log('train/GS_render_loss', Ll1)

                # Scalling loss 
                scaling_reg = scaling.prod(dim=1).mean()
                self.log('train/GS_scaling_reg', scaling_reg)
                
                # Neus指导的深度和法线一致性损失
                # Predicted depth and normal of Instant-NSR, taken as GT of the GS side.
                fixed_neus_picked_depth = out['depth'][out['rays_valid'][...,0]].detach()
                fixed_neus_picked_normal = out['comp_normal'][out['rays_valid'][...,0]].detach()
               
                #SSIM loss
                ssim_loss = 1.0 - ssim(image, gt_image)
                self.log('train/GS_ssim_loss', ssim_loss)

                # Normal loss of the GS side.
                # Ignore normal loss in the warmup period of Instant-NSR 
                if self.current_epoch_set < self.config.model.geometry.xyz_encoding_config.start_step:                
                    normal_loss_gs = 0.0
                else:
                    normal_loss_gs = self.cos_similarity_loss(picked_gs_normal[out['rays_valid'][...,0]],fixed_neus_picked_normal)* self.config.system.loss.normal_w

                self.log('train/GS_normal_loss_gs', normal_loss_gs)
                # depth loss of GS side
                diff = torch.abs(fixed_neus_picked_depth - picked_gs_depth[out['rays_valid'][...,0]])
                
                # Filter out the huge depth differents, which could be the impact of background.
                # 同样有背景过滤
                depth_ratio = 10.0
                
                diff[diff > self.config.model.radius/depth_ratio] = 0
                
                diff_count = (diff>0.0).sum()

                loss_depth_L1_gs = diff.sum() / (diff_count+1e-8)
                self.log('train/GS_loss_depth', loss_depth_L1_gs)

                # normalzied the depth loss by the frontground size.
                depth_loss_gs = loss_depth_L1_gs * self.C(self.config.system.loss.depth_w)/self.config.model.radius

                # GS loss and backward
                loss_gaussian= (1.0 - self.op.lambda_dssim) * Ll1 + \
                    self.op.lambda_dssim * ssim_loss + 0.01*scaling_reg + \
                        depth_loss_gs  + \
                            normal_loss_gs
                
                self.log('train/GS_loss_gaussian', float(loss_gaussian))
                time41=time.time()
                time_41=time41-time4
                self.tb_writer.add_scalar(f'{datasetname}'+'/time_41', time_41, self.current_epoch_set)

                loss_gaussian.backward()
                iter_end.record()

                time42=time.time()
                time_42=time42-time41
                self.tb_writer.add_scalar(f'{datasetname}'+'/time_42', time_42, self.current_epoch_set)

                # GS densification
                with torch.no_grad():
                    self.ema_loss_for_log = 0.4 * loss_gaussian.item() + 0.6 * self.ema_loss_for_log
                    
                    if current_epoch_gs % 10 == 0:
                        self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                        self.progress_bar.update(10)
                    
                    if current_epoch_gs == self.op.iterations:
                        self.progress_bar.close()

                    training_report(self.tb_writer, self.args.source_path.split('/')[-1], current_epoch_gs, Ll1, loss_gaussian, l1_loss, iter_start.elapsed_time(iter_end), self.args.test_iterations, self.scene, gaussian_renderer.render, (self.piplin, self.background), None, self.loggger)

                    # NeuS分支的渲染报告（使用独立的测试间隔，输出到同一个tensorboard）
                    neus_rendering_report(
                        neus_system=self,
                        dataset_name=self.args.source_path.split('/')[-1],
                        iteration=current_epoch_gs,
                        testing_iterations=self.neus_render_iterations,
                        tb_writer=self.tb_writer,  # 与GS分支共享同一个tb_writer
                        logger=self.loggger
                    )

                    if (current_epoch_gs in self.saving_iterations):
                        self.loggger.info("\n[ITER {}] Saving Gaussians".format(current_epoch_gs))
                        self.scene.save(current_epoch_gs)
                    time43=time.time()
                    time_43=time43-time42
                    self.tb_writer.add_scalar(f'{datasetname}'+'/time_43', time_43, self.current_epoch_set)
                    if current_epoch_gs < self.op.update_until and current_epoch_gs > self.op.start_stat: 
                        
                        self.gaussians.training_statis(viewspace_point_tensor, opacity_gs, visibility_filter, offset_selection_mask, voxel_visible_mask, grad_threshold=self.op.densify_grad_threshold)
                        
                        if current_epoch_gs > self.op.update_from and current_epoch_gs % 100 == 0: # opt.update_intern_interval == 0:
                            if self.geometry_awared_control:
                                # Original density control
                                self.gaussians.adjust_anchor(check_interval=self.op.update_interval, extent=self.scene.cameras_extent, success_threshold=self.op.success_threshold, grad_threshold=self.op.densify_grad_threshold, min_opacity=self.op.min_opacity, growing_weight=self.config.system.growing_weight)
                                
                            else:
                                #  Density control guided by predicted sdf
                                if self.current_epoch_set > self.config.model.geometry.xyz_encoding_config.start_step:
                                    # guide density control after warmup of Instant-nsr
                                    # Identify the 3D Gaussians in the frontground
                                    scaling = self.gaussians.get_scaling[:,:3]
                                    scaling_repeat = scaling.unsqueeze(dim=1).repeat([1, self.gaussians.n_offsets, 1]).view([-1, 3]) 
                                    gs_positions = self.gaussians.get_anchor.unsqueeze(dim=1).repeat([1, self.gaussians.n_offsets, 1]).view([-1, 3]) + self.gaussians._offset.view([-1, 3])*scaling_repeat

                                    min_point = torch.tensor([-self.config.model.radius, -self.config.model.radius, -self.config.model.radius],device=gs_positions.device)
                                    max_point = torch.tensor([self.config.model.radius, self.config.model.radius, self.config.model.radius],device=gs_positions.device)
                                    inside_box = (gs_positions > min_point) & (gs_positions < max_point)
                                    inside_box = inside_box.all(dim=1)

                                    # 在区域内的3D高斯点位置，后续传入SDF网络计算inside_xyz_sdf
                                    inside_positions = gs_positions[inside_box]
                                    # set the sdf of 3D gaussians in the background to 100000.
                                    xyz_sdf = torch.ones(gs_positions.shape[0]).to(gs_positions.device)*100000

                                    # # 分批查询，减少峰值显存
                                    # batch_size = 100000  # 每批10万点
                                    # inside_xyz_sdf_list = []
      
                                    # for i in range(0, len(inside_positions), batch_size):
                                    #     batch_pos = inside_positions[i:i+batch_size]
                                    #     batch_sdf = self.model.geometry(batch_pos, with_grad=False, with_feature=False)
                                    #     inside_xyz_sdf_list.append(batch_sdf.detach())  # 立即detach
                                    
                                    # calculate the sdf of 3D Gaussians in the frontground.
                                    inside_xyz_sdf = self.model.geometry(inside_positions, with_grad=False, with_feature=False)

                                    xyz_sdf[inside_box] = inside_xyz_sdf

                                    # calculate the sdf of anchor points in the frontground
                                    anchor_positions = self.gaussians.get_anchor
                                    anchor_inside_box = (anchor_positions > min_point) & (anchor_positions < max_point)
                                    anchor_inside_box = anchor_inside_box.all(dim=1)
                                    anchor_sdf = self.model.geometry(anchor_positions, with_grad=False, with_feature=False)
                                      
                                else:
                                    # using the original density control in the warmup of Instant-nsr
                                    xyz_sdf=None
                                    anchor_sdf=None
                                    inside_box=None
                                    anchor_inside_box=None
                                
                                # 根据上面的sdf信息调整高斯点密度
                                self.gaussians.adjust_anchor(check_interval=self.op.update_interval, extent=self.scene.cameras_extent, success_threshold=self.op.success_threshold, grad_threshold=self.op.densify_grad_threshold, min_opacity=self.op.min_opacity, xyz_sdf=xyz_sdf, anchor_sdf=anchor_sdf, inside_box=inside_box, anchor_inside_box=anchor_inside_box, growing_weight=self.config.system.growing_weight)

                    elif current_epoch_gs == self.op.update_until:
                        del self.gaussians.opacity_accum
                        del self.gaussians.offset_gradient_accum
                        del self.gaussians.offset_denom
                        torch.cuda.empty_cache()
                    time44=time.time()
                    time_44=time44-time43
                    self.tb_writer.add_scalar(f'{datasetname}'+'/time_44', time_44, self.current_epoch_set)
                    
                    # Optimizer step
                    if current_epoch_gs < self.op.iterations:
                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none = True)

                    if (current_epoch_gs in self.args.checkpoint_iterations):
                        self.loggger.info("\n[ITER {}] Saving Checkpoint".format(current_epoch_gs))
                        torch.save((self.gaussians.capture(), current_epoch_gs), self.scene.model_path + "/chkpnt" + str(current_epoch_gs) + ".pth")
                    time45=time.time()
                    time_45=time45-time44
                    self.tb_writer.add_scalar(f'{datasetname}'+'/time_45', time_45, self.current_epoch_set)
        
        self.last_iteration_time=time.time()

        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)
            self.export()         

    def test_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }      
    
    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    

            self.export()
    
    def export(self):
        mesh = self.model.export(self.config.export)
        # if self.config.model.if_gaussian:
        #     tc = torch.tensor(self.scene.center).reshape(3)
        #     pts = mesh['v_pos']
        #     pts = pts * self.scene.scale
        #     pts += tc
        #     mesh['v_pos'] = pts
      
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.ply",
            **mesh
        )        

    def prepare_output_and_logger(self, args, opt, pipe):   

        if not args.model_path:
            if os.getenv('OAR_JOB_ID'):
                unique_str=os.getenv('OAR_JOB_ID')
            else:
                unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[0:10])
                
        # Set up output folder
        print("Output folder: {}".format(args.model_path))
        os.makedirs(args.model_path, exist_ok = True)
        with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(args))))

        with open(os.path.join(args.model_path, "GS_cfg_args_origin"), 'w') as cfg_log_f:
            cfg_log_f.write(' '.join(sys.argv))

        with open(os.path.join(args.model_path, "GS_cfg_args_opt"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(opt))))

        with open(os.path.join(args.model_path, "GS_cfg_args_pipe"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(pipe))))

        # Create Tensorboard writer
        tb_writer = None
        if TENSORBOARD_FOUND:
            tb_writer = SummaryWriter(args.model_path)
        else:
            print("Tensorboard not available: not logging progress")
        return tb_writer

