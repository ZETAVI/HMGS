# NeuS SDF分支训练框架深度剖析

> 结合PyTorch Lightning框架的SDF训练系统详解
>
> 分析日期：2025-12-31
> 项目：HMGS (GSDF Implementation)

## 目录

- [一、PyTorch Lightning框架概述与使用](#一pytorch-lightning框架概述与使用)
- [二、NeuSSystem完整生命周期](#二neussystem完整生命周期)
- [三、数据加载与预处理流程](#三数据加载与预处理流程)
- [四、模型架构详解](#四模型架构详解)
- [五、Volume Rendering实现](#五volume-rendering实现)
- [六、训练循环与优化](#六训练循环与优化)
- [七、损失函数与正则化](#七损失函数与正则化)
- [八、完整训练流程图](#八完整训练流程图)

---

## 一、PyTorch Lightning框架概述与使用

### 1.1 为什么使用PyTorch Lightning？

PyTorch Lightning是PyTorch的高级封装框架，主要优势：

1. **解耦训练逻辑**：将模型代码、训练逻辑、数据加载分离
2. **自动化训练流程**：自动处理设备分配、梯度累积、日志记录等
3. **易于扩展**：通过Callbacks系统方便地添加功能
4. **分布式训练支持**：简化多GPU、多节点训练

### 1.2 Lightning在本项目中的架构

```
launch.py (主入口)
    ↓
pytorch_lightning.Trainer
    ↓
    ├─ NeuSSystem (LightningModule)
    │  ├─ __init__(): 初始化模型
    │  ├─ prepare(): 准备训练前的设置
    │  ├─ forward(): 前向传播
    │  ├─ training_step(): 训练步骤
    │  ├─ validation_step(): 验证步骤
    │  ├─ configure_optimizers(): 配置优化器
    │  └─ on_xxx_batch_start(): 各种钩子函数
    │
    └─ ColmapDataModule (LightningDataModule)
       ├─ setup(): 准备数据集
       ├─ train_dataloader(): 训练数据加载器
       ├─ val_dataloader(): 验证数据加载器
       └─ test_dataloader(): 测试数据加载器
```

### 1.3 核心组件对比

| PyTorch原生 | PyTorch Lightning | 本项目实现 |
|------------|-------------------|-----------|
| 模型定义 | `nn.Module` | `LightningModule` | `NeuSSystem` |
| 训练循环 | 手动for循环 | `trainer.fit()` | 自动执行 |
| 数据加载 | `DataLoader` | `LightningDataModule` | `ColmapDataModule` |
| 优化器 | 手动创建 | `configure_optimizers()` | 自动调用 |
| 日志记录 | 手动tensorboard | `self.log()` | 自动记录 |

### 1.4 launch.py中的Trainer配置

**位置**：`launch.py:50-176`

```python
def main():
    # 1. 加载配置
    config = load_config(args.config, cli_args=extras)

    # 2. 创建数据模块 (LightningDataModule)
    dm = instant_nsr.datasets.make(config.dataset.name, config.dataset)
    # 实际创建的是 ColmapDataModule

    # 3. 创建系统模块 (LightningModule)
    system = instant_nsr.systems.make(config.system.name, config, ...)
    # 实际创建的是 NeuSSystem

    # 4. 配置Callbacks
    callbacks = [
        ModelCheckpoint(dirpath=config.ckpt_dir, **config.checkpoint),
        LearningRateMonitor(logging_interval='step'),
        ConfigSnapshotCallback(config, config.config_dir),
        CustomProgressBar(refresh_rate=1),
    ]

    # 5. 配置Loggers
    loggers = [
        TensorBoardLogger(args.runs_dir, name=config.name, version=config.trial_name),
        CSVLogger(config.exp_dir, name=config.trial_name, version='csv_logs')
    ]

    # 6. 创建Trainer
    trainer = Trainer(
        devices=n_gpus,              # GPU数量
        accelerator='gpu',           # 使用GPU
        callbacks=callbacks,         # 回调函数列表
        logger=loggers,              # 日志记录器列表
        strategy='ddp_find_unused_parameters_false',  # 分布式策略
        **config.trainer             # 其他配置（max_steps, val_check_interval等）
    )

    # 7. 开始训练
    if args.train:
        trainer.fit(system, datamodule=dm)
```

**config.trainer典型配置**（configs/tnt/barn.yaml）：

```yaml
trainer:
  max_steps: 30000           # 最大训练步数
  log_every_n_steps: 100     # 每100步记录一次日志
  num_sanity_val_steps: 0    # 训练前验证步数（0表示不验证）
  val_check_interval: 1000   # 每1000步执行一次验证
  limit_train_batches: 1.0   # 使用100%训练数据
  limit_val_batches: 2       # 验证时只用2个batch
  enable_progress_bar: true  # 显示进度条
  precision: 16              # 使用混合精度训练（FP16）
```

---

## 二、NeuSSystem完整生命周期

### 2.1 类继承关系

```python
# instant_nsr/systems/neus.py:141
class NeuSSystem(BaseSystem):
    ...

# instant_nsr/systems/base.py:11
class BaseSystem(pl.LightningModule):
    ...
```

**继承链**：`NeuSSystem` → `BaseSystem` → `pytorch_lightning.LightningModule` → `torch.nn.Module`

### 2.2 生命周期阶段

#### **阶段1：初始化 (__init__)**

**位置**：`instant_nsr/systems/neus.py:148-225`

```python
def __init__(self, config):
    super().__init__(config)  # 调用BaseSystem.__init__

    # === 1. 训练计数器 ===
    self.current_epoch_set = 0      # 全局训练步数（跨GS和SDF）
    self.pretrain_step = 15000      # GS预训练步数
    self.geometry_awared_control = False  # 几何感知控制开关

    # === 2. GS分支初始化（如果启用） ===
    if self.config.model.if_gaussian:
        # 2.1 创建参数解析器
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)      # 场景参数
        op = OptimizationParams(parser)  # 优化参数
        pp = PipelineParams(parser)   # 渲染管线参数

        # 2.2 解析命令行参数
        args = parser.parse_args(fake_input)

        # 2.3 创建日志器
        self.loggger = get_logger(args.model_path)
        self.tb_writer = self.prepare_output_and_logger(lp.extract(args))

        # 2.4 创建GaussianModel
        self.gaussians = GaussianModel(
            self.lp.feat_dim,           # 特征维度
            self.lp.n_offsets,          # 每个anchor的offset数量
            self.lp.voxel_size,         # voxel大小
            self.lp.update_depth,       # 更新深度
            self.lp.update_init_factor,
            self.lp.update_hierachy_factor,
            self.lp.use_feat_bank,      # 是否使用feature bank
            self.lp.use_tcnn            # 是否使用tiny-cuda-nn
        )

        # 2.5 创建Scene（加载或初始化点云）
        if self.config.model.using_pretrain:
            # 从预训练模型加载
            self.scene = Scene(self.lp, self.gaussians,
                              load_iteration=15000,
                              if_pretrain=True,
                              pretrain_path=self.config.model.using_pretrain_path)
        else:
            # 从头开始训练
            self.scene = Scene(self.lp, self.gaussians, shuffle=False)
            # 执行GS预训练
            self.pretrain_gs()

        # 2.6 设置优化器
        self.gaussians.training_setup(self.op)

        # 2.7 创建进度条和视点栈
        self.progress_bar = tqdm(range(0, self.op.iterations))
        self.viewpoint_stack = self.scene.getTrainCameras().copy()
```

**BaseSystem.__init__调用链**：

```python
# instant_nsr/systems/base.py:14-19
def __init__(self, config):
    super().__init__()  # 调用LightningModule.__init__
    self.config = config
    self.rank = get_rank()  # 获取当前GPU rank
    self.prepare()          # 准备数据集
    self.model = models.make(self.config.model.name, self.config.model)
    # 创建NeuSModel实例
```

**prepare()做了什么**：

```python
# instant_nsr/systems/base.py:21-22
def prepare(self):
    self.criterions = criterions.make(self.config.system.criterion.name,
                                     self.config.system.criterion)
    # 创建损失函数
```

**models.make()创建NeuSModel**：

```python
# instant_nsr/models/__init__.py
def make(name, config):
    model = models[name](config)  # models['neus'] = NeuSModel
    return model

# instant_nsr/models/neus.py:44-81
class NeuSModel(BaseModel):
    def setup(self):
        # 创建geometry、texture、variance网络
        self.geometry = models.make('volume-sdf-sg', self.config.geometry)
        self.texture = models.make('volume-radiance', self.config.texture)
        self.variance = VarianceNetwork(self.config.variance)

        # 如果启用背景
        if self.config.learned_background:
            self.geometry_bg = models.make('volume-density', ...)
            self.texture_bg = models.make('volume-radiance', ...)

        # 创建occupancy grid（用于加速采样）
        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(...)
```

#### **阶段2：准备训练 (on_train_batch_start)**

**位置**：`instant_nsr/systems/base.py:37-38`

```python
def on_train_batch_start(self, batch, batch_idx):
    # 更新全局步数
    self.dataset = self.trainer.datamodule.train_dataloader().dataset
    self.global_step = self.trainer.global_step

    # 更新模型参数（如progressive hash grid的激活层级）
    self.model.update_step(self.current_epoch, self.global_step)
```

**update_step做什么**：

```python
# instant_nsr/models/neus.py:83-95
def update_step(self, epoch, global_step):
    # 更新variance network
    update_module_step(self.variance, epoch, global_step)

    # 更新geometry网络（重要！激活新的hash grid层级）
    update_module_step(self.geometry, epoch, global_step)

    # 更新texture网络
    update_module_step(self.texture, epoch, global_step)

    # 如果有背景，也更新
    if self.config.learned_background:
        update_module_step(self.geometry_bg, epoch, global_step)
        update_module_step(self.texture_bg, epoch, global_step)

    # 更新occupancy grid（如果启用）
    if self.config.grid_prune and not self.config.gs_sampling:
        self.occupancy_grid.every_n_step(...)
```

**geometry的update_step**（关键！）：

```python
# instant_nsr/models/geometry.py:267-289
def update_step(self, epoch, global_step):
    # 更新encoding和network
    update_module_step(self.encoding, epoch, global_step)
    update_module_step(self.network, epoch, global_step)

    # === 渐进式激活Hash Grid ===
    if self.finite_difference_eps == 'progressive':
        # 计算当前应激活的层级
        current_level = min(
            start_level + max(global_step - start_step, 0) // update_steps,
            n_levels
        )
        # start_level=8, start_step=5000, update_steps=2000

        # 计算当前分辨率
        grid_res = base_resolution * per_level_scale^(current_level - 1)

        # 计算epsilon（用于finite difference）
        grid_size = 2 * radius / grid_res
        self._finite_difference_eps = grid_size
```

#### **阶段3：数据预处理 (preprocess_data)**

**位置**：`instant_nsr/systems/neus.py:237-304`

```python
def preprocess_data(self, batch, stage):
    """
    将batch数据转换为rays格式
    """
    # === 1. 选择图像index ===
    if 'index' in batch:  # 验证/测试阶段
        index = batch['index']
    else:  # 训练阶段
        if self.config.model.batch_image_sampling:
            # 从所有图像中随机采样光线
            index = torch.randint(0, len(self.dataset.all_images),
                                 size=(self.train_num_rays,))
        else:
            # 从单张图像中采样光线
            index = torch.randint(0, len(self.dataset.all_images), size=(1,))

    # === 2. 生成光线 ===
    if stage in ['train']:
        # 2.1 随机采样像素坐标
        x = torch.randint(0, self.dataset.w, size=(self.train_num_rays,))
        y = torch.randint(0, self.dataset.h, size=(self.train_num_rays,))

        # 2.2 获取相机参数
        c2w = self.dataset.all_c2w[index]  # camera-to-world矩阵

        # 2.3 获取光线方向
        if self.dataset.directions.ndim == 3:  # (H, W, 3)
            directions = self.dataset.directions[y, x]
        elif self.dataset.directions.ndim == 4:  # (N, H, W, 3)
            directions = self.dataset.directions[index, y, x]

        # 2.4 计算光线原点和方向
        rays_o, rays_d = get_rays(directions, c2w)

        # 2.5 获取GT颜色和mask
        rgb = self.dataset.all_images[index, y, x]
        fg_mask = self.dataset.all_fg_masks[index, y, x]

    else:  # 验证/测试：渲染整张图像
        c2w = self.dataset.all_c2w[index][0]
        directions = self.dataset.directions  # 所有像素
        rays_o, rays_d = get_rays(directions, c2w)
        rgb = self.dataset.all_images[index].view(-1, 3)
        fg_mask = self.dataset.all_fg_masks[index].view(-1)

    # === 3. 合并rays ===
    rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)
    # rays: [N, 6] 前3维是origin，后3维是direction（归一化）

    # === 4. 设置背景颜色 ===
    if stage in ['train']:
        if self.config.model.background_color == 'white':
            self.model.background_color = torch.ones((3,))
        elif self.config.model.background_color == 'random':
            self.model.background_color = torch.rand((3,))
    else:
        self.model.background_color = torch.ones((3,))

    # === 5. 应用mask ===
    if self.dataset.apply_mask:
        rgb = rgb * fg_mask[...,None] + \
              self.model.background_color * (1 - fg_mask[...,None])

    # === 6. 更新batch ===
    if stage in ['train']:
        batch.update({
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask,
            'used_index': index,  # GS分支需要这个来获取对应相机
            'used_y': y,          # GS分支需要这个来索引像素
            'used_x': x,
        })
    else:
        batch.update({
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask,
        })
```

**get_rays函数**（计算光线）：

```python
# instant_nsr/datasets/utils.py
def get_rays(directions, c2w):
    """
    Args:
        directions: (N, 3) 或 (H, W, 3)，相机空间的方向向量
        c2w: (4, 4) 或 (N, 4, 4)，camera-to-world变换矩阵

    Returns:
        rays_o: (N, 3)，光线原点（世界坐标）
        rays_d: (N, 3)，光线方向（世界坐标）
    """
    # 1. 光线原点 = 相机中心
    rays_o = c2w[..., :3, 3]  # 取平移部分

    # 2. 光线方向 = 旋转矩阵 @ 相机空间方向
    rays_d = torch.sum(directions[..., None, :] * c2w[..., :3, :3], dim=-1)
    # c2w[:3, :3]是旋转矩阵，将相机空间方向转到世界空间

    return rays_o, rays_d
```

#### **阶段4：训练步骤 (training_step)**

**位置**：`instant_nsr/systems/neus.py:381-668`

这是NeuS分支的核心！已在之前的分析文档中详细介绍，这里简化流程：

```python
def training_step(self, batch, batch_idx):
    # 1. 预处理数据
    # （已在on_train_batch_start之后自动调用preprocess_data）

    # 2. GS分支前向传播（如果启用）
    if self.config.model.if_gaussian:
        render_pkg = gaussian_renderer.render(viewpoint_cam, self.gaussians, ...)
        image_gs = render_pkg["render"]
        depth_gs = render_pkg["depth_hand"]
        normal_gs = render_pkg["gs_normal"]

    # 3. SDF分支前向传播
    if self.current_epoch_set > start_step:
        out = self(batch, depth_gs.detach(), use_depth_guide=True)
    else:
        out = self(batch, depth_gs.detach(), use_depth_guide=False)

    # 4. 计算SDF损失
    loss_sdf = L_rgb + L_depth + L_normal + L_eikonal + L_curvature

    # 5. 计算GS损失（如果启用）
    if self.config.model.if_gaussian:
        loss_gs = L_rgb + L_SSIM + L_scaling + L_depth + L_normal
        loss_gs.backward()  # GS分支反向传播

    # 6. GS密度控制（每100步）
    if current_epoch_gs % 100 == 0:
        self.gaussians.adjust_anchor(xyz_sdf=..., anchor_sdf=...)

    # 7. GS优化器步进
    self.gaussians.optimizer.step()

    # 8. 返回SDF损失（Lightning自动处理backward和optimizer step）
    return {'loss': loss_sdf}
```

**forward()调用链**：

```python
# NeuSSystem.forward (neus.py:234-235)
def forward(self, batch, gs_depth=None, use_depth_guide=False):
    return self.model(batch['rays'], gs_depth, use_depth_guide)

# NeuSModel.forward (models/neus.py:436-445)
def forward(self, rays, gs_depth=None, use_depth_guide=False):
    if self.training:
        out = self.forward_(rays, gs_depth, use_depth_guide)
    else:
        out = chunk_batch(self.forward_, self.config.ray_chunk, True, rays)
    return {**out, 'inv_s': self.variance.inv_s}

# NeuSModel.forward_ (models/neus.py:309-435) - 核心渲染逻辑
def forward_(self, rays, gs_depth=None, use_depth_guide=False):
    # ... 见第五节Volume Rendering详解
```

#### **阶段5：优化器步进 (configure_optimizers)**

**位置**：`instant_nsr/systems/base.py:143-152`

```python
def configure_optimizers(self):
    """
    Lightning自动调用，配置优化器和学习率调度器
    """
    # 1. 解析优化器配置
    optim = parse_optimizer(self.config.system.optimizer, self.model)

    ret = {'optimizer': optim}

    # 2. 如果有学习率调度器
    if 'scheduler' in self.config.system:
        ret.update({
            'lr_scheduler': parse_scheduler(self.config.system.scheduler, optim),
        })

    return ret
```

**parse_optimizer实现**：

```python
# instant_nsr/utils/optimizer.py
def parse_optimizer(config, model):
    """
    支持的优化器：Adam, AdamW, SGD, RMSprop
    """
    if config.name == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.args.lr,
            betas=config.args.betas,
            eps=config.args.eps
        )
    # ... 其他优化器
```

**典型配置**（configs/tnt/barn.yaml）：

```yaml
system:
  optimizer:
    name: Adam
    args:
      lr: 0.01
      betas: [0.9, 0.99]
      eps: 1.0e-15

  scheduler:
    name: MultiStepLR
    interval: step
    args:
      milestones: [10000, 15000, 18000]
      gamma: 0.33
```

**注意**：GS分支有自己独立的优化器！

```python
# gaussian_splatting/scene/gaussian_model.py:297-373
def training_setup(self, training_args):
    self.optimizer = torch.optim.Adam([
        {'params': [self._anchor], 'lr': training_args.position_lr_init, ...},
        {'params': [self._offset], 'lr': training_args.offset_lr_init, ...},
        {'params': [self._anchor_feat], 'lr': training_args.feature_lr, ...},
        {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, ...},
        # ... 其他参数组
    ])
```

所以实际上有**两个优化器同时工作**：
- `NeuSSystem.optimizer`：优化SDF分支（geometry, texture, variance）
- `self.gaussians.optimizer`：优化GS分支（anchors, offsets, MLPs）

#### **阶段6：验证步骤 (validation_step)**

**位置**：`instant_nsr/systems/neus.py:682-699`

```python
def validation_step(self, batch, batch_idx):
    # 1. 前向传播（不使用深度引导）
    out = self(batch)

    # 2. 计算指标
    psnr = self.criterions(out['comp_rgb_full'], batch['rgb'])

    # 3. 保存渲染结果（第一个batch）
    if batch_idx == 0:
        W, H = self.dataset.w, self.dataset.h
        img = out['comp_rgb_full'].view(H, W, 3)
        img_gt = batch['rgb'].view(H, W, 3)
        depth = out['depth'].view(H, W)
        opacity = out['opacity'].view(H, W)

        self.save_image_grid(f"it{self.global_step}-val.png",
                            [img_gt, img, depth, opacity])

    return psnr
```

#### **阶段7：验证结束 (validation_epoch_end)**

**位置**：`instant_nsr/systems/neus.py:708-722`

```python
def validation_epoch_end(self, out):
    # 聚合所有验证batch的结果
    out_mean = torch.stack(out).mean()

    # 记录到TensorBoard
    self.log('val/psnr', out_mean, prog_bar=True, rank_zero_only=True)
```

---

## 三、数据加载与预处理流程

### 3.1 LightningDataModule架构

**位置**：`instant_nsr/datasets/colmap.py:261-370`

```python
@register('colmap')
class ColmapDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        """
        在fit/test前调用一次，准备数据集
        """
        if stage in [None, 'fit']:
            # 创建训练数据集
            self.train_dataset = ColmapIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            # 创建验证数据集
            self.val_dataset = ColmapDataset(self.config, 'val')
        if stage in [None, 'test']:
            # 创建测试数据集
            self.test_dataset = ColmapDataset(self.config, 'test')

    def train_dataloader(self):
        # IterableDataset不需要指定batch_size
        return DataLoader(
            self.train_dataset,
            num_workers=0,  # IterableDataset通常用0
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
```

### 3.2 ColmapDatasetBase：数据加载核心

**位置**：`instant_nsr/datasets/colmap.py:55-169`

```python
class ColmapDatasetBase:
    def setup(self, config, split):
        """
        加载COLMAP数据
        """
        # === 1. 加载COLMAP相机参数 ===
        camdata = read_cameras_binary(
            os.path.join(self.config.root_dir, 'sparse/0/cameras.bin')
        )

        # === 2. 加载相机位姿 ===
        imdata = read_images_binary(
            os.path.join(self.config.root_dir, 'sparse/0/images.bin')
        )

        # === 3. 提取相机内参 ===
        w, h = imdata[1].width, imdata[1].height
        if config.img_downscale > 1:
            w = w // config.img_downscale
            h = h // config.img_downscale

        # 获取焦距
        if camdata[1].model == 'SIMPLE_PINHOLE':
            fx = fy = camdata[1].params[0] / config.img_downscale
        elif camdata[1].model == 'PINHOLE':
            fx = camdata[1].params[0] / config.img_downscale
            fy = camdata[1].params[1] / config.img_downscale

        # 获取主点
        cx = camdata[1].params[2] / config.img_downscale if len(camdata[1].params) > 2 else w / 2
        cy = camdata[1].params[3] / config.img_downscale if len(camdata[1].params) > 3 else h / 2

        # === 4. 计算每个像素的光线方向 ===
        directions = get_ray_directions(h, w, fx, fy, cx, cy)
        # directions: [H, W, 3]，相机空间的归一化方向向量

        # === 5. 加载所有图像和位姿 ===
        all_c2w, all_images, all_fg_masks = [], [], []

        for i, imdata_single in enumerate(imdata.values()):
            # 5.1 读取图像
            img_path = os.path.join(self.config.root_dir, 'images', imdata_single.name)
            img = Image.open(img_path)

            # 下采样
            if config.img_downscale > 1:
                img = img.resize((w, h), Image.LANCZOS)

            img = TF.to_tensor(img).permute(1, 2, 0)[..., :3]  # [H, W, 3]
            all_images.append(img)

            # 5.2 提取位姿（COLMAP格式 → OpenGL格式）
            R = imdata_single.qvec2rotmat()  # quaternion → rotation
            t = imdata_single.tvec

            # COLMAP: world-to-camera
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = t

            # 转换为camera-to-world
            c2w = np.linalg.inv(w2c)

            # 坐标系转换：COLMAP (right, down, forward) → OpenGL (right, up, back)
            c2w[:3, 1:3] *= -1

            all_c2w.append(torch.from_numpy(c2w).float())

            # 5.3 加载mask（如果有）
            if config.apply_mask:
                mask_path = os.path.join(self.config.root_dir, 'masks', ...)
                mask = Image.open(mask_path).convert('L')
                mask = TF.to_tensor(mask)[0]  # [H, W]
            else:
                mask = torch.ones_like(img[..., 0])

            all_fg_masks.append(mask)

        # === 6. 归一化相机位姿 ===
        all_c2w = torch.stack(all_c2w, dim=0)  # [N, 4, 4]

        # 计算场景中心和尺度
        if config.center_est_method == 'lookat':
            center = get_center_lookat(all_c2w)
        elif config.center_est_method == 'point':
            # 从COLMAP点云计算
            pts3d = read_points3D_binary(...)
            center = torch.from_numpy(pts3d.xyz.mean(axis=0))

        # 归一化到unit sphere
        if config.neuralangelo_scale == 0:
            # 自动计算scale
            radius = np.linalg.norm(all_c2w[:, :3, 3] - center, axis=-1).max()
            scale = 1.0 / radius
        else:
            # 使用给定scale
            center = torch.tensor(config.neuralangelo_center)
            scale = config.neuralangelo_scale

        all_c2w[:, :3, 3] = (all_c2w[:, :3, 3] - center) * scale

        # === 7. 保存到GPU（如果配置） ===
        if config.load_data_on_gpu:
            self.all_images = torch.stack(all_images).cuda()  # [N, H, W, 3]
            self.all_c2w = all_c2w.cuda()
            self.all_fg_masks = torch.stack(all_fg_masks).cuda()
            self.directions = directions.cuda()
        else:
            self.all_images = torch.stack(all_images)
            self.all_c2w = all_c2w
            self.all_fg_masks = torch.stack(all_fg_masks)
            self.directions = directions

        # === 8. 划分train/val/test ===
        all_indices = torch.arange(len(self.all_images))
        if split == 'train':
            self.all_indices = all_indices[::config.n_test_traj_steps]
        elif split == 'val':
            self.all_indices = all_indices[1::config.n_test_traj_steps]
        elif split == 'test':
            self.all_indices = all_indices
```

**get_ray_directions函数**：

```python
# instant_nsr/datasets/utils.py
def get_ray_directions(H, W, fx, fy, cx, cy):
    """
    生成相机空间的光线方向

    Args:
        H, W: 图像高度和宽度
        fx, fy: 焦距
        cx, cy: 主点

    Returns:
        directions: [H, W, 3]，归一化的方向向量
    """
    # 1. 生成像素坐标网格
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing='xy'
    )
    # i: [H, W]，列坐标
    # j: [H, W]，行坐标

    # 2. 像素坐标 → 归一化设备坐标
    directions = torch.stack([
        (i - cx) / fx,   # x方向
        (j - cy) / fy,   # y方向
        torch.ones_like(i)  # z方向（朝前）
    ], dim=-1)  # [H, W, 3]

    # 3. 归一化（可选）
    # directions = F.normalize(directions, p=2, dim=-1)

    return directions
```

### 3.3 两种Dataset实现

#### **ColmapDataset（普通Dataset）**

用于验证和测试，返回整张图像。

```python
# instant_nsr/datasets/colmap.py:172-194
class ColmapDataset(Dataset, ColmapDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_indices)

    def __getitem__(self, index):
        return {'index': self.all_indices[index]}
        # 实际数据在preprocess_data中提取
```

#### **ColmapIterableDataset（可迭代Dataset）**

用于训练，无限循环生成batch。

```python
# instant_nsr/datasets/colmap.py:197-258
class ColmapIterableDataset(IterableDataset, ColmapDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}  # 空dict，数据在preprocess_data中采样
```

**为什么训练用IterableDataset？**

1. **无限训练**：不需要定义epoch的概念，Lightning根据`max_steps`控制
2. **动态采样**：每次迭代随机采样不同的rays，无需预先生成所有组合
3. **内存效率**：不需要存储所有可能的(image, pixel)组合

---

## 四、模型架构详解

### 4.1 NeuSModel整体架构

**位置**：`instant_nsr/models/neus.py:44-461`

```
NeuSModel
├─ geometry: VolumeSDF_gaussian (前景SDF)
│  ├─ encoding: ProgressiveBandHashGrid
│  └─ network: VanillaMLP
│
├─ texture: VolumeRadiance (颜色场)
│  ├─ dir_encoding: SphericalHarmonics
│  └─ network: VanillaMLP
│
├─ variance: VarianceNetwork (学习密度转换参数)
│
├─ geometry_bg: VolumeDensity (背景密度场，可选)
│  └─ encoding_with_network: HashGrid + MLP
│
└─ texture_bg: VolumeRadiance (背景颜色场，可选)
   └─ network: VanillaMLP
```

### 4.2 Geometry：VolumeSDF_gaussian

**位置**：`instant_nsr/models/geometry.py:129-289`

#### **网络结构**

```python
def setup(self):
    # === 1. Progressive Band Hash Grid ===
    self.encoding = get_encoding(3, self.config.xyz_encoding_config)
    # 输入：(x,y,z) ∈ [0,1]³
    # 输出：[n_levels * n_features_per_level] = 16*4 = 64维特征

    # === 2. MLP网络 ===
    self.network = get_mlp(
        encoding.n_output_dims,     # 输入：64（hash features）+ 3（xyz）= 67
        self.n_output_dims,         # 输出：64维特征
        self.config.mlp_network_config
    )
    # VanillaMLP: [67, 128, 64]
    # 激活函数：Softplus
    # 使用weight normalization
```

**配置**：

```yaml
geometry:
  feature_dim: 64  # 输出特征维度

  xyz_encoding_config:
    otype: ProgressiveBandHashGrid
    n_levels: 16
    n_features_per_level: 4
    log2_hashmap_size: 21
    base_resolution: 32
    per_level_scale: 1.3195
    include_xyz: true          # 拼接原始xyz
    start_level: 8
    start_step: 5000
    update_steps: 2000

  mlp_network_config:
    otype: VanillaMLP
    activation: Softplus
    n_neurons: 128
    n_hidden_layers: 1         # 只有1层隐藏层
    sphere_init: true          # 初始化为球形SDF
    sphere_init_radius: 0.8
    weight_norm: true
```

#### **前向传播**

```python
def forward(self, points, with_grad=True, with_feature=True, with_laplace=False):
    """
    Args:
        points: [N, 3]，世界坐标的3D点
        with_grad: 是否计算梯度（法线）
        with_feature: 是否返回特征
        with_laplace: 是否计算拉普拉斯（曲率）

    Returns:
        sdf: [N]，SDF值
        grad: [N, 3]，梯度（法线）
        feature: [N, 64]，特征向量
        laplace: [N]，拉普拉斯（曲率）
    """
    # === 1. 坐标归一化 ===
    points_ = points  # 保存原始坐标（用于autograd）
    points = contract_to_unisphere(points, self.radius, self.contraction_type)
    # 从 [-radius, radius]³ 映射到 [0, 1]³

    # === 2. Hash编码 + MLP ===
    with torch.set_grad_enabled(self.training or (with_grad and self.grad_type == 'analytic')):
        if with_grad and self.grad_type == 'analytic':
            points.requires_grad_(True)

        # Hash Grid编码
        encoded = self.encoding(points.view(-1, 3))  # [N, 64+3=67]

        # MLP前向
        out = self.network(encoded).view(*points.shape[:-1], self.n_output_dims)
        # out: [N, 64]

        sdf, feature = out[..., 0], out  # 第一维是SDF，全部是特征

        # === 3. 计算梯度 ===
        if with_grad:
            if self.grad_type == 'analytic':
                # 自动微分
                grad = torch.autograd.grad(
                    sdf, points_,
                    grad_outputs=torch.ones_like(sdf),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                # grad: [N, 3]

            elif self.grad_type == 'finite_difference':
                # 有限差分
                eps = self._finite_difference_eps  # 动态调整

                # 6个方向的偏移
                offsets = torch.tensor([
                    [eps, 0, 0], [-eps, 0, 0],
                    [0, eps, 0], [0, -eps, 0],
                    [0, 0, eps], [0, 0, -eps]
                ], device=points.device)

                points_d = (points_[..., None, :] + offsets).clamp(-self.radius, self.radius)
                # [N, 6, 3]

                # 归一化到[0,1]
                points_d = contract_to_unisphere(points_d, self.radius, ...)

                # 查询6个邻域点的SDF
                sdf_d = self.network(self.encoding(points_d.view(-1, 3)))[..., 0]
                sdf_d = sdf_d.view(*points.shape[:-1], 6)
                # [N, 6]

                # 中心差分
                grad = 0.5 * (sdf_d[..., 0::2] - sdf_d[..., 1::2]) / eps
                # [N, 3] = [(sdf(x+ε)-sdf(x-ε))/(2ε), ...]

                # === 4. 计算拉普拉斯（可选） ===
                if with_laplace:
                    laplace = (sdf_d[..., 0::2] + sdf_d[..., 1::2] - 2*sdf[..., None]).sum(-1) / (eps**2)
                    # ∇²f = (f(x+ε)+f(x-ε)-2f(x))/ε²

    # === 5. 返回结果 ===
    rv = [sdf]
    if with_grad:
        rv.append(grad)
    if with_feature:
        rv.append(feature)
    if with_laplace:
        rv.append(laplace)

    return rv[0] if len(rv) == 1 else rv
```

**contract_to_unisphere函数**：

```python
# instant_nsr/models/geometry.py:17-28
def contract_to_unisphere(x, radius, type=None):
    """
    将世界坐标映射到[0,1]³

    Args:
        x: [N, 3]，世界坐标
        radius: 场景半径
        type: 收缩类型（AABB或UN_BOUNDED_SPHERE）

    Returns:
        x_contracted: [N, 3]，归一化坐标
    """
    if type == ContractionType.AABB:
        # 简单线性映射
        return (x + radius) / (2 * radius)
        # [-radius, radius] → [0, 1]

    elif type == ContractionType.UN_BOUNDED_SPHERE:
        # mip-NeRF 360的收缩函数
        # 适用于无界场景
        mag = x.norm(p=2, dim=-1, keepdim=True)
        return torch.where(
            mag < 1,
            x,  # 内部：不变
            (2 - 1/mag) * (x / mag)  # 外部：收缩到[1,2]
        )
```

### 4.3 Texture：VolumeRadiance

**位置**：`instant_nsr/models/texture.py`

```python
@register('volume-radiance')
class VolumeRadiance(BaseModel):
    def setup(self):
        # === 1. 方向编码（球谐函数） ===
        self.dir_encoding = get_encoding(3, self.config.dir_encoding_config)
        # SphericalHarmonics degree=4 → 15维

        # === 2. MLP网络 ===
        self.network = get_mlp(
            self.config.input_feature_dim +  # 64（SDF特征）+ 3（法线）= 67
            self.dir_encoding.n_output_dims, # 15（方向编码）
            3,  # 输出RGB
            self.config.mlp_network_config
        )
        # VanillaMLP: [67+15=82, 128, 128, 3]

        # === 3. 颜色激活函数 ===
        self.color_activation = get_activation(self.config.color_activation)
        # sigmoid：将输出映射到[0,1]

    def forward(self, features, dirs, normals=None):
        """
        Args:
            features: [N, 64]，SDF网络的输出特征
            dirs: [N, 3]，光线方向（归一化）
            normals: [N, 3]，表面法线（归一化）

        Returns:
            rgb: [N, 3]，预测颜色
        """
        # === 1. 拼接输入 ===
        if normals is not None:
            network_inp = torch.cat([features, normals], dim=-1)
            # [N, 64+3=67]
        else:
            network_inp = features

        # === 2. 编码方向 ===
        dirs_encoded = self.dir_encoding(dirs)  # [N, 15]

        # === 3. 拼接方向特征 ===
        network_inp = torch.cat([network_inp, dirs_encoded], dim=-1)
        # [N, 67+15=82]

        # === 4. MLP前向 ===
        rgb = self.network(network_inp)  # [N, 3]

        # === 5. 激活函数 ===
        rgb = self.color_activation(rgb)  # sigmoid → [0, 1]

        return rgb
```

**配置**：

```yaml
texture:
  name: volume-radiance
  input_feature_dim: 67  # ${add:64,3} = SDF特征(64) + 法线(3)

  dir_encoding_config:
    otype: SphericalHarmonics
    degree: 4  # 4阶球谐 → (degree+1)² = 25维（实际使用15维）

  mlp_network_config:
    otype: VanillaMLP
    activation: ReLU
    n_neurons: 128
    n_hidden_layers: 2
    weight_norm: true

  color_activation: sigmoid
```

### 4.4 Variance：密度转换网络

**位置**：`instant_nsr/models/neus.py:15-42`

NeuS使用可学习的variance来控制SDF→density的转换锐度。

```python
class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 初始化variance值
        init_val = self.config.init_val  # 0.3
        self.variance = nn.Parameter(torch.tensor(init_val))

    @property
    def inv_s(self):
        """
        inverse std，用于SDF→density转换
        """
        val = torch.exp(self.variance * 10.0)
        # 初始：exp(0.3 * 10) = exp(3) ≈ 20
        return val

    def forward(self, x):
        # 可以接受输入（兼容性），但实际不使用
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s
```

**作用**：

在NeuS的volume rendering中，密度由SDF通过sigmoid函数计算：

```
α(t) = sigmoid(-inv_s * sdf(t))
```

- `inv_s`大：sigmoid陡峭，密度在表面附近急剧变化（类似hard surface）
- `inv_s`小：sigmoid平缓，密度变化平滑（类似soft surface）

训练初期`inv_s`较小（~20），允许模糊的表面；随着训练，`inv_s`增大（~100+），表面越来越锐利。

---

## 五、Volume Rendering实现

### 5.1 采样策略

**位置**：`instant_nsr/models/neus.py:309-342`

NeuS支持两种采样方式：

#### **方式1：GS深度引导采样（本项目使用）**

```python
def forward_(self, rays, gs_depth=None, use_depth_guide=False):
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]

    if self.config.gs_sampling:
        # === NeuS层次采样 + GS深度引导 ===
        ray_indices, midpoints, positions, dists, intersected_ray_indices = \
            self.ray_upsampe_hier(rays_o, rays_d, gs_depth, use_depth_guide)
```

**ray_upsampe_hier详解**：

```python
# instant_nsr/models/neus.py:141-242
def ray_upsampe_hier(self, rays_o, rays_d, gs_depth=None, use_depth_guide=False):
    """
    NeuS的层次采样 + GS深度引导
    """
    n_rays = rays_o.shape[0]
    n_equispaced = self.config.num_samples_equispaced  # 64
    n_equispaced_fine = 0  # 初始化
    stratified = self.randomized

    # === 1. 光线与场景AABB相交测试 ===
    t_min, t_max = ray_aabb_intersect(rays_o, rays_d, self.scene_aabb)
    # t_min, t_max: [N_rays]，光线进入和离开AABB的参数

    intersected_ray_indices = torch.where(t_max > t_min)[0]
    # 只处理相交的光线

    # === 2. GS深度引导（如果启用） ===
    if use_depth_guide and gs_depth is not None:
        n_equispaced_fine = self.config.num_samples_per_ray - n_equispaced
        # 额外采样点数：1024 - 64 = 960

        # 2.1 在GS深度处查询SDF
        depth_points = rays_o[intersected_ray_indices] + \
                      rays_d[intersected_ray_indices] * gs_depth[intersected_ray_indices, None]
        # [N_intersected, 3]

        sdf_depth = self.geometry(depth_points, with_grad=False, with_feature=False)
        # [N_intersected]，深度处的SDF值

        # 2.2 计算采样范围
        k = 10.0  # 自适应系数
        delta = k * torch.abs(sdf_depth)

        t_min_fine = torch.clamp(gs_depth[intersected_ray_indices] - delta, min=0.0)
        t_max_fine = gs_depth[intersected_ray_indices] + delta
        # 采样范围：[D - k|s|, D + k|s|]

    # === 3. 过滤相交光线 ===
    t_min = t_min[intersected_ray_indices][:, None]
    t_max = t_max[intersected_ray_indices][:, None]
    rays_o_ = rays_o[intersected_ray_indices]
    rays_d_ = rays_d[intersected_ray_indices]

    # === 4. 粗略均匀采样 ===
    if stratified:
        rands = torch.rand(n_equispaced, device=rays_o.device)
    else:
        rands = torch.ones(n_equispaced, device=rays_o.device) * 0.5

    rands += torch.arange(n_equispaced, dtype=torch.float32, device=rays_o.device)
    dists = rands[None, :] / n_equispaced * (t_max - t_min) + t_min
    # [N_intersected, 64]

    # === 5. 精细引导采样 ===
    if use_depth_guide:
        t_min_fine = t_min_fine[intersected_ray_indices][:, None]
        t_max_fine = t_max_fine[intersected_ray_indices][:, None]

        if stratified:
            rands_fine = torch.rand(n_equispaced_fine, device=rays_o.device)
        else:
            rands_fine = torch.ones(n_equispaced_fine, device=rays_o.device) * 0.5

        rands_fine += torch.arange(n_equispaced_fine, dtype=torch.float32, device=rays_o.device)
        dists_fine = rands_fine[None, :] / n_equispaced_fine * (t_max_fine - t_min_fine) + t_min_fine
        # [N_intersected, 960]

        # 合并并排序
        dists = torch.cat([dists, dists_fine], dim=-1)  # [N_intersected, 1024]
        dists, _ = torch.sort(dists, dim=-1)

    # === 6. 格式化输出 ===
    ray_indices = torch.arange(n_rays, device=rays_o.device)[intersected_ray_indices][:, None]
    ray_indices = ray_indices.expand(-1, n_equispaced + n_equispaced_fine).reshape(-1)
    # [N_intersected * 1024]，每个采样点对应的光线index

    midpoints = dists.reshape(-1, 1)  # [N_samples, 1]

    positions = rays_o_[:, None, :] + rays_d_[:, None, :] * dists[..., None]
    positions = positions.reshape(-1, 3)  # [N_samples, 3]

    # 计算采样间隔
    interval_dists = dists[..., 1:] - dists[..., :-1]
    last_i_dists = 1.732 * 2 * self.config.radius / (n_equispaced + n_equispaced_fine)
    interval_dists = torch.cat([interval_dists,
                                torch.full_like(interval_dists[..., :1], last_i_dists)],
                               dim=-1).reshape(-1, 1)
    # [N_samples, 1]

    return ray_indices, midpoints, positions, interval_dists, intersected_ray_indices
```

**采样可视化**：

```
无引导采样（warmup阶段）:
ray ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
    64个均匀采样点，分布在整个[t_min, t_max]

GS深度引导采样（mutual guidance阶段）:
ray ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ├─┼─┼─┼─┤      ║║║║║║║║║║      ├─┼─┼─┼─┤
    粗略64点    精细960点集中在深度附近    粗略
                   ↑
                 D_gs ± k|s|
```

#### **方式2：Occupancy Grid采样（备选）**

```python
else:  # not self.config.gs_sampling
    # === nerfacc的occupancy grid采样 ===
    ray_indices, t_starts, t_ends = ray_marching(
        rays_o, rays_d,
        scene_aabb=self.scene_aabb,
        grid=self.occupancy_grid if self.config.grid_prune else None,
        render_step_size=self.render_step_size,
        stratified=self.randomized,
        cone_angle=0.0,
        alpha_thre=0.0
    )

    midpoints = (t_starts + t_ends) / 2.
    positions = rays_o[ray_indices] + rays_d[ray_indices] * midpoints
    dists = t_ends - t_starts
```

### 5.2 Volume Rendering核心

**位置**：`instant_nsr/models/neus.py:344-435`

```python
# 前面已经采样得到：
# - positions: [N_samples, 3]，采样点位置
# - ray_indices: [N_samples]，每个点对应的光线index
# - dists: [N_samples, 1]，采样间隔

# === 1. 查询SDF和特征 ===
if self.config.geometry.grad_type == 'finite_difference':
    sdf, sdf_grad, feature, sdf_laplace = self.geometry(
        positions, with_grad=True, with_feature=True, with_laplace=True
    )
else:
    sdf, sdf_grad, feature = self.geometry(
        positions, with_grad=True, with_feature=True
    )
# sdf: [N_samples]
# sdf_grad: [N_samples, 3]
# feature: [N_samples, 64]

# === 2. 计算法线 ===
normal = F.normalize(sdf_grad, p=2, dim=-1)
# normal: [N_samples, 3]

# === 3. SDF → 密度（NeuS核心） ===
alpha = self.get_alpha(sdf, normal, t_dirs, dists)[..., None]
# alpha: [N_samples, 1]

# get_alpha实现：
def get_alpha(self, sdf, normal, dirs, dists):
    """
    NeuS的SDF→密度转换
    """
    inv_s = self.variance.inv_s.clip(1e-6, 1e6)

    # 估计当前和下一个采样点的SDF
    true_cos = (dirs * normal).sum(-1, keepdim=True)
    # cos(θ)，光线方向与法线夹角

    # 沿光线方向迭代一步的SDF估计
    iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                 F.relu(-true_cos) * cos_anneal_ratio)

    estimated_next_sdf = sdf + iter_cos * dists.reshape(-1) * 0.5
    estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1) * 0.5

    # NeuS的unbiased密度估计
    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
    return alpha

# === 4. 查询颜色 ===
rgb = self.texture(feature, t_dirs, normal)
# rgb: [N_samples, 3]

# === 5. 体渲染累积 ===
# 5.1 计算权重
weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
# weights: [N_samples, 1]
# w_i = α_i * Π(1 - α_j) for j < i

# render_weight_from_alpha实现（nerfacc）：
def render_weight_from_alpha(alpha, ray_indices, n_rays):
    """
    从alpha值计算体渲染权重
    """
    # 排序（按光线index）
    sorted_indices = torch.argsort(ray_indices)
    alpha_sorted = alpha[sorted_indices]
    ray_indices_sorted = ray_indices[sorted_indices]

    # 计算累积透明度 T_i = Π(1 - α_j) for j < i
    trans = torch.cumprod(1 - alpha_sorted + 1e-10, dim=0)
    trans = torch.cat([torch.ones_like(trans[:1]), trans[:-1]], dim=0)

    # 权重 = α * T
    weights = alpha_sorted * trans

    # 恢复原始顺序
    weights_unsorted = torch.zeros_like(weights)
    weights_unsorted[sorted_indices] = weights

    return weights_unsorted

# 5.2 累积不透明度
opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
# opacity: [N_rays, 1]，每条光线的总不透明度

# 5.3 累积深度
depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
# depth: [N_rays, 1]，加权深度

# 5.4 累积颜色
comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
# comp_rgb: [N_rays, 3]，最终颜色

# 5.5 累积法线
comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
comp_normal = F.normalize(comp_normal, p=2, dim=-1)
# comp_normal: [N_rays, 3]

# accumulate_along_rays实现（nerfacc）：
def accumulate_along_rays(weights, ray_indices, values, n_rays):
    """
    沿光线累积属性
    """
    if values is None:
        # 累积权重（不透明度）
        outputs = torch.zeros((n_rays, 1), device=weights.device)
        outputs.scatter_add_(0, ray_indices.unsqueeze(-1), weights)
    else:
        # 累积加权属性
        outputs = torch.zeros((n_rays, values.shape[-1]), device=weights.device)
        outputs.scatter_add_(0, ray_indices.unsqueeze(-1).expand(-1, values.shape[-1]),
                            weights * values)
    return outputs

# === 6. 计算曲率（训练时） ===
if self.training:
    curvature = self.geometry.get_sdf_and_curvature_1d_precomputed_gradient_normal_based(
        positions, normal
    )
    # curvature: [N_samples]

# === 7. 组装输出 ===
out = {
    'comp_rgb': comp_rgb,          # [N_rays, 3]
    'comp_normal': comp_normal,    # [N_rays, 3]
    'opacity': opacity,            # [N_rays, 1]
    'depth': depth,                # [N_rays, 1]
    'rays_valid': opacity > 0,     # [N_rays]
    'num_samples': torch.as_tensor([len(midpoints)])
}

if self.training:
    out.update({
        'sdf_samples': sdf,
        'sdf_grad_samples': sdf_grad,
        'weights': weights.view(-1),
        'smoothing': curvature,  # 用于曲率损失
        'positions': positions,
        'zero_samples': False,
        'intersected_ray_indices': intersected_ray_indices
    })
```

### 5.3 背景渲染

**位置**：`instant_nsr/models/neus.py:245-307`

```python
def forward_bg_(self, rays):
    """
    渲染背景（无界区域）
    """
    n_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]

    # === 1. 定义密度函数 ===
    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = rays_o[ray_indices.long()]
        t_dirs = rays_d[ray_indices.long()]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
        density, _ = self.geometry_bg(positions)
        return density[..., None]

    # === 2. 确定采样范围 ===
    _, t_max = ray_aabb_intersect(rays_o, rays_d, self.scene_aabb)
    # 如果光线与前景AABB相交，从far intersection开始
    # 否则从near_plane_bg开始
    near_plane = torch.where(t_max > 1e9, self.near_plane_bg, t_max)

    # === 3. Ray Marching采样 ===
    with torch.no_grad():
        ray_indices, t_starts, t_ends = ray_marching(
            rays_o, rays_d,
            scene_aabb=None,  # 无界
            grid=self.occupancy_grid_bg if self.config.grid_prune else None,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=self.far_plane_bg,  # 1e3
            render_step_size=self.render_step_size_bg,
            stratified=self.randomized,
            cone_angle=self.cone_angle_bg,
            alpha_thre=0.0
        )

    # === 4. 查询密度和特征 ===
    ray_indices = ray_indices.long()
    t_origins = rays_o[ray_indices]
    t_dirs = rays_d[ray_indices]
    midpoints = (t_starts + t_ends) / 2.
    positions = t_origins + t_dirs * midpoints
    intervals = t_ends - t_starts

    density, feature = self.geometry_bg(positions)
    rgb = self.texture_bg(feature, t_dirs)

    # === 5. 体渲染 ===
    weights = render_weight_from_density(
        t_starts, t_ends, density[..., None],
        ray_indices=ray_indices, n_rays=n_rays
    )
    opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
    depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
    comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)

    # 添加背景颜色
    comp_rgb = comp_rgb + self.background_color * (1.0 - opacity)

    return {
        'comp_rgb': comp_rgb,
        'opacity': opacity,
        'depth': depth,
        'rays_valid': opacity > 0,
        'num_samples': torch.as_tensor([len(t_starts)])
    }
```

### 5.4 前景+背景合成

```python
# 前景渲染
out = self.forward_(rays, gs_depth, use_depth_guide)

# 背景渲染
if self.config.learned_background:
    out_bg = self.forward_bg_(rays)
else:
    out_bg = {
        'comp_rgb': self.background_color[None, :].expand(*comp_rgb.shape),
        'num_samples': torch.zeros_like(out['num_samples']),
        'rays_valid': torch.zeros_like(out['rays_valid'])
    }

# 合成
out_full = {
    'comp_rgb': out['comp_rgb'] + out_bg['comp_rgb'] * (1.0 - out['opacity']),
    'num_samples': out['num_samples'] + out_bg['num_samples'],
    'rays_valid': out['rays_valid'] | out_bg['rays_valid']
}

return {
    **out,                                    # 前景结果
    **{k + '_bg': v for k, v in out_bg.items()},  # 背景结果（带_bg后缀）
    **{k + '_full': v for k, v in out_full.items()}  # 合成结果（带_full后缀）
}
```

---

## 六、训练循环与优化

### 6.1 Lightning训练循环

PyTorch Lightning自动执行以下循环：

```python
# 伪代码表示
for epoch in range(max_epochs):
    for batch_idx, batch in enumerate(train_dataloader):
        # 1. Lightning自动调用
        system.on_train_batch_start(batch, batch_idx)

        # 2. 预处理数据（自动调用）
        batch = system.preprocess_data(batch, 'train')

        # 3. 前向+损失计算（用户定义）
        result = system.training_step(batch, batch_idx)
        loss = result['loss']

        # 4. 反向传播（Lightning自动）
        loss.backward()

        # 5. 优化器步进（Lightning自动）
        optimizer.step()
        optimizer.zero_grad()

        # 6. 学习率调度（Lightning自动）
        if scheduler:
            scheduler.step()

        # 7. 记录日志（Lightning自动）
        # system.log()的内容自动发送到loggers

        # 8. 验证（按val_check_interval）
        if global_step % val_check_interval == 0:
            for val_batch in val_dataloader:
                result = system.validation_step(val_batch, val_batch_idx)
            system.validation_epoch_end(validation_outputs)
```

### 6.2 双优化器协调

本项目的特殊之处在于有**两个独立的优化器**：

```python
# training_step中
def training_step(self, batch, batch_idx):
    # ... 前向传播 ...

    # === SDF分支优化 ===
    loss_sdf = L_rgb + L_depth + L_normal + L_eikonal + L_curvature
    # Lightning会自动调用：
    # self.optimizer.zero_grad()  # SDF优化器
    # loss_sdf.backward()
    # self.optimizer.step()

    # === GS分支优化（手动控制） ===
    if self.config.model.if_gaussian:
        loss_gs = L_rgb + L_SSIM + ...
        loss_gs.backward()  # 手动反向传播

        with torch.no_grad():
            # 密度控制
            if iteration % 100 == 0:
                self.gaussians.adjust_anchor(...)

            # 手动优化器步进
            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)

    # 只返回SDF损失（给Lightning）
    return {'loss': loss_sdf}
```

**时间线**：

```
Single Iteration:
├─ Lightning调用training_step
│  ├─ GS前向 → depth_gs, normal_gs
│  ├─ SDF前向(use depth_gs) → depth_sdf, normal_sdf
│  │
│  ├─ 计算loss_sdf
│  ├─ 计算loss_gs
│  │
│  ├─ loss_gs.backward() [手动]
│  ├─ GS密度控制 [手动]
│  ├─ GS optimizer.step() [手动]
│  │
│  └─ return {'loss': loss_sdf}
│
├─ Lightning执行
│  ├─ loss_sdf.backward() [自动]
│  └─ SDF optimizer.step() [自动]
│
└─ 记录日志、更新进度条等 [自动]
```

### 6.3 学习率调度

**SDF分支**（Lightning管理）：

```yaml
system:
  scheduler:
    name: MultiStepLR
    interval: step
    args:
      milestones: [10000, 15000, 18000]
      gamma: 0.33
```

调度过程：
- 0-10000步：lr = 0.01
- 10000-15000步：lr = 0.01 × 0.33 = 0.0033
- 15000-18000步：lr = 0.0033 × 0.33 = 0.0011
- 18000步后：lr = 0.0011 × 0.33 = 0.00036

**GS分支**（手动管理）：

```python
# gaussian_splatting/scene/gaussian_model.py:375-398
def update_learning_rate(self, iteration):
    """
    指数衰减学习率
    """
    for param_group in self.optimizer.param_groups:
        if param_group["name"] == "anchor":
            lr = get_expon_lr_func(
                lr_init=self.position_lr_init,
                lr_final=self.position_lr_final,
                lr_delay_mult=self.position_lr_delay_mult,
                max_steps=self.position_lr_max_steps
            )(iteration)
            param_group['lr'] = lr

        # ... 其他参数组类似
```

**get_expon_lr_func**：

```python
def get_expon_lr_func(lr_init, lr_final, lr_delay_mult, max_steps):
    def func(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0

        if lr_delay_mult < 1:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / max_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)

        return delay_rate * log_lerp

    return func
```

曲线形状：先warmup，再指数衰减。

---

## 七、损失函数与正则化

### 7.1 SDF分支损失

**位置**：`instant_nsr/systems/neus.py:413-496`

```python
loss_sdf = 0.0

# === 1. RGB损失 ===
loss_rgb_l1 = F.l1_loss(
    out['comp_rgb_full'][out['rays_valid_full']],
    batch['rgb'][out['rays_valid_full']]
)
loss_sdf += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)
# self.C(): 处理动态权重调度

# === 2. 深度损失（mutual supervision from GS） ===
fixed_picked_gs_depth = picked_gs_depth[out['rays_valid']].detach()
diff_neus = torch.abs(out['depth'][out['rays_valid']] - fixed_picked_gs_depth)

# 过滤异常值
depth_ratio = 10.0 if self.current_epoch_set > start_step else 2.0
diff_neus[diff_neus > self.config.model.radius / depth_ratio] = 0

loss_depth_L1 = diff_neus.sum() / (diff_neus > 0).sum()
loss_sdf += loss_depth_L1 * self.C(self.config.system.loss.depth_w) / self.config.model.radius

# === 3. 法线损失（mutual supervision from GS） ===
if self.current_epoch_set > start_step:
    fixed_picked_gs_normal = picked_gs_normal[out['rays_valid']].detach()
    normal_diff = self.cos_similarity_loss(
        fixed_picked_gs_normal,
        out['comp_normal'][out['rays_valid']]
    )
    loss_sdf += normal_diff * self.config.system.loss.normal_w

# cos_similarity_loss实现：
def cos_similarity_loss(self, pred, target):
    """
    余弦相似度损失（鼓励法线对齐）
    """
    return (1 - F.cosine_similarity(pred, target, dim=-1)).mean()

# === 4. Eikonal损失 ===
loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.) ** 2).mean()
loss_sdf += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)

# === 5. 曲率损失 ===
if self.C(self.config.system.loss.lambda_smoothing) > 0:
    loss_smoothing = out['smoothing'].abs().mean()
    loss_sdf += loss_smoothing * self.C(self.config.system.loss.lambda_smoothing)
```

**损失权重调度**（self.C()）：

```python
# instant_nsr/systems/base.py:28-36
def C(self, value):
    """
    处理动态权重调度
    """
    if isinstance(value, int) or isinstance(value, float):
        return value
    elif isinstance(value, list):
        # 线性插值
        # value格式：[[step1, val1], [step2, val2], ...]
        return self.interpolate_schedule(value, self.global_step)
```

**interpolate_schedule实现**：

```python
def interpolate_schedule(schedule, step):
    """
    Args:
        schedule: [[0, 0.0], [5000, 1.0], [15000, 0.1]]
        step: 当前步数
    """
    for i in range(len(schedule) - 1):
        if schedule[i][0] <= step < schedule[i+1][0]:
            t = (step - schedule[i][0]) / (schedule[i+1][0] - schedule[i][0])
            return schedule[i][1] * (1 - t) + schedule[i+1][1] * t

    return schedule[-1][1]
```

**典型权重调度**：

```yaml
loss:
  lambda_rgb_l1: 1.0  # 固定

  depth_w:  # 深度损失
    - [0, 1.0]
    - [5000, 1.0]
    - [15000, 0.1]  # 15000步后衰减

  normal_w:  # 法线损失
    - [0, 0.0]      # 初始不使用
    - [5000, 1.0]   # 5000步启用
    - [15000, 0.1]  # 15000步衰减

  lambda_eikonal: 0.1  # 固定

  lambda_smoothing:  # 曲率损失
    - [0, 0.0]
    - [5000, 0.001]
    - [10000, 0.01]  # 渐进增强
```

### 7.2 GS分支损失

**位置**：`instant_nsr/systems/neus.py:499-532`

```python
loss_gs = 0.0

# === 1. RGB L1损失 ===
gt_image = viewpoint_cam.original_image.cuda()
Ll1 = l1_loss(image, gt_image)

# === 2. SSIM损失 ===
ssim_loss = 1.0 - ssim(image, gt_image)

# === 3. Scaling正则化 ===
scaling_reg = scaling.prod(dim=1).mean()

# === 4. 深度损失（mutual supervision from SDF） ===
fixed_neus_picked_depth = out['depth'][out['rays_valid']].detach()
diff = torch.abs(fixed_neus_picked_depth - picked_gs_depth[out['rays_valid']])

depth_ratio = 10.0
diff[diff > self.config.model.radius / depth_ratio] = 0

loss_depth_L1_gs = diff.sum() / (diff > 0).sum()
depth_loss_gs = loss_depth_L1_gs * self.C(self.config.system.loss.depth_w) / self.config.model.radius

# === 5. 法线损失（mutual supervision from SDF） ===
if self.current_epoch_set >= start_step:
    fixed_neus_picked_normal = out['comp_normal'][out['rays_valid']].detach()
    normal_loss_gs = self.cos_similarity_loss(
        picked_gs_normal[out['rays_valid']],
        fixed_neus_picked_normal
    ) * self.config.system.loss.normal_w
else:
    normal_loss_gs = 0.0

# === 6. 总损失 ===
loss_gs = (
    (1.0 - self.op.lambda_dssim) * Ll1 +
    self.op.lambda_dssim * ssim_loss +
    0.01 * scaling_reg +
    depth_loss_gs +
    normal_loss_gs
)
```

### 7.3 正则化总结

| 损失项 | 作用 | 权重 | 调度 |
|-------|------|-----|------|
| **SDF分支** |
| L_rgb | RGB重建 | 1.0 | 固定 |
| L_depth | 深度对齐（from GS） | 1.0 → 0.1 | 15k步衰减 |
| L_normal | 法线对齐（from GS） | 0 → 1.0 → 0.1 | 5k启用，15k衰减 |
| L_eikonal | 约束∇sdf=1 | 0.1 | 固定 |
| L_curvature | 平滑表面 | 0 → 0.01 | 渐进增强 |
| **GS分支** |
| L_rgb | RGB重建 | 1 - λ_dssim | 固定 |
| L_SSIM | 结构相似度 | λ_dssim | 固定 |
| L_scaling | 限制Gaussian尺度 | 0.01 | 固定 |
| L_depth | 深度对齐（from SDF） | 同SDF | 同步 |
| L_normal | 法线对齐（from SDF） | 同SDF | 同步 |

---

## 八、完整训练流程图

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        GSDF Training Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────┐
│                    1. Initialization                        │
└─────────────────────────────────────────────────────────────┘
launch.py
  ├─ load_config(args.config)
  ├─ dm = ColmapDataModule(config.dataset)
  │  └─ 加载COLMAP数据（images, poses, intrinsics）
  ├─ system = NeuSSystem(config)
  │  ├─ BaseSystem.__init__
  │  │  ├─ self.model = NeuSModel(config.model)
  │  │  │  ├─ geometry = VolumeSDF_gaussian
  │  │  │  ├─ texture = VolumeRadiance
  │  │  │  └─ variance = VarianceNetwork
  │  │  └─ self.criterions = ...
  │  └─ GaussianModel初始化（如果启用）
  │     ├─ self.gaussians = GaussianModel(...)
  │     ├─ self.scene = Scene(...)
  │     └─ self.pretrain_gs() [可选]
  └─ trainer = Trainer(callbacks, loggers, ...)

┌─────────────────────────────────────────────────────────────┐
│                    2. Training Loop                         │
└─────────────────────────────────────────────────────────────┘
trainer.fit(system, dm)
  │
  └─ for step in range(max_steps):
       │
       ├─ [on_train_batch_start]
       │  ├─ model.update_step(epoch, global_step)
       │  │  └─ geometry.update_step()  # 激活新Hash Grid层级
       │  └─ preprocess_data(batch, 'train')
       │     ├─ 随机采样rays（from random pixels）
       │     └─ batch = {'rays', 'rgb', 'fg_mask', 'used_x', 'used_y'}
       │
       ├─ [training_step]
       │  │
       │  ├─── GS Branch (if enabled) ─────────────────────────┐
       │  │    │                                                │
       │  │    ├─ viewpoint_cam = scene.getTrainCameras()[idx] │
       │  │    ├─ render_pkg = gaussian_renderer.render(...)   │
       │  │    │  ├─ RGB_gs [3, H, W]                          │
       │  │    │  ├─ Depth_gs [1, H, W]                        │
       │  │    │  └─ Normal_gs [3, H, W]                       │
       │  │    │                                                │
       │  │    └─ 提取对应像素                                  │
       │  │       ├─ picked_gs_depth = Depth_gs[y, x]          │
       │  │       └─ picked_gs_normal = Normal_gs[y, x]        │
       │  │                                                     │
       │  ├─── SDF Branch ──────────────────────────────────────┤
       │  │    │                                                │
       │  │    ├─ if current_epoch > 5000:                     │
       │  │    │    out = self(batch, picked_gs_depth, True)   │
       │  │    │    # 使用深度引导采样                          │
       │  │    │ else:                                          │
       │  │    │    out = self(batch, picked_gs_depth, False)  │
       │  │    │    # 均匀采样                                  │
       │  │    │                                                │
       │  │    └─ NeuSModel.forward()                          │
       │  │       ├─ ray_upsampe_hier() → positions, indices   │
       │  │       ├─ geometry(positions) → sdf, grad, feature  │
       │  │       ├─ get_alpha(sdf, ...) → alpha               │
       │  │       ├─ texture(feature, dirs, normal) → rgb      │
       │  │       └─ volume_rendering → comp_rgb, depth, normal│
       │  │                                                     │
       │  ├─── Mutual Supervision ──────────────────────────────┤
       │  │    │                                                │
       │  │    ├─ SDF Loss:                                    │
       │  │    │  ├─ L_rgb                                     │
       │  │    │  ├─ L_depth = |depth_sdf - depth_gs.detach()| │
       │  │    │  ├─ L_normal = cos(normal_sdf, normal_gs.detach()) │
       │  │    │  ├─ L_eikonal                                 │
       │  │    │  └─ L_curvature                               │
       │  │    │                                                │
       │  │    └─ GS Loss:                                     │
       │  │       ├─ L_rgb + L_SSIM                            │
       │  │       ├─ L_depth = |depth_gs - depth_sdf.detach()| │
       │  │       ├─ L_normal = cos(normal_gs, normal_sdf.detach()) │
       │  │       └─ L_scaling                                 │
       │  │                                                     │
       │  ├─── GS Backward & Density Control ───────────────────┤
       │  │    │                                                │
       │  │    ├─ loss_gs.backward()  [手动]                   │
       │  │    │                                                │
       │  │    ├─ if iteration % 100 == 0:                     │
       │  │    │    # 查询SDF值                                │
       │  │    │    xyz_sdf = geometry(gs_positions)           │
       │  │    │    anchor_sdf = geometry(anchor_positions)    │
       │  │    │    # 几何感知密度控制                          │
       │  │    │    gaussians.adjust_anchor(                   │
       │  │    │       xyz_sdf=xyz_sdf,                        │
       │  │    │       anchor_sdf=anchor_sdf,                  │
       │  │    │       growing_weight=0.0002                   │
       │  │    │    )                                          │
       │  │    │                                                │
       │  │    └─ gaussians.optimizer.step()  [手动]           │
       │  │       gaussians.optimizer.zero_grad()              │
       │  │                                                     │
       │  └─ return {'loss': loss_sdf}                         │
       │                                                        │
       ├─ [Lightning Auto Operations]                          │
       │  ├─ loss_sdf.backward()  [自动]                       │
       │  ├─ optimizer.step()  [自动，SDF优化器]               │
       │  ├─ optimizer.zero_grad()  [自动]                     │
       │  └─ scheduler.step()  [自动]                          │
       │                                                        │
       ├─ [Logging]                                            │
       │  ├─ TensorBoard                                       │
       │  └─ CSV Logger                                        │
       │                                                        │
       └─ if step % val_check_interval == 0:                   │
          └─ validation_step()                                 │
             ├─ 渲染完整图像                                    │
             ├─ 计算PSNR                                       │
             └─ 保存可视化结果                                  │

┌─────────────────────────────────────────────────────────────┐
│                    3. Training Timeline                     │
└─────────────────────────────────────────────────────────────┘

Step 0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 30000

Phase 1: Warmup (0-5000)
├─ GS: 独立训练，标准密度控制
├─ SDF: 均匀采样，无mutual supervision
└─ Hash Grid: Level 0-8激活

Phase 2: Mutual Guidance (5000-15000)
├─ GS → SDF: 深度引导采样
├─ SDF → GS: 几何感知密度控制
├─ 双向监督: depth_w=1.0, normal_w=1.0
└─ Hash Grid: Level 8→15渐进激活

Phase 3: Refinement (15000-30000)
├─ GS: 停止密度控制，只优化参数
├─ SDF: 继续优化
├─ 权重衰减: depth_w=0.1, normal_w=0.1
└─ Hash Grid: 全部激活

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 总结

### PyTorch Lightning的核心优势

1. **自动化训练循环**：无需手写训练loop
2. **模块化设计**：System（模型+训练逻辑）、DataModule（数据加载）分离
3. **易于扩展**：通过Callbacks添加功能
4. **日志记录**：self.log()自动同步到所有loggers
5. **分布式训练**：只需改strategy参数

### NeuS分支的关键设计

1. **渐进式Hash Grid**：coarse-to-fine学习，避免高频噪声
2. **深度引导采样**：GS深度缩小采样范围，加速收敛
3. **Mutual Supervision**：双向深度法线监督，几何一致性
4. **几何感知密度控制**：SDF引导GS分布，去除floaters
5. **背景建模**：前景+背景分离渲染

### 与传统NeuS的主要区别

| 特性 | 传统NeuS | GSDF NeuS分支 |
|-----|---------|--------------|
| 采样策略 | 均匀采样 | GS深度引导采样 |
| 训练框架 | 原生PyTorch | PyTorch Lightning |
| 几何监督 | Mask + RGB | GS深度法线监督 |
| 收敛速度 | 慢（需50k+步） | 快（30k步） |
| 双分支协作 | 无 | 与GS紧密耦合 |

### 关键文件速查

| 功能 | 文件路径 | 关键类/函数 |
|-----|---------|-----------|
| **训练入口** | launch.py | main() |
| **系统模块** | instant_nsr/systems/neus.py | NeuSSystem |
| **基类** | instant_nsr/systems/base.py | BaseSystem |
| **模型** | instant_nsr/models/neus.py | NeuSModel |
| **SDF网络** | instant_nsr/models/geometry.py | VolumeSDF_gaussian |
| **颜色网络** | instant_nsr/models/texture.py | VolumeRadiance |
| **数据加载** | instant_nsr/datasets/colmap.py | ColmapDataModule |
| **Volume Rendering** | instant_nsr/models/neus.py | forward_(), ray_upsampe_hier() |

这份文档应该能帮助你深入理解NeuS SDF分支的训练框架和PyTorch Lightning的使用方式！
