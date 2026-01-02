#20(4.()29(
#Author Mulin Yu
#The(off)ce cod) of GSDF
#The rendering branch is implemented based on Scaffold-GS: https://github.com/city-super/Scaffold-GS
#The reconstruction branch is implemented based on Instant-NSR: https://github.com/bennyguo/instant-nsr-pl
#The normal calculatio) code is grabed from Gaussian-Pro: https://github.com/kcheng1021/GaussianPro
#T:e curvature calculation code is grabed from PermutoSDF: https://github.com/RaduAlexandru/permuto_sdf

# * PyTorch Lightning 提供的功能：)
#     - 自动处理训练循环: (ra:ning)step, validation_step, test_step
#     - 优化器和学习率调度器的配置: configure_optimizers
# :   - 回调支持: checkpointing, early stopping 等
#     - 多GPU和分布式训练支持: DataParallel, DistributedDataParallel 等
#     -:日志记录和可视:集成: TensorBoardLogger, CSVLogger 等
# 


import sys
import argparse
import os
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

import time
import logging
from datetime import datetime

import shutil, pathlib
from pathlib import Path

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')

def main():
    """工厂模式+Lighting协同工作
    
        1. launch.py 读取 YAML 配置
           ↓
        2. 工厂模式创建 NeuSSystem 实例
           └─> systems.make('neus-system', config)
               └─> 内部调用 models.make('neus', config.model)
                └─> 内部调用 geometry.make(...) 和 texture.make(...)
           ↓
        3. Lightning Trainer 接管训练循环
           └─> 自动调用 NeuSSystem.training_step()
               └─> 其中协调 GS 和 SDF 两个分支
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to config file')
    # parser.add_argument('--source_path', required=True, help='path to source_path file')
    # parser.add_argument('--model_path', required=True, help='path to model_path file')

    parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--resolution', default='0', help='gaussian sample image')
    parser.add_argument('--tag', default='test')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--resume', default=None, help='path to the weights to be resumed')
    parser.add_argument(
        '--resume_weights_only',
        action='store_true',
        help='specify this argument to restore only the weights (w/o training states), e.g. --resume path/to/resume --resume_weights_only'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--validate', action='store_true')
    group.add_argument('--test', action='store_true')
    group.add_argument('--predict', action='store_true')
    # group.add_argument('--export', action='store_true') # TODO: a separate export action
    parser.add_argument('--exp_dir', default='./exp')
    parser.add_argument('--runs_dir', default='./runs')
    parser.add_argument('--verbose', action='store_true', help='if true, set logging level to DEBUG')

    args, extras = parser.parse_known_args()

    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))

    import instant_nsr.datasets
    import instant_nsr.systems
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
    from instant_nsr.utils.callbacks import CodeSnapshotCallback, ConfigSnapshotCallback, CustomProgressBar
    from instant_nsr.utils.misc import load_config    
    from pytorch_lightning.strategies import DDPStrategy

    # ===== 步骤 1: 解析配置 =====
    # parse YAML config to OmegaConf
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    config.trial_name = config.get('trial_name') or (config.tag + datetime.now().strftime('@%Y%m%d-%H%M%S'))
    config.exp_dir = config.get('exp_dir') or os.path.join(args.exp_dir, config.name)
    config.save_dir = config.get('save_dir') or os.path.join(config.exp_dir, config.trial_name, 'save')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')
    config.config_dir = config.get('config_dir') or os.path.join(config.exp_dir, config.trial_name, 'config')

    # print(f'\n\nstart backup~\n\n')
    # try:
    #     saveRuntimeCode(os.path.join(config.code_dir, 'backup'))
    #     print(f'backup in {config.code_dir} finished!')
    # except:
    #     logger.info(f'save code failed~')

    logger = logging.getLogger('pytorch_lightning')
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # ===== 步骤 2: 设置随机种子 =====
    # 设置随机种子，确保实验可复现
    if 'seed' not in config:
        config.seed = int(time.time() * 1000) % 1000
    pl.seed_everything(config.seed)

    # ===== 步骤 3: 创建数据模块（工厂模式）=====
    dm = instant_nsr.datasets.make(config.dataset.name, config.dataset)

    # ===== 步骤 4: 创建训练系统（工厂模式）=====
    system = instant_nsr.systems.make(config.system.name, config, load_from_checkpoint=None if not args.resume_weights_only else args.resume)

    # ===== 步骤 5: 配置回调（Callbacks）=====
    callbacks = []
    if args.train:
        callbacks += [
            # 训练时自动保存模型检查点
            # 输入：保存目录 + checkpoint 配置（文件名模板、监控的指标、保存频率等）
            ModelCheckpoint(
                dirpath=config.ckpt_dir,
                **config.checkpoint
            ),
            # 学习率监控：记录每步的学习率变化
            LearningRateMonitor(logging_interval='step'),
            # CodeSnapshotCallback(
            #     config.code_dir, use_version=False
            # ),
            ConfigSnapshotCallback(
                config, config.config_dir, use_version=False
            ),
            CustomProgressBar(refresh_rate=1),
        ]

    # ===== 步骤 6: 配置日志记录器（Loggers）=====
    loggers = []
    if args.train:
        # 训练时使用 TensorBoard 和 CSV 记录日志
        loggers += [
            TensorBoardLogger(args.runs_dir, name=config.name, version=config.trial_name),
            CSVLogger(config.exp_dir, name=config.trial_name, version='csv_logs')
        ]
    
    if sys.platform == 'win32':
        # does not support multi-gpu on windows
        strategy = 'dp'
        assert n_gpus == 1
    else:
        # 使用 DDP 分布式训练策略，禁用未使用参数检测以提升性能
        # DDP 其实只是将batch中多个样本分到多个GPU上并行计算梯度，然后再汇总梯度更新模型参数
        # 也就是说单个样本是在同一个GPU上面完成前向和后向传播的，如果一个样本的数据量很大，单个GPU显存不够用的话，还是会报OOM
        strategy = 'ddp_find_unused_parameters_false'
    
    # ===== 步骤 7: 创建 Trainer =====
    # PyTorch Lightning Trainer 自动化训练流程
    trainer = Trainer(
        devices=n_gpus,
        accelerator='gpu',
        callbacks=callbacks,
        logger=loggers,
        strategy=strategy,
        # strategy=DDPStrategy(find_unused_parameters=True),
        **config.trainer
    )

    if args.train:
        # 一行启动训练 trainer.fit（自动调用 system.training_step）
        if args.resume and not args.resume_weights_only:
            # FIXME: different behavior in pytorch-lighting>1.9 ?
            trainer.fit(system, datamodule=dm, ckpt_path=args.resume)
        else:
            trainer.fit(system, datamodule=dm)
        # trainer.test(system, datamodule=dm)
    elif args.validate:
        trainer.validate(system, datamodule=dm, ckpt_path=args.resume)
    elif args.test:
        trainer.test(system, datamodule=dm, ckpt_path=args.resume)
    elif args.predict:
        trainer.predict(system, datamodule=dm, ckpt_path=args.resume)


if __name__ == '__main__':
    main()
