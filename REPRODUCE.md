# ZeroDiff 最简复现实验指南

本文档按当前实验设置固定使用 `att` 语义嵌入，不再使用 `sent`。

## 0. 服务器信息与环境准备

### 0.1 服务器信息

- GPU: RTX 5090
- Driver: 580.126.09
- System CUDA: 13.0

说明：推荐对齐到已验证可用组合：PyTorch 2.9.1+cu130。

### 0.2 创建虚拟环境（名称固定：zerodiff）

```bash
conda create -n zerodiff python=3.10 -y
conda activate zerodiff
python -m pip install --upgrade pip
```

### 0.3 安装 PyTorch（cu130）

```bash
pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 torchaudio==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

### 0.4 安装实验依赖

```bash
pip install scikit-learn==1.3.0 scipy==1.10.0 numpy==1.24.3 pillow==9.4.0
```

### 0.5 验证环境

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0))"
```

### 0.6 单卡运行约定

本机单卡训练统一使用：

```bash
export CUDA_VISIBLE_DEVICES=0
```

## 1. ce_ce.mat 和 con_paco.mat 的具体作用

读取逻辑见 [pytorch/lqf/Zerodiff-LQF/datasets/image_util.py](pytorch/lqf/Zerodiff-LQF/datasets/image_util.py)。

- `ce_ce.mat`：主视觉特征输入。
	- 在数据加载阶段，`ce_ce.mat` 的 `features` 会作为 `train_feature / test_seen_feature / test_unseen_feature`。
	- 这些特征直接送入生成器与判别器相关训练流程，是 DRG/DFG 的核心视觉表征。
- `con_paco.mat`：对比表示分支输入。
	- 其 `features` 会被读成 `train_paco / test_seen_paco / test_unseen_paco`。
	- 该分支用于提供对比学习语义相关的辅助特征，与主视觉特征联合约束生成质量与泛化。

这两个 `.mat` 都必须包含键：`features`。

## 2. 数据完整性一键检查

在项目根目录执行（固定 att）：

```bash
bash scripts/check_dataset_mats.sh --dataroot ./Dataset --class-embedding att
```

如果还要检查低比例实验所需文件：

```bash
bash scripts/check_dataset_mats.sh --dataroot ./Dataset --class-embedding att --check-split
```

## 3. 最简运行方式（Linux）

必须在项目根目录运行，避免相对路径偏移：

```bash
cd /home/st/pytorch/lqf/Zerodiff-LQF
```

先运行 DRG，再运行 DFG。示例（以 CUB 为例）：

```bash
python zerodiff_DRG_train.py --dataset CUB --dataroot ./Dataset --class_embedding att --gzsl --cuda
```

```bash
python zerodiff_DFG_train.py --dataset CUB --dataroot ./Dataset --class_embedding att --gzsl --cuda --netR_model_path <DRG生成的tar路径>
```

## 4. 输出位置

- 日志目录： [pytorch/lqf/Zerodiff-LQF/log](pytorch/lqf/Zerodiff-LQF/log)
- 权重目录： [pytorch/lqf/Zerodiff-LQF/out](pytorch/lqf/Zerodiff-LQF/out)

说明：DRG 阶段脚本依赖这两个目录已存在，因此已提前创建。
