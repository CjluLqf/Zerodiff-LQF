# ZeroDiff 最简复现实验指南

本文档固定使用项目内路径，避免把数据和输出写到其他目录。

## 1. 已创建的最小目录

在项目根目录 [pytorch/lqf/Zerodiff-LQF](pytorch/lqf/Zerodiff-LQF) 下：

- [pytorch/lqf/Zerodiff-LQF/Dataset](pytorch/lqf/Zerodiff-LQF/Dataset)
- [pytorch/lqf/Zerodiff-LQF/Dataset/AWA2](pytorch/lqf/Zerodiff-LQF/Dataset/AWA2)
- [pytorch/lqf/Zerodiff-LQF/Dataset/CUB](pytorch/lqf/Zerodiff-LQF/Dataset/CUB)
- [pytorch/lqf/Zerodiff-LQF/Dataset/SUN](pytorch/lqf/Zerodiff-LQF/Dataset/SUN)
- [pytorch/lqf/Zerodiff-LQF/log](pytorch/lqf/Zerodiff-LQF/log)
- [pytorch/lqf/Zerodiff-LQF/out](pytorch/lqf/Zerodiff-LQF/out)

## 2. .mat 文件放置位置

数据读取逻辑在 [pytorch/lqf/Zerodiff-LQF/datasets/image_util.py](pytorch/lqf/Zerodiff-LQF/datasets/image_util.py)。
训练时建议统一使用参数：--dataroot ./Dataset。

三个数据集都按下面路径放置：

- AWA2: [pytorch/lqf/Zerodiff-LQF/Dataset/AWA2](pytorch/lqf/Zerodiff-LQF/Dataset/AWA2)
- CUB: [pytorch/lqf/Zerodiff-LQF/Dataset/CUB](pytorch/lqf/Zerodiff-LQF/Dataset/CUB)
- SUN: [pytorch/lqf/Zerodiff-LQF/Dataset/SUN](pytorch/lqf/Zerodiff-LQF/Dataset/SUN)

每个数据集目录最少需要：

- res101.mat
- ce_ce.mat
- con_paco.mat
- att_splits.mat 或 sent_splits.mat（二选一，取决于 --class_embedding）

若做低比例实验（--split_percent 为 10 或 30），额外放：

- split_10percent.mat
- split_30percent.mat

`ce_ce.mat` 和 `con_paco.mat` 说明：

- 这两个文件都是特征文件，训练代码会直接读取其中的 `features` 键。
- 优先从官方 README 的 fine-tuned features 网盘下载后放入对应数据集目录。
- 若你想自行生成，可使用：
	- [pytorch/lqf/Zerodiff-LQF/FineTune/PACO/extract_features_ce_ce.py](pytorch/lqf/Zerodiff-LQF/FineTune/PACO/extract_features_ce_ce.py)
	- [pytorch/lqf/Zerodiff-LQF/FineTune/PACO/extract_features_con_paco.py](pytorch/lqf/Zerodiff-LQF/FineTune/PACO/extract_features_con_paco.py)
	其中第二个脚本默认输出名是 `ce_paco.mat`，使用前需要改名为 `con_paco.mat`。

## 3.1 数据完整性一键检查

在项目根目录执行：

```bash
bash scripts/check_dataset_mats.sh --dataroot ./Dataset --class-embedding sent
```

如果你使用 `--class_embedding att` 训练，请改为：

```bash
bash scripts/check_dataset_mats.sh --dataroot ./Dataset --class-embedding att
```

如果还要检查低比例实验所需文件：

```bash
bash scripts/check_dataset_mats.sh --dataroot ./Dataset --class-embedding sent --check-split
```

## 4. 最简运行方式（Linux）

必须在项目根目录运行，避免相对路径偏移：

```bash
cd /home/st/pytorch/lqf/Zerodiff-LQF
```

先运行 DRG，再运行 DFG。示例（以 CUB 为例）：

```bash
python zerodiff_DRG_train.py --dataset CUB --dataroot ./Dataset --class_embedding sent --gzsl --cuda
```

```bash
python zerodiff_DFG_train.py --dataset CUB --dataroot ./Dataset --class_embedding sent --gzsl --cuda --netR_model_path <DRG生成的tar路径>
```

## 5. 输出位置

- 日志目录： [pytorch/lqf/Zerodiff-LQF/log](pytorch/lqf/Zerodiff-LQF/log)
- 权重目录： [pytorch/lqf/Zerodiff-LQF/out](pytorch/lqf/Zerodiff-LQF/out)

说明：DRG 阶段脚本依赖这两个目录已存在，因此已提前创建。
