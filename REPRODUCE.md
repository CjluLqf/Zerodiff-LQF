# ZeroDiff 复现实验指南

本文档给出该仓库在 CUB 数据集上的推荐复现流程，并说明参数入口与数据文件要求。

## 1. 参数配置在哪里

- 全部默认参数定义在 [config_zerodiff.py](config_zerodiff.py)。
- 每个数据集的推荐参数模板在 [scripts](scripts) 目录：
  - [scripts/run_cub_zerodiff_DRG_train.py](scripts/run_cub_zerodiff_DRG_train.py)
  - [scripts/run_cub_zerodiff_DFG_train.py](scripts/run_cub_zerodiff_DFG_train.py)
  - [scripts/run_awa2_zerodiff_DRG_train.py](scripts/run_awa2_zerodiff_DRG_train.py)
  - [scripts/run_awa2_zerodiff_DFG_train.py](scripts/run_awa2_zerodiff_DFG_train.py)
  - [scripts/run_sun_zerodiff_DRG_train.py](scripts/run_sun_zerodiff_DRG_train.py)
  - [scripts/run_sun_zerodiff_DFG_train.py](scripts/run_sun_zerodiff_DFG_train.py)

说明：脚本参数会覆盖默认参数。

## 2. CUB 训练流程（先 DRG 再 DFG）

建议在仓库根目录执行，并先创建输出目录：

```powershell
New-Item -ItemType Directory -Force -Path .\log\CUB | Out-Null
New-Item -ItemType Directory -Force -Path .\out\CUB | Out-Null
```

### 第一步：训练 DRG

```powershell
python .\zerodiff_DRG_train.py --dataset CUB --gzsl --manualSeed 3483 --image_embedding res101 --class_embedding sent --eval_interval 1 --encoded_noise --preprocessing --cuda --nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataroot YOUR_XLSA17_DATA_ROOT --nclass_all 200 --noiseSize 1024 --attSize 1024 --resSize 2048 --gamma_ADV 1 --gamma_VAE 0.0 --embed_type VA --gamma_recons 1.0 --n_T 4 --dim_t 1024 --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 1.0 --batch_size 64 --syn_num 300 --split_percent 100
```

### 第二步：训练 DFG

将下面命令中的 `YOUR_NETR_TAR_PATH` 替换成上一步输出的 DRG 模型路径：

```powershell
python .\zerodiff_DFG_train.py --gzsl --encoded_noise --manualSeed 3483 --preprocessing --cuda --image_embedding res101 --class_embedding sent --nepoch 300 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --nclass_all 200 --dataroot YOUR_XLSA17_DATA_ROOT --dataset CUB --eval_interval 5 --batch_size 64 --noiseSize 1024 --attSize 1024 --resSize 2048 --lr 0.0001 --classifier_lr 0.001 --gamma_recons 0.01 --dec_lr 0.0001 --gamma_ADV 10 --gamma_VAE 1.0 --embed_type VA --n_T 4 --dim_t 1024 --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 2.0 --factor_dist 1.5 --split_percent 100 --syn_num 1440 --netR_model_path YOUR_NETR_TAR_PATH
```

## 3. 需要准备/上传的数据文件

数据读取逻辑见 [datasets/image_util.py](datasets/image_util.py)。

设 `--dataroot` 为 `D:/Dataset/xlsa17/data` 时，目录应为：

- `D:/Dataset/xlsa17/data/CUB/`
- `D:/Dataset/xlsa17/data/AWA2/`
- `D:/Dataset/xlsa17/data/SUN/`

每个数据集目录下必需文件：

- `res101.mat`
- `<class_embedding>_splits.mat`
- `ce_ce.mat`
- `con_paco.mat`

其中 `<class_embedding>_splits.mat` 取决于参数：

- `--class_embedding att` 时为 `att_splits.mat`
- `--class_embedding sent` 时为 `sent_splits.mat`

若做小样本比例实验（`--split_percent` 为 10 或 30），还需要：

- `split_10percent.mat`
- `split_30percent.mat`

## 4. 运行结果检查

- 日志输出在 `log/<dataset>/`。
- 模型权重在 `out/<dataset>/`。
- DRG 阶段应先产出一个可被 DFG 阶段读取的 `*_gzsl.tar` 文件。
