# OSTrack - 完整配置与推理指南

这是 OSTrack 目标跟踪框架的完整设置指南，从零开始配置环境到完成OTB数据集推理。

## 📋 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [详细步骤](#详细步骤)
  - [1. 环境配置](#1-环境配置)
  - [2. 项目设置](#2-项目设置)
  - [3. 数据集准备](#3-数据集准备)
  - [4. 下载预训练权重](#4-下载预训练权重)
  - [5. OTB推理](#5-otb推理)
- [故障排除](#故障排除)


## ⚡ 快速开始

如果你已经熟悉Python环境配置，可以使用以下命令快速完成整个流程：

```bash
# 1. 克隆仓库并进入目录
git clone <repository-url>
cd OSTrack

# 2. 安装uv (如果还没安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 创建虚拟环境并安装PyTorch (CUDA 11.7)
uv venv --python 3.8
source .venv/bin/activate
uv pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117

# 4. 安装项目依赖
uv pip install -r requirements.txt

# 5. 设置项目路径
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output

# 6. 下载OTB数据集
uv pip install openxlab
openxlab login  # 输入你的AK/SK
openxlab dataset get --dataset-repo OpenDataLab/OTB100
bash setup_otb_dataset.sh

# 7. 下载MAE预训练权重
mkdir -p pretrained_models
cd pretrained_models
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
cd ..

# 8. 运行快速推理（MAE模型）
python run_mae_inference.py
```

---

## 📚 详细步骤

### 1. 环境配置

#### 1.1 安装uv (如果还没安装)

```bash
# 使用官方安装脚本
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者使用pip安装
pip install uv

# 验证安装
uv --version
```

#### 1.2 创建虚拟环境

```bash
# 进入项目目录
cd /path/to/OSTrack

# 使用uv创建Python 3.8虚拟环境
uv venv --python 3.8

# 激活虚拟环境
source .venv/bin/activate
```

#### 1.3 安装PyTorch

```bash
# 使用uv安装PyTorch (CUDA 11.7版本，兼容CUDA 12.x驱动)
uv pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
```

#### 1.4 安装核心依赖

**方法1: 使用requirements.txt (推荐)**

创建 `requirements.txt` 文件：

```txt
PyYAML==6.0.1
easydict==1.10
cython==0.29.36
opencv-python==4.12.0.88
pillow==10.4.0
jpeg4py==0.1.4
numpy==1.24.4
pandas==2.0.3
scipy==1.10.1
timm==1.0.22
einops==0.8.0
tqdm==4.67.1
tensorboard==2.18.0
wandb==0.19.0
pycocotools==2.0.8
lmdb==1.5.1
visdom==0.2.4
matplotlib==3.7.5
seaborn==0.13.2
colorama==0.4.6
tikzplotlib==0.10.1
setuptools==59.5.0
openxlab==0.0.37
```

然后一键安装（uv会并行下载，速度极快）：

```bash
uv pip install -r requirements.txt
```

**方法2: 单独安装**

```bash
# 使用uv可以一次安装多个包，速度更快
uv pip install \
    PyYAML==6.0.1 \
    easydict==1.10 \
    cython==0.29.36 \
    opencv-python==4.12.0.88 \
    pillow==10.4.0 \
    jpeg4py==0.1.4 \
    numpy==1.24.4 \
    pandas==2.0.3 \
    scipy==1.10.1 \
    timm==1.0.22 \
    einops==0.8.0 \
    tqdm==4.67.1 \
    tensorboard==2.18.0 \
    wandb==0.19.0 \
    pycocotools==2.0.8 \
    lmdb==1.5.1 \
    visdom==0.2.4 \
    matplotlib==3.7.5 \
    seaborn==0.13.2 \
    colorama==0.4.6 \
    tikzplotlib==0.10.1 \
    setuptools==59.5.0 \
    openxlab==0.0.37
```

#### 1.5 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

预期输出：

```
PyTorch: 1.13.1+cu117
CUDA: True
GPU: NVIDIA GeForce RTX 4090
```

---

### 2. 项目设置

#### 2.1 配置路径

```bash
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

这个命令会创建两个配置文件：

- `lib/train/admin/local.py` - 训练相关路径
- `lib/test/evaluation/local.py` - 测试相关路径

#### 2.2 验证目录结构

```bash
# 创建必要的目录
mkdir -p data pretrained_models output
```

---

### 3. 数据集准备

#### 3.1 安装OpenXLab工具

```bash
pip install openxlab
```

#### 3.2 登录OpenXLab

```bash
openxlab login
```

你需要输入AK/SK，从这里获取：https://sso.openxlab.org.cn/usercenter

#### 3.3 下载OTB100数据集

```bash
# 查看数据集信息
openxlab dataset info --dataset-repo OpenDataLab/OTB100

# 下载数据集 (约3GB，uv环境中openxlab工作正常)
openxlab dataset get --dataset-repo OpenDataLab/OTB100
```

数据将下载到: `data/OpenDataLab___OTB100/`

#### 3.4 解压数据集

```bash
# 使用提供的脚本自动解压
bash setup_otb_dataset.sh
```

或手动解压：

```bash
mkdir -p data/otb
cd data/OpenDataLab___OTB100/raw
for zip in *.zip; do
    echo "解压 $zip..."
    unzip -q "$zip" -d ../../otb/
done
cd ../../..
```

#### 3.5 验证数据集

```bash
ls data/otb/
```

应该看到100个视频序列目录：

```
Basketball  Biker  Bird1  Bird2  BlurBody  BlurCar1  ...
```

每个目录结构：

```
data/otb/Basketball/
├── img/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
└── groundtruth_rect.txt
```

---

### 4. 下载预训练权重

#### 4.1 下载MAE预训练权重 (快速演示用)

```bash
mkdir -p pretrained_models
cd pretrained_models
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
cd ..
```

**注意**: MAE权重仅包含backbone，tracking head是随机初始化的，性能较低（约20-30% Success AUC）。

#### 4.2 下载完整训练的检查点 (高性能推理用)

从 [Google Drive](https://drive.google.com/drive/folders/1PS4inLS8bWNCecpYZ0W2fE5-A04DvTcd) 下载完整训练的模型权重：

推荐下载：

- `vitb_256_mae_ce_32x4_ep300.pth.tar` - 256×256输入，较快
- `vitb_384_mae_ce_32x4_ep300.pth.tar` - 384×384输入，更精确

下载后放置在：

```
output/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_ep300/
└── OSTrack_ep0300.pth.tar
```

---

### 5. OTB推理

#### 5.1 快速演示 (MAE预训练模型)

```bash
# 一键运行推理
python run_mae_inference.py

# 可选参数
python run_mae_inference.py --threads 4 --num_gpus 1
```

此脚本自动完成：

1. 将MAE权重转换为OSTrack格式
2. 在OTB100上运行推理
3. 生成性能报告

**预期性能** (MAE模型):

- Success AUC: ~20-30%
- Precision: ~30-40%

#### 5.2 高性能推理 (完整训练模型)

```bash
# 使用256×256模型
python tracking/test.py ostrack vitb_256_mae_ce_32x4_ep300 --dataset otb --threads 4 --num_gpus 1

# 使用384×384模型 (更精确但更慢)
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset otb --threads 4 --num_gpus 1
```

#### 5.3 生成评估报告

```bash
python tracking/analysis_results.py
```

需要修改脚本中的tracker名称和配置。

**预期性能** (完整训练模型):

- Success AUC: ~68-70%
- Precision: ~88-90%

#### 5.4 输出位置

推理结果保存在：

```
output/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300/otb/
├── Basketball.txt
├── Biker.txt
└── ...
```

每个.txt文件包含该视频序列的跟踪边界框坐标。

---

## 🐛 故障排除

### uv相关问题

**uv命令未找到**

```bash
# 重新安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或添加到PATH
export PATH="$HOME/.cargo/bin:$PATH"
source ~/.bashrc
```

**uv pip速度优势**

- uv使用Rust编写，比pip快10-100倍
- 自动并行下载和解析依赖
- 使用: `uv pip install` 替代 `pip install`

### CUDA不可用

```bash
# 检查CUDA版本
nvidia-smi

# 检查PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

如果返回False，重新安装匹配的PyTorch版本。

### 内存不足

```bash
# 减少批次大小（修改配置文件）
vim experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml
# 修改 TRAIN.BATCH_SIZE 为更小的值
```

### 数据集路径错误

```bash
# 检查路径配置
cat lib/test/evaluation/local.py
```

确保`otb_path`指向正确的数据集目录。

### 推理速度慢

```bash
# 增加线程数
python tracking/test.py ostrack vitb_256_mae_ce_32x4_ep300 --dataset otb --threads 8 --num_gpus 1
```

### OpenXLab下载失败

如果网络不稳定，可以手动下载：

1. 访问 https://openxlab.org.cn/datasets/OpenDataLab/OTB100
2. 手动下载所有.zip文件到 `data/OpenDataLab___OTB100/raw/`
3. 运行解压脚本：`bash setup_otb_dataset.sh`

---

## 📊 性能对比

| 模型           | Success AUC | Precision | 速度 (RTX 4090) |
| -------------- | ----------- | --------- | --------------- |
| MAE预训练      | ~25%        | ~35%      | ~180 FPS        |
| 完整训练 (256) | ~68%        | ~88%      | ~180 FPS        |
| 完整训练 (384) | ~70%        | ~90%      | ~120 FPS        |

---

## 📖 更多功能

### 可视化调试

```bash
# 启动Visdom服务器
visdom

# 运行推理并可视化
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset otb --threads 1 --debug 1
```

在浏览器打开 http://localhost:8097 查看可视化结果。

### 测试其他数据集

```bash
# LaSOT
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset lasot --threads 4 --num_gpus 1

# GOT-10K
python tracking/test.py ostrack vitb_384_mae_ce_32x4_got10k_ep100 --dataset got10k_test --threads 4 --num_gpus 1

# TrackingNet
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset trackingnet --threads 4 --num_gpus 1
```

### 模型训练

```bash
# 单GPU训练
python tracking/train.py \
    --script ostrack \
    --config vitb_256_mae_ce_32x4_ep300 \
    --save_dir ./output \
    --mode single \
    --use_wandb 0
```

训练时间 (RTX 4090):

- 256×256模型: ~16-18小时
- 384×384模型: ~24-28小时

---

## 🔗 相关资源

- **论文**: [Joint Feature Learning and Relation Modeling for Tracking](https://arxiv.org/abs/2203.11991)
- **预训练模型**: [Google Drive](https://drive.google.com/drive/folders/1PS4inLS8bWNCecpYZ0W2fE5-A04DvTcd)
- **OTB100数据集**: [OpenXLab](https://openxlab.org.cn/datasets/OpenDataLab/OTB100)
- **MAE预训练权重**: [Facebook Research](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)

---

## 📝 引用

```bibtex
@inproceedings{ye2022ostrack,
  title={Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework},
  author={Ye, Botao and Chang, Hong and Ma, Bingpeng and Shan, Shiguang and Chen, Xilin},
  booktitle={ECCV},
  year={2022}
}
```

---

**最后更新**: 2024-11-15