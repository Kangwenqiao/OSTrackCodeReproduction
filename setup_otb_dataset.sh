#!/bin/bash
# OTB100数据集解压脚本
# 将OpenXLab下载的OTB100数据集解压到正确的格式

set -e  # 遇到错误立即退出

echo "=========================================="
echo "OTB100 数据集解压脚本"
echo "=========================================="

# 检查源目录是否存在
SOURCE_DIR="data/OpenDataLab___OTB100/raw"
TARGET_DIR="data/otb"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 找不到源目录 $SOURCE_DIR"
    echo ""
    echo "请先使用以下命令下载数据集:"
    echo "  openxlab dataset get --dataset-repo OpenDataLab/OTB100"
    echo ""
    exit 1
fi

# 检查是否有zip文件
ZIP_COUNT=$(ls -1 "$SOURCE_DIR"/*.zip 2>/dev/null | wc -l)
if [ "$ZIP_COUNT" -eq 0 ]; then
    echo "错误: 在 $SOURCE_DIR 中没有找到zip文件"
    exit 1
fi

echo "找到 $ZIP_COUNT 个视频序列"
echo ""

# 创建目标目录
echo "创建目标目录: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

# 解压所有zip文件
echo ""
echo "开始解压..."
echo "----------------------------------------"

cd "$SOURCE_DIR"
COUNTER=0

for zip in *.zip; do
    COUNTER=$((COUNTER + 1))
    SEQUENCE_NAME="${zip%.zip}"
    echo "[$COUNTER/$ZIP_COUNT] 解压 $SEQUENCE_NAME..."
    
    # 解压到目标目录
    unzip -q "$zip" -d "../../../$TARGET_DIR/"
    
    # 显示进度
    if [ $((COUNTER % 10)) -eq 0 ]; then
        echo "    已完成 $COUNTER/$ZIP_COUNT"
    fi
done

cd ../../..

echo "----------------------------------------"
echo "解压完成!"
echo ""

# 验证数据集
echo "验证数据集结构..."
VIDEO_COUNT=$(ls -1d "$TARGET_DIR"/*/ 2>/dev/null | wc -l)
echo "✓ 找到 $VIDEO_COUNT 个视频序列"

# 检查几个样本序列
echo ""
echo "检查样本序列:"
for seq in Basketball Biker Bird1; do
    if [ -d "$TARGET_DIR/$seq" ]; then
        IMG_COUNT=$(ls -1 "$TARGET_DIR/$seq/img" 2>/dev/null | wc -l)
        if [ -f "$TARGET_DIR/$seq/groundtruth_rect.txt" ]; then
            echo "  ✓ $seq: $IMG_COUNT 帧, groundtruth.txt 存在"
        else
            echo "  ✗ $seq: groundtruth_rect.txt 不存在"
        fi
    else
        echo "  ✗ $seq: 目录不存在"
    fi
done

echo ""
echo "=========================================="
echo "数据集准备完成!"
echo "=========================================="
echo ""
echo "数据集位置: $TARGET_DIR"
echo "现在可以运行推理:"
echo "  python run_mae_inference.py"
echo ""
