#!/bin/bash
# SimLingo 完整评估启动脚本（所有路由，单个种子）

echo "==========================================="
echo "SimLingo 完整评估"
echo "==========================================="
echo ""
echo "这将评估所有 Bench2Drive 路由"
echo "使用种子: 1"
echo ""

# 配置
REPO_ROOT="/home/wang/simlingo"
CHECKPOINT="${REPO_ROOT}/output/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
ROUTE_PATH="${REPO_ROOT}/leaderboard/data/bench2drive_split"
OUTPUT_DIR="${REPO_ROOT}/eval_results/Bench2Drive"

# 检查文件
echo "检查配置..."
if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: 找不到模型文件: $CHECKPOINT"
    exit 1
fi

if [ ! -d "$ROUTE_PATH" ]; then
    echo "错误: 找不到路由目录: $ROUTE_PATH"
    exit 1
fi

# 统计路由数量
ROUTE_COUNT=$(ls "$ROUTE_PATH"/*.xml 2>/dev/null | wc -l)

echo "✓ 模型文件: $CHECKPOINT"
echo "✓ 路由目录: $ROUTE_PATH"
echo "✓ 路由数量: $ROUTE_COUNT"
echo ""
echo "预计时间: 取决于每个路由的复杂度（可能需要数小时）"
echo ""

# 确认
read -p "确认开始完整评估？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 运行评估
cd "$REPO_ROOT"
python eval_simlingo_local.py \
    --checkpoint "$CHECKPOINT" \
    --route-path "$ROUTE_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --seeds 1 \
    --port 2000 \
    --tm-port 8000 \
    --gpu-rank 0

echo ""
echo "==========================================="
echo "完整评估完成！"
echo "==========================================="
echo ""
echo "结果保存在: $OUTPUT_DIR/simlingo/bench2drive/1/"
echo ""
