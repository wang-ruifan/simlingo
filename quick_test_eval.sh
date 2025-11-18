#!/bin/bash
# SimLingo 本地评估快速启动脚本

echo "==========================================="
echo "SimLingo 本地评估 - 快速测试"
echo "==========================================="
echo ""
echo "这个脚本会评估前3个路由作为测试"
echo "如果测试成功，你可以运行完整评估"
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

echo "✓ 模型文件: $CHECKPOINT"
echo "✓ 路由目录: $ROUTE_PATH"
echo ""

# 确认
read -p "按 Enter 开始测试评估 (Ctrl+C 取消)..."

# 运行评估
cd "$REPO_ROOT"
python eval_simlingo_local.py \
    --checkpoint "$CHECKPOINT" \
    --route-path "$ROUTE_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --seeds 1 \
    --max-routes 3 \
    --port 2000 \
    --tm-port 8000 \
    --gpu-rank 0

echo ""
echo "==========================================="
echo "测试评估完成！"
echo "==========================================="
echo ""
echo "结果保存在: $OUTPUT_DIR/simlingo/bench2drive/1/"
echo ""
echo "如果测试成功，运行完整评估："
echo "  python eval_simlingo_local.py --seeds 1"
echo ""
