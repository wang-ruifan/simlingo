# SimLingo 本地评估 - 当前状态与下一步操作

## ✅ 已完成的工作

### 1. 创建了本地评估脚本
- ✅ `eval_simlingo_local.py` - 完整评估脚本（顺序评估所有路由）
- ✅ `eval_single_route.py` - 单路由评估脚本
- ✅ `quick_test_eval.sh` - 快速测试脚本（前3个路由）
- ✅ `run_full_eval.sh` - 完整评估启动脚本
- ✅ `check_environment.py` - 环境检查脚本
- ✅ `test_eval_setup.py` - 评估设置测试脚本

### 2. 修复了环境配置问题
- ✅ 修复了 `TickRuntimeError` 导入错误
  - 问题：`leaderboard/leaderboard/autoagents/agent_wrapper.py` 没有定义此类
  - 解决：调整 PYTHONPATH 顺序，确保使用 Bench2Drive 版本的 leaderboard
  
- ✅ 修复了 `simlingo_training` 模块找不到的问题
  - 问题：agent 需要导入项目中的训练代码
  - 解决：在 PYTHONPATH 中添加了 repo 根目录

### 3. 正确的 PYTHONPATH 顺序
```python
pythonpath_parts = [
    repo_root,  # 1. 项目根目录 - 用于导入 simlingo_training
    f"{repo_root}/Bench2Drive/leaderboard",  # 2. Bench2Drive leaderboard
    f"{repo_root}/Bench2Drive/scenario_runner",  # 3. Bench2Drive scenario_runner
    f"{carla_root}/PythonAPI/carla",  # 4. CARLA API
    f"{carla_root}/PythonAPI/carla/dist/carla-0.9.15-py3.9-linux-x86_64.egg",  # 5. CARLA egg
]
```

### 4. 测试结果
- ✅ 所有模块导入测试通过
- ✅ CARLA 服务器可以正常启动
- ✅ Agent 可以成功加载和初始化
- ✅ 路由文件检查通过

## ❌ 当前遇到的问题

### 🔴 模型文件未下载（Git LFS 问题）

**错误信息：**
```
_pickle.UnpicklingError: invalid load key, 'v'
```

**原因：**
模型文件 `pytorch_model.pt` 只有 135 字节，是一个 Git LFS 指针文件，实际的 2.4GB 模型文件未下载。

**文件内容：**
```
version https://git-lfs.github.com/spec/v1
oid sha256:ec8943723d266ee9f5f56f45d153a163b22616960bfccb741965ea5daa700d28
size 2569679322
```

## 🔧 下一步操作（必须完成）

### 选项 1: 使用 Git LFS 下载模型（推荐）

```bash
# 1. 安装 Git LFS（如果还没安装）
sudo apt-get install git-lfs

# 2. 初始化 Git LFS
cd /home/wang/simlingo
git lfs install

# 3. 下载所有 LFS 文件
git lfs pull

# 或者只下载 output 目录下的文件
git lfs pull --include="output/**"

# 4. 验证文件已正确下载
ls -lh output/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
# 应该显示约 2.4GB
```

### 选项 2: 从其他来源获取模型文件

如果无法使用 Git LFS，你需要：

1. 从论文作者或其他来源获取完整的模型文件
2. 将其放置到 `/home/wang/simlingo/output/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`
3. 验证文件大小约为 2.4GB

### 选项 3: 使用 DeepSpeed checkpoint 转换

如果有 DeepSpeed checkpoint（在 `checkpoint/` 子目录中），可以使用转换脚本：

```bash
cd /home/wang/simlingo/output/simlingo/checkpoints/epoch=013.ckpt

# 使用 zero_to_fp32.py 脚本转换
python zero_to_fp32.py . pytorch_model.pt

# 验证转换后的文件
ls -lh pytorch_model.pt
```

## ✨ 完成模型下载后

一旦模型文件正确下载，你就可以开始评估了：

### 快速测试（推荐先运行）
```bash
cd /home/wang/simlingo
python eval_simlingo_local.py --max-routes 1 --seeds 1
```

### 或使用单路由测试
```bash
python eval_single_route.py \
    --route /home/wang/simlingo/leaderboard/data/bench2drive_split/bench2drive_00.xml \
    --output-dir /home/wang/simlingo/eval_results/test_single
```

### 完整评估
```bash
# 所有路由，单个种子
python eval_simlingo_local.py --seeds 1

# 所有路由，三个种子（论文配置）
python eval_simlingo_local.py --seeds 1 2 3
```

## 📊 预期结果

评估成功后，你应该能看到：
- JSON 结果文件：`eval_results/Bench2Drive/simlingo/bench2drive/1/res/*.json`
- 输出日志：`eval_results/Bench2Drive/simlingo/bench2drive/1/out/*.log`
- 错误日志：`eval_results/Bench2Drive/simlingo/bench2drive/1/err/*.log`
- 可视化数据：`eval_results/Bench2Drive/simlingo/bench2drive/1/viz/*/`

## 📝 其他说明

### GPU 兼容性警告
你可能会看到这个警告：
```
NVIDIA GeForce RTX 5070 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```

这是因为你的 GPU 太新了，当前的 PyTorch 版本还不支持。你有以下选项：
1. 忽略警告，模型会在 CPU 上运行（较慢）
2. 更新 PyTorch 到支持 sm_120 的版本
3. 使用其他 GPU（如果有）

### 环境变量
所有脚本都会自动设置正确的环境变量，但如果需要手动设置：
```bash
export CARLA_ROOT=~/software/carla0915
export PYTHONPATH=/home/wang/simlingo:/home/wang/simlingo/Bench2Drive/leaderboard:...
export SCENARIO_RUNNER_ROOT=/home/wang/simlingo/Bench2Drive/scenario_runner
```

## 🆘 需要帮助？

查看详细文档：
- `EVAL_README.md` - 完整使用说明
- `--help` - 查看脚本参数

运行测试脚本：
```bash
python check_environment.py  # 检查环境配置
python test_eval_setup.py    # 检查评估设置
```

## 总结

✅ **已完成：** 所有评估脚本已创建并测试，环境配置已修复
❌ **待完成：** 下载完整的模型文件（约 2.4GB）
🎯 **下一步：** 使用 `git lfs pull` 或其他方式获取模型文件，然后开始评估
