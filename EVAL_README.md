# SimLingo 本地评估脚本使用说明

## 概述

我为你创建了两个本地评估脚本，用于在本地机器上顺序运行 SimLingo 评估，而不是使用 Slurm 集群并行执行。

## 脚本说明

### 1. `eval_simlingo_local.py` - 完整评估脚本

这个脚本会：
- 自动启动和停止 CARLA 服务器
- 顺序评估所有路由（而不是并行）
- 支持多个随机种子
- 自动创建输出目录结构
- 保存评估结果、日志和可视化数据

**基本用法：**
```bash
# 使用默认配置运行所有路由
python eval_simlingo_local.py

# 只运行前3个路由进行测试
python eval_simlingo_local.py --max-routes 3

# 使用多个种子
python eval_simlingo_local.py --seeds 1 2 3

# 自定义端口
python eval_simlingo_local.py --port 3000 --tm-port 9000

# 查看所有选项
python eval_simlingo_local.py --help
```

**主要参数：**
- `--checkpoint`: 模型文件路径（默认：`/home/wang/simlingo/output/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`）
- `--route-path`: 路由文件目录（默认：`/home/wang/simlingo/leaderboard/data/bench2drive_split`）
- `--output-dir`: 结果输出目录（默认：`/home/wang/simlingo/eval_results/Bench2Drive`）
- `--seeds`: 随机种子列表（默认：`[1]`）
- `--max-routes`: 限制评估路由数量（用于测试）
- `--port`: CARLA 服务器端口（默认：2000）
- `--tm-port`: Traffic Manager 端口（默认：8000）
- `--gpu-rank`: GPU 编号（默认：0）

### 2. `eval_single_route.py` - 单路由评估脚本

这个脚本用于快速测试单个路由。

**基本用法：**
```bash
# 评估单个路由（自动启动CARLA）
python eval_single_route.py --route /home/wang/simlingo/leaderboard/data/bench2drive_split/bench2drive_00.xml

# 使用已运行的CARLA服务器（不启动新服务器）
python eval_single_route.py --route /path/to/route.xml --no-server --port 2000

# 查看所有选项
python eval_single_route.py --help
```

**主要参数：**
- `--route`: 路由XML文件路径（**必需**）
- `--checkpoint`: 模型文件路径
- `--output-dir`: 结果输出目录（默认：`/home/wang/simlingo/eval_results/single_route`）
- `--no-server`: 不启动CARLA服务器（假设已在运行）
- `--seed`: 随机种子（默认：1）
- `--port`: CARLA 服务器端口
- `--tm-port`: Traffic Manager 端口

## 输出结构

评估结果会保存在如下目录结构中：

```
eval_results/Bench2Drive/
└── simlingo/
    └── bench2drive/
        └── 1/  # seed
            ├── res/     # 评估结果JSON文件
            │   ├── 000_res.json
            │   ├── 001_res.json
            │   └── ...
            ├── out/     # 标准输出日志
            │   ├── 000_out.log
            │   ├── 001_out.log
            │   └── ...
            ├── err/     # 错误输出日志
            │   ├── 000_err.log
            │   ├── 001_err.log
            │   └── ...
            └── viz/     # 可视化数据
                ├── 000/
                ├── 001/
                └── ...
```

## 与原脚本的主要区别

| 特性 | 原脚本 (`start_eval_simlingo.py`) | 新脚本 (`eval_simlingo_local.py`) |
|------|-----------------------------------|-----------------------------------|
| 执行方式 | Slurm 集群并行 | 本地顺序执行 |
| CARLA 服务器 | 每个job单独启动 | 一次启动，重复使用 |
| 并发控制 | 多个job同时运行 | 一次只运行一个路由 |
| 端口管理 | 动态分配多个端口 | 固定端口 |
| 任务重试 | 自动重试失败任务 | 继续下一个任务 |
| 适用场景 | 大规模并行评估 | 本地测试和调试 |

## 快速开始示例

### 1. 快速测试（评估3个路由）

```bash
cd /home/wang/simlingo
python eval_simlingo_local.py --max-routes 3 --seeds 1
```

### 2. 完整评估（所有路由，单个种子）

```bash
cd /home/wang/simlingo
python eval_simlingo_local.py --seeds 1
```

### 3. 完整评估（所有路由，三个种子 - 论文配置）

```bash
cd /home/wang/simlingo
python eval_simlingo_local.py --seeds 1 2 3
```

### 4. 测试单个路由

```bash
cd /home/wang/simlingo
python eval_single_route.py --route /home/wang/simlingo/leaderboard/data/bench2drive_split/bench2drive_00.xml
```

## 注意事项

1. **CARLA 服务器**: 脚本会自动启动 CARLA 服务器，启动后会等待 60 秒。如果你的机器启动较慢，可能需要调整等待时间。

2. **端口占用**: 确保默认端口（2000 和 8000）没有被占用，或使用 `--port` 和 `--tm-port` 指定其他端口。

3. **GPU**: 默认使用 GPU 0，如果需要使用其他 GPU，使用 `--gpu-rank` 参数。

4. **中断恢复**: 如果评估被中断，可以手动删除已完成的路由文件，脚本会跳过已存在结果文件的路由（需要手动实现此功能，或删除结果重新运行）。

5. **内存**: 确保有足够的内存（建议至少 40GB），否则可能导致 CARLA 崩溃。

6. **磁盘空间**: 评估会生成大量数据，确保有足够的磁盘空间。

## 故障排查

### 模型文件是 Git LFS 指针文件（常见问题！）

如果看到错误 `_pickle.UnpicklingError: invalid load key, 'v'`，说明模型文件没有正确下载。

**检查方法：**
```bash
ls -lh /home/wang/simlingo/output/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
```

如果文件大小只有几百字节（如 135B），说明这是 Git LFS 指针文件，需要下载实际文件。

**解决方法：**

1. 安装 Git LFS（如果还没安装）：
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# 或者从源码安装
# 访问 https://git-lfs.github.com/
```

2. 初始化 Git LFS：
```bash
cd /home/wang/simlingo
git lfs install
```

3. 下载 LFS 文件：
```bash
# 下载所有 LFS 文件
git lfs pull

# 或者只下载 output 目录下的文件
git lfs pull --include="output/**"
```

4. 验证文件已下载：
```bash
ls -lh output/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
# 应该显示约 2.4GB 的大小
```

**或者直接下载模型文件：**

如果 Git LFS 不可用，你可以从其他来源手动下载模型文件并放置到正确位置。

### CARLA 无法启动

- 检查 CARLA 路径是否正确：`ls ~/software/carla0915/CarlaUE4.sh`
- 检查端口是否被占用：`lsof -i :2000`
- 尝试手动启动 CARLA 测试：`~/software/carla0915/CarlaUE4.sh -RenderOffScreen`

### 找不到模型文件

- 检查模型路径：`ls /home/wang/simlingo/output/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`
- 使用 `--checkpoint` 参数指定正确的路径

### Python 模块找不到

- 确保已激活 conda 环境：`conda activate simlingo`
- 脚本会自动设置 PYTHONPATH，但如果仍有问题，检查 CARLA 版本是否为 0.9.15

### 评估失败

- 查看错误日志：`eval_results/Bench2Drive/simlingo/bench2drive/1/err/XXX_err.log`
- 查看输出日志：`eval_results/Bench2Drive/simlingo/bench2drive/1/out/XXX_out.log`

### GPU 兼容性问题

如果看到 GPU 兼容性警告（如 `CUDA capability sm_120 is not compatible`），这通常不影响运行，但模型可能会运行在 CPU 上。如需使用 GPU，可能需要更新 PyTorch 版本。

## 获取帮助

```bash
# 查看完整评估脚本的帮助
python eval_simlingo_local.py --help

# 查看单路由评估脚本的帮助
python eval_single_route.py --help
```

## 环境要求

- Python 3.7+
- CARLA 0.9.15
- Conda 环境：simlingo
- 已安装所有依赖包
- GPU 支持

## 下一步

1. 先用 `--max-routes 1` 测试单个路由
2. 确认一切正常后，运行完整评估
3. 查看结果文件中的评估指标
