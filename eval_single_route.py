#!/usr/bin/env python
"""
单路由快速评估脚本
用于快速测试单个路由的评估
"""

import os
import sys
import subprocess
import signal
import time
import argparse


def setup_environment(carla_root, repo_root):
    """设置环境变量"""
    # 重要: Bench2Drive 的 leaderboard 必须在最前面，以确保使用正确的版本
    pythonpath_parts = [
        repo_root,  # 添加repo根目录，以便导入 simlingo_training 等模块
        f"{repo_root}/Bench2Drive/leaderboard",
        f"{repo_root}/Bench2Drive/scenario_runner",
        f"{carla_root}/PythonAPI/carla",
        f"{carla_root}/PythonAPI/carla/dist/carla-0.9.15-py3.9-linux-x86_64.egg",
    ]
    
    # 添加现有的 PYTHONPATH（如果有）
    existing_pythonpath = os.environ.get('PYTHONPATH', '')
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    
    env_vars = {
        'CARLA_ROOT': carla_root,
        'PYTHONPATH': ':'.join(pythonpath_parts),
        'SCENARIO_RUNNER_ROOT': f"{repo_root}/Bench2Drive/scenario_runner",
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("环境变量已设置")


def start_carla_server(carla_root, port, gpu_rank=0):
    """启动 CARLA 服务器"""
    carla_exec = os.path.join(carla_root, 'CarlaUE4.sh')
    
    if not os.path.exists(carla_exec):
        raise FileNotFoundError(f"CARLA executable not found: {carla_exec}")
    
    cmd = f"{carla_exec} -nosound -carla-rpc-port={port} -graphicsadapter={gpu_rank}"
    print(f"\n启动 CARLA 服务器...")
    print(f"命令: {cmd}")
    
    server = subprocess.Popen(
        cmd, 
        shell=True, 
        preexec_fn=os.setsid
    )
    
    print(f"CARLA 服务器 PID: {server.pid}")
    print("等待 CARLA 服务器启动 (60秒)...")
    time.sleep(60)
    
    return server


def stop_carla_server(server):
    """停止 CARLA 服务器"""
    if server:
        print("\n正在停止 CARLA 服务器...")
        try:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=10)
            print("CARLA 服务器已停止")
        except Exception as e:
            print(f"停止 CARLA 服务器时出错: {e}")


def run_evaluation(args):
    """运行评估"""
    # 设置环境
    setup_environment(args.carla_root, args.repo_root)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    viz_path = os.path.join(args.output_dir, "viz")
    os.makedirs(viz_path, exist_ok=True)
    os.environ['SAVE_PATH'] = viz_path
    
    result_file = os.path.join(args.output_dir, "result.json")
    
    # 启动 CARLA 服务器
    server = None
    try:
        # 启动逻辑说明:
        # - 默认不由此脚本启动 CARLA，因为 leaderboard_evaluator 可能会自行启动一个实例。
        # - 如果用户显式传入 --start-server，则由本脚本启动 CARLA。
        if args.no_server and getattr(args, 'start_server', False):
            print("警告: 同时指定了 --no-server 和 --start-server，已忽略 --no-server（优先使用 --start-server）")

        if getattr(args, 'start_server', False):
            server = start_carla_server(args.carla_root, args.port, args.gpu_rank)
        
        # 构建评估命令
        cmd = [
            'python', '-u',
            f"{args.repo_root}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py",
            f"--routes={args.route}",
            '--repetitions=1',
            '--track=SENSORS',
            f"--checkpoint={result_file}",
            f"--timeout={args.timeout}",
            f"--agent={args.agent_file}",
            f"--agent-config={args.checkpoint}",
            f"--traffic-manager-seed={args.seed}",
            f"--port={args.port}",
            f"--traffic-manager-port={args.tm_port}",
            f"--gpu-rank={args.gpu_rank}"
        ]
        
        print(f"\n{'='*80}")
        print("开始评估")
        print(f"路由文件: {args.route}")
        print(f"模型: {args.checkpoint}")
        print(f"结果文件: {result_file}")
        print(f"Seed: {args.seed}")
        print(f"{'='*80}\n")
        
        # 运行评估
        process = subprocess.run(cmd, cwd=args.repo_root)
        
        if process.returncode == 0:
            print(f"\n✓ 评估完成")
            print(f"结果保存在: {result_file}")
        else:
            print(f"\n✗ 评估失败 (返回码: {process.returncode})")
        
        return process.returncode
        
    finally:
        # 仅当本脚本启动了 CARLA 时，才停止它
        if server and getattr(args, 'start_server', False):
            stop_carla_server(server)


def main():
    parser = argparse.ArgumentParser(
        description="SimLingo 单路由评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估单个路由（自动启动CARLA）
  python eval_single_route.py --route /path/to/route.xml
  
  # 使用已运行的CARLA服务器
  python eval_single_route.py --route /path/to/route.xml --no-server --port 2000
        """
    )
    
    parser.add_argument('--route', type=str, required=True,
                       help='路由XML文件路径')
    parser.add_argument('--checkpoint', type=str,
                       default='/home/wang/simlingo/output/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt',
                       help='模型checkpoint路径')
    parser.add_argument('--output-dir', type=str,
                       default='/home/wang/simlingo/eval_results/single_route',
                       help='评估结果输出目录')
    parser.add_argument('--carla-root', type=str,
                       default=os.path.expanduser('~/software/carla0915'),
                       help='CARLA安装目录')
    parser.add_argument('--repo-root', type=str,
                       default='/home/wang/simlingo',
                       help='代码仓库根目录')
    parser.add_argument('--agent-file', type=str,
                       default='/home/wang/simlingo/team_code/agent_simlingo.py',
                       help='Agent文件路径')
    parser.add_argument('--seed', type=int, default=1,
                       help='随机种子')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA服务器端口')
    parser.add_argument('--tm-port', type=int, default=8000,
                       help='Traffic Manager端口')
    parser.add_argument('--gpu-rank', type=int, default=0,
                       help='GPU编号')
    parser.add_argument('--timeout', type=float, default=600.0,
                       help='超时时间（秒）')
    parser.add_argument('--no-server', action='store_true',
                       help='不启动CARLA服务器（假设已在运行）')
    parser.add_argument('--start-server', action='store_true',
                       help='由此脚本在调用 leaderboard_evaluator 之前启动 CARLA（默认 False，通常让 leaderboard_evaluator 启动以避免重复）')
    
    args = parser.parse_args()
    
    # 验证路径
    if not os.path.exists(args.route):
        print(f"错误: 找不到路由文件: {args.route}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"错误: 找不到模型文件: {args.checkpoint}")
        sys.exit(1)

    # 只有当用户显式要求由本脚本启动 CARLA 时，才验证 CARLA 路径
    if getattr(args, 'start_server', False):
        if not os.path.exists(args.carla_root):
            print(f"错误: 找不到CARLA目录: {args.carla_root}")
            sys.exit(1)
    
    if not os.path.exists(args.agent_file):
        print(f"错误: 找不到agent文件: {args.agent_file}")
        sys.exit(1)
    
    try:
        returncode = run_evaluation(args)
        sys.exit(returncode)
    except KeyboardInterrupt:
        print("\n\n评估被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
