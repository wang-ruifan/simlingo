#!/usr/bin/env python
"""
本地评估脚本 - 一次只运行一个评估任务
使用本地模型评估 SimLingo 在 Bench2Drive 基准测试上的性能
"""

import os
import sys
import subprocess
import select
import signal
import time
import argparse
from pathlib import Path


class LocalEvaluator:
    """本地评估器 - 顺序执行评估任务"""
    
    def __init__(self, config):
        self.config = config
        self.carla_server = None
        # 标记是否由本脚本启动了 CARLA，只有在为 True 时才停止它
        self.started_carla = False
        
    def setup_environment(self):
        """设置环境变量"""
        carla_root = self.config['carla_root']
        repo_root = self.config['repo_root']
        
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
            
        print(f"Environment variables set:")
        for key, value in env_vars.items():
            print(f"  {key}={value}")
    
    def start_carla_server(self, port, gpu_rank=0):
        """启动 CARLA 服务器"""
        carla_path = self.config['carla_root']
        carla_exec = os.path.join(carla_path, 'CarlaUE4.sh')
        
        if not os.path.exists(carla_exec):
            raise FileNotFoundError(f"CARLA executable not found: {carla_exec}")
        
        cmd = f"{carla_exec} -RenderOffScreen -nosound -carla-rpc-port={port} -graphicsadapter={gpu_rank}"
        print(f"\n启动 CARLA 服务器: {cmd}")
        
        self.carla_server = subprocess.Popen(
            cmd, 
            shell=True, 
            preexec_fn=os.setsid,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 记录本脚本已启动 CARLA
        self.started_carla = True

        print(f"CARLA 服务器 PID: {self.carla_server.pid}")
        print("等待 CARLA 服务器启动 (60秒)...")
        time.sleep(60)
        
        return self.carla_server
    
    def stop_carla_server(self):
        """停止 CARLA 服务器"""
        # 如果脚本未启动 CARLA，且没有要求强制停止，则跳过停止以避免关闭由 leaderboard 或其它进程启动的实例
        force_stop = self.config.get('stop_carla_on_exit', False)
        if not self.started_carla and not force_stop:
            print("跳过停止 CARLA：本脚本未启动 CARLA（或已由其他进程管理）")
            return

        # 如果我们有本地启动的进程句柄，优先基于进程组停止
        if self.carla_server:
            print("\n正在停止本脚本启动的 CARLA 服务器...")
            try:
                os.killpg(self.carla_server.pid, signal.SIGKILL)
                self.carla_server.wait(timeout=10)
            except Exception as e:
                print(f"停止 CARLA 服务器时出错: {e}")
            finally:
                self.carla_server = None
                self.started_carla = False

        # 如果要求强制停止（默认开启），尝试查找并终止系统上由 leaderboard 启动的 CARLA 进程
        if force_stop:
            try:
                self._kill_external_carla_processes()
            except Exception as e:
                print(f"尝试停止外部 CARLA 进程时发生错误: {e}")

    def _kill_external_carla_processes(self):
        """查找并终止系统上与 CARLA 相关的进程（基于 CARLA_ROOT 或可执行名）。
        仅终止与当前用户相同 UID 的进程以减少误杀风险。"""
        carla_root = self.config.get('carla_root', '')
        uid = os.getuid()
        # 使用 ps 输出并筛选
        print("尝试查找并停止系统上的 CARLA 进程...")
        try:
            # 列出所有进程的 pid, uid, cmd
            proc = subprocess.Popen(['ps', '-eo', 'pid,uid,cmd'], stdout=subprocess.PIPE, text=True)
            stdout, _ = proc.communicate(timeout=5)
        except Exception as e:
            print(f"无法列出系统进程: {e}")
            return

        pids_to_kill = []
        for line in stdout.splitlines()[1:]:
            parts = line.strip().split(None, 2)
            if len(parts) < 3:
                continue
            pid_s, uid_s, cmd = parts
            try:
                pid_i = int(pid_s)
                uid_i = int(uid_s)
            except Exception:
                continue
            if uid_i != uid:
                continue
            # 匹配 cmd 中包含 CarlaUE4 或 carla root 路径
            if 'CarlaUE4' in cmd or 'CarlaUE4.sh' in cmd or (carla_root and carla_root in cmd):
                pids_to_kill.append(pid_i)

        if not pids_to_kill:
            print("未发现符合条件的 CARLA 进程")
            return

        print(f"找到 CARLA 进程 PIDs: {pids_to_kill}，将尝试终止...")
        for pid in pids_to_kill:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass

        # 短等待，然后强制 kill 尚未退出的
        time.sleep(2)
        for pid in pids_to_kill:
            try:
                # 检查是否仍存在
                os.kill(pid, 0)
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
            except OSError:
                # 进程已不存在
                pass
    
    def run_single_route(self, route_file, route_id, seed, port, tm_port, output_dir):
        """运行单个路由评估"""
        cfg = self.config
        
        # 创建输出目录
        viz_path = os.path.join(output_dir, "viz", route_id)
        os.makedirs(viz_path, exist_ok=True)
        os.environ['SAVE_PATH'] = viz_path
        
        result_file = os.path.join(output_dir, "res", f"{route_id}_res.json")
        log_file = os.path.join(output_dir, "out", f"{route_id}_out.log")
        err_file = os.path.join(output_dir, "err", f"{route_id}_err.log")
        
        # 构建评估命令
        cmd = [
            'python', '-u',
            f"{cfg['repo_root']}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py",
            f"--routes={route_file}",
            '--repetitions=1',
            '--track=SENSORS',
            f"--checkpoint={result_file}",
            '--timeout=600',
            f"--agent={cfg['agent_file']}",
            f"--agent-config={cfg['checkpoint']}",
            f"--traffic-manager-seed={seed}",
            f"--port={port}",
            f"--traffic-manager-port={tm_port}",
            f"--gpu-rank={cfg.get('gpu_rank', 0)}"
        ]
        
        print(f"\n{'='*80}")
        print(f"运行路由: {route_id}")
        print(f"路由文件: {route_file}")
        print(f"Seed: {seed}")
        print(f"Port: {port}, TM Port: {tm_port}")
        print(f"结果文件: {result_file}")
        print(f"{'='*80}\n")
        
        # 运行评估
        with open(log_file, 'w') as out_f, open(err_file, 'w') as err_f:
            process = subprocess.Popen(
                cmd,
                stdout=out_f,
                stderr=err_f,
                cwd=cfg['repo_root']
            )

            try:
                # 轮询子进程状态，同时监听 stdin 的 Enter 键以跳过当前路由
                returncode = None
                print("（按 Enter 可跳过当前路由并继续下一个）")
                while True:
                    ret = process.poll()
                    if ret is not None:
                        returncode = ret
                        break

                    # 非阻塞检查 stdin 是否有输入（Linux 支持）
                    rlist, _, _ = select.select([sys.stdin], [], [], 1.0)
                    if rlist:
                        line = sys.stdin.readline()
                        # 只要按下 Enter（空行）就跳过当前路由；也接受任何输入行
                        print("检测到输入，跳过当前路由...")
                        try:
                            process.terminate()
                            process.wait(timeout=5)
                        except Exception:
                            try:
                                process.kill()
                                process.wait()
                            except Exception:
                                pass
                        # 使用 -2 表示已被用户跳过
                        returncode = -2
                        break

                if returncode == 0:
                    print(f"✓ 路由 {route_id} 评估完成")
                elif returncode == -2:
                    print(f"→ 路由 {route_id} 已被用户跳过")
                else:
                    print(f"✗ 路由 {route_id} 评估失败 (返回码: {returncode})")
                return returncode
            except KeyboardInterrupt:
                print("\n收到中断信号，正在停止评估...")
                try:
                    process.terminate()
                    process.wait(timeout=10)
                except Exception:
                    pass
                raise
    
    def run_evaluation(self, route_files=None, seeds=None, start_port=2000, start_tm_port=8000):
        """运行完整评估流程"""
        cfg = self.config
        
        # 获取路由文件列表
        if route_files is None:
            route_path = Path(cfg['route_path'])
            if not route_path.exists():
                raise FileNotFoundError(f"路由目录不存在: {route_path}")
            route_files = sorted([str(f) for f in route_path.glob("*.xml")])
        
        if not route_files:
            raise ValueError("没有找到路由文件")
        
        # 使用配置的seeds
        if seeds is None:
            seeds = cfg.get('seeds', [1])
        
        print(f"\n找到 {len(route_files)} 个路由文件")
        print(f"将使用 {len(seeds)} 个种子: {seeds}")
        
        # 设置环境变量
        self.setup_environment()
        
        # 根据配置决定是否由本脚本启动 CARLA 服务器
        try:
            if cfg.get('start_carla', False):
                print("配置要求本脚本启动 CARLA 服务器...")
                self.start_carla_server(start_port, cfg.get('gpu_rank', 0))
            else:
                print("跳过由本脚本启动 CARLA：假定 leaderboard 或其它流程会启动 CARLA。")
            
            # 对每个seed运行所有路由
            for seed in seeds:
                seed_str = str(seed)
                output_dir = os.path.join(
                    cfg['out_root'], 
                    cfg['agent'], 
                    cfg['benchmark'], 
                    seed_str
                )
                
                # 创建输出目录
                for subdir in ['res', 'out', 'err', 'viz']:
                    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
                
                print(f"\n{'#'*80}")
                print(f"# 开始评估 - Seed {seed}")
                print(f"# 输出目录: {output_dir}")
                print(f"{'#'*80}\n")
                
                # 运行每个路由
                for idx, route_file in enumerate(route_files, 1):
                    route_filename = os.path.basename(route_file)
                    route_id = route_filename.split("_")[-1].replace(".xml", "").zfill(3)
                    
                    print(f"\n进度: [{idx}/{len(route_files)}]")
                    
                    try:
                        self.run_single_route(
                            route_file=route_file,
                            route_id=route_id,
                            seed=seed_str,
                            port=start_port,
                            tm_port=start_tm_port,
                            output_dir=output_dir
                        )
                    except KeyboardInterrupt:
                        print("\n评估被用户中断")
                        raise
                    except Exception as e:
                        print(f"✗ 路由 {route_id} 评估出错: {e}")
                        import traceback
                        traceback.print_exc()
                        # 继续下一个路由
                        continue
                    
                    # 每个路由之间短暂休息
                    time.sleep(2)
                
                print(f"\n完成 Seed {seed} 的所有评估")
            
            print(f"\n{'='*80}")
            print("所有评估任务完成！")
            print(f"{'='*80}\n")
            
        finally:
            self.stop_carla_server()


def main():
    parser = argparse.ArgumentParser(
        description="SimLingo 本地评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行所有路由，使用默认配置
  python eval_simlingo_local.py
  
  # 只运行前3个路由
  python eval_simlingo_local.py --max-routes 3
  
  # 使用特定seed
  python eval_simlingo_local.py --seeds 1 2
  
  # 指定端口
  python eval_simlingo_local.py --port 3000 --tm-port 9000
        """
    )
    
    parser.add_argument('--checkpoint', type=str,
                       default='/home/wang/simlingo/output/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt',
                       help='模型checkpoint路径')
    parser.add_argument('--route-path', type=str,
                       default='/home/wang/simlingo/leaderboard/data/bench2drive_split',
                       help='路由文件目录')
    parser.add_argument('--output-dir', type=str,
                       default='/home/wang/simlingo/eval_results/Bench2Drive',
                       help='评估结果输出目录')
    parser.add_argument('--carla-root', type=str,
                       default=os.path.expanduser('~/software/carla0915'),
                       help='CARLA安装目录')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[1],
                       help='评估使用的随机种子')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA服务器端口')
    parser.add_argument('--tm-port', type=int, default=8000,
                       help='Traffic Manager端口')
    parser.add_argument('--gpu-rank', type=int, default=0,
                       help='GPU编号')
    parser.add_argument('--max-routes', type=int, default=None,
                       help='最大评估路由数量（用于测试）')
    parser.add_argument('--start-carla', action='store_true', default=False,
                       help='由本脚本在开始时启动 CARLA（默认: False），若 leaderboard 会自动启动 CARLA 请不要使用此选项')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--stop-carla-on-exit', dest='stop_carla_on_exit', action='store_true',
                       help='在脚本退出或收到 Ctrl+C 时尝试停止 CARLA（默认：开启，可能会关闭由其它流程启动的 CARLA 实例）')
    group.add_argument('--no-stop-carla-on-exit', dest='stop_carla_on_exit', action='store_false',
                       help='不要在退出时停止 CARLA（禁用强制停止）')
    parser.set_defaults(stop_carla_on_exit=True)
    
    args = parser.parse_args()
    
    # 构建配置
    config = {
        'agent': 'simlingo',
        'checkpoint': args.checkpoint,
        'benchmark': 'bench2drive',
        'route_path': args.route_path,
        'seeds': args.seeds,
        'out_root': args.output_dir,
        'carla_root': args.carla_root,
        'repo_root': '/home/wang/simlingo',
        'agent_file': '/home/wang/simlingo/team_code/agent_simlingo.py',
        'gpu_rank': args.gpu_rank,
        'start_carla': args.start_carla,
        'stop_carla_on_exit': args.stop_carla_on_exit,
    }
    
    # 验证路径
    if not os.path.exists(config['checkpoint']):
        print(f"错误: 找不到模型文件: {config['checkpoint']}")
        sys.exit(1)
    
    if not os.path.exists(config['route_path']):
        print(f"错误: 找不到路由目录: {config['route_path']}")
        sys.exit(1)
    
    if not os.path.exists(config['carla_root']):
        print(f"错误: 找不到CARLA目录: {config['carla_root']}")
        sys.exit(1)
    
    if not os.path.exists(config['agent_file']):
        print(f"错误: 找不到agent文件: {config['agent_file']}")
        sys.exit(1)
    
    print("SimLingo 本地评估脚本")
    print("=" * 80)
    print(f"模型: {config['checkpoint']}")
    print(f"路由目录: {config['route_path']}")
    print(f"输出目录: {config['out_root']}")
    print(f"CARLA: {config['carla_root']}")
    print(f"Seeds: {config['seeds']}")
    print(f"端口: {args.port} (CARLA), {args.tm_port} (TM)")
    print("=" * 80)
    
    # 获取路由文件
    route_files = sorted(Path(config['route_path']).glob("*.xml"))
    route_files = [str(f) for f in route_files]
    
    if args.max_routes:
        route_files = route_files[:args.max_routes]
        print(f"\n注意: 限制评估路由数量为 {args.max_routes}")
    
    # 运行评估
    evaluator = LocalEvaluator(config)
    # 注册 SIGINT 处理器：在 Ctrl+C 时优先尝试清理 CARLA（若配置允许）然后退出
    def _handle_sigint(sig, frame):
        try:
            print("\n收到 SIGINT，尝试清理并退出...")
            evaluator.stop_carla_server()
        except Exception as e:
            print(f"清理 CARLA 时出错: {e}")
        # 退出主进程
        sys.exit(1)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        evaluator.run_evaluation(
            route_files=route_files,
            seeds=args.seeds,
            start_port=args.port,
            start_tm_port=args.tm_port
        )
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
