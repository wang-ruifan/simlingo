#!/usr/bin/env python
"""
测试评估脚本的导入和配置（不启动CARLA）
"""

import os
import sys

def test_imports():
    """测试所有必要的导入"""
    
    # 设置环境
    carla_root = os.path.expanduser("~/software/carla0915")
    repo_root = "/home/wang/simlingo"
    
    pythonpath_parts = [
        f"{repo_root}/Bench2Drive/leaderboard",
        f"{repo_root}/Bench2Drive/scenario_runner",
        f"{carla_root}/PythonAPI/carla",
        f"{carla_root}/PythonAPI/carla/dist/carla-0.9.15-py3.9-linux-x86_64.egg",
    ]
    
    os.environ['CARLA_ROOT'] = carla_root
    os.environ['PYTHONPATH'] = ':'.join(pythonpath_parts)
    os.environ['SCENARIO_RUNNER_ROOT'] = f"{repo_root}/Bench2Drive/scenario_runner"
    
    for path in pythonpath_parts:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    print("测试导入...")
    
    try:
        # 这是 leaderboard_evaluator.py 的关键导入
        from leaderboard.autoagents.agent_wrapper import AgentError, validate_sensor_configuration, TickRuntimeError
        print("✓ agent_wrapper 导入成功")
        
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        print("✓ CarlaDataProvider 导入成功")
        
        from leaderboard.scenarios.scenario_manager import ScenarioManager
        print("✓ ScenarioManager 导入成功")
        
        from leaderboard.utils.statistics_manager import StatisticsManager
        print("✓ StatisticsManager 导入成功")
        
        from leaderboard.utils.route_indexer import RouteIndexer
        print("✓ RouteIndexer 导入成功")
        
        import carla
        print("✓ CARLA 导入成功")
        
        print("\n所有导入测试通过！✓")
        return True
        
    except Exception as e:
        print(f"\n✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_route_file():
    """测试路由文件是否存在"""
    route_file = "/home/wang/simlingo/leaderboard/data/bench2drive_split/bench2drive_00.xml"
    
    print(f"\n检查路由文件: {route_file}")
    if os.path.exists(route_file):
        print(f"✓ 路由文件存在")
        with open(route_file, 'r') as f:
            content = f.read()
            print(f"  文件大小: {len(content)} 字节")
        return True
    else:
        print(f"✗ 路由文件不存在")
        return False


def test_checkpoint():
    """测试模型文件是否存在"""
    checkpoint = "/home/wang/simlingo/output/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
    
    print(f"\n检查模型文件: {checkpoint}")
    if os.path.exists(checkpoint):
        size_mb = os.path.getsize(checkpoint) / (1024 * 1024)
        print(f"✓ 模型文件存在")
        print(f"  文件大小: {size_mb:.2f} MB")
        return True
    else:
        print(f"✗ 模型文件不存在")
        return False


def main():
    print("=" * 80)
    print("SimLingo 评估环境测试")
    print("=" * 80)
    
    all_pass = True
    
    all_pass = test_imports() and all_pass
    all_pass = test_route_file() and all_pass
    all_pass = test_checkpoint() and all_pass
    
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ 所有测试通过！环境配置正确，可以开始评估")
    else:
        print("✗ 部分测试失败，请检查配置")
    print("=" * 80)
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
