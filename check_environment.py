#!/usr/bin/env python
"""
验证 PYTHONPATH 和模块导入是否正确配置
"""

import os
import sys

def check_environment():
    """检查环境配置"""
    
    print("=" * 80)
    print("环境检查")
    print("=" * 80)
    
    # 设置环境变量
    carla_root = os.path.expanduser("~/software/carla0915")
    repo_root = "/home/wang/simlingo"
    
    # 重要: Bench2Drive 的 leaderboard 必须在最前面
    pythonpath_parts = [
        f"{repo_root}/Bench2Drive/leaderboard",
        f"{repo_root}/Bench2Drive/scenario_runner",
        f"{carla_root}/PythonAPI/carla",
        f"{carla_root}/PythonAPI/carla/dist/carla-0.9.15-py3.9-linux-x86_64.egg",
    ]
    
    os.environ['CARLA_ROOT'] = carla_root
    os.environ['PYTHONPATH'] = ':'.join(pythonpath_parts)
    os.environ['SCENARIO_RUNNER_ROOT'] = f"{repo_root}/Bench2Drive/scenario_runner"
    
    # 清理 sys.path 并重新添加
    for path in pythonpath_parts:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    print("\n✓ CARLA_ROOT:", os.environ['CARLA_ROOT'])
    print("✓ SCENARIO_RUNNER_ROOT:", os.environ['SCENARIO_RUNNER_ROOT'])
    print("\n✓ PYTHONPATH 设置顺序:")
    for i, path in enumerate(pythonpath_parts, 1):
        print(f"  {i}. {path}")
    
    print("\n" + "=" * 80)
    print("模块导入测试")
    print("=" * 80)
    
    # 测试导入
    try:
        print("\n1. 测试导入 agent_wrapper...")
        from leaderboard.autoagents.agent_wrapper import AgentError, validate_sensor_configuration, TickRuntimeError
        print("   ✓ 成功导入: AgentError, validate_sensor_configuration, TickRuntimeError")
        
        # 检查模块来源
        import leaderboard.autoagents.agent_wrapper as agent_wrapper_module
        print(f"   ✓ 模块路径: {agent_wrapper_module.__file__}")
        
        if "Bench2Drive" in agent_wrapper_module.__file__:
            print("   ✓ 正在使用 Bench2Drive 版本 ✓")
        else:
            print("   ✗ 警告: 没有使用 Bench2Drive 版本！")
            
    except ImportError as e:
        print(f"   ✗ 导入失败: {e}")
        return False
    
    try:
        print("\n2. 测试导入 StatisticsManager...")
        from leaderboard.utils.statistics_manager import StatisticsManager
        print("   ✓ 成功导入 StatisticsManager")
        import leaderboard.utils.statistics_manager as stats_module
        print(f"   ✓ 模块路径: {stats_module.__file__}")
    except ImportError as e:
        print(f"   ✗ 导入失败: {e}")
        return False
    
    try:
        print("\n3. 测试导入 CARLA...")
        import carla
        print(f"   ✓ 成功导入 CARLA")
        print(f"   ✓ CARLA 版本: {carla.Client.get_server_version.__doc__ if hasattr(carla.Client, 'get_server_version') else 'Unknown'}")
    except ImportError as e:
        print(f"   ✗ 导入 CARLA 失败: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✓ 所有检查通过！环境配置正确")
    print("=" * 80)
    return True


if __name__ == '__main__':
    success = check_environment()
    sys.exit(0 if success else 1)
