import os
import sys
import shutil
import subprocess

print("=" * 50)
print("SUMO 环境检查")
print("=" * 50)

# 1. 检查 Python
print("Python:", sys.version)

# 2. 检查 SUMO_HOME
sumo_home = os.environ.get("SUMO_HOME")

print("SUMO_HOME:", sumo_home)

if not sumo_home:
    raise RuntimeError("未配置 SUMO_HOME")

# 3. 检查 SUMO 程序
sumo_binary = shutil.which("sumo")
gui_binary = shutil.which("sumo-gui")
netedit_binary = shutil.which("netedit")

print("sumo:", sumo_binary)
print("sumo-gui:", gui_binary)
print("netedit:", netedit_binary)

if not sumo_binary:
    raise RuntimeError("找不到 sumo.exe")

if not gui_binary:
    raise RuntimeError("找不到 sumo-gui.exe")

# 4. 检查 TraCI
try:
    import traci
    print("TraCI 导入成功")
except Exception as e:
    raise RuntimeError(f"TraCI 导入失败: {e}")

# 5. 检查 SUMO 版本
result = subprocess.run(
    [sumo_binary, "--version"],
    capture_output=True,
    text=True
)

print(result.stdout)

print("=" * 50)
print("环境检查完成")
print("=" * 50)
