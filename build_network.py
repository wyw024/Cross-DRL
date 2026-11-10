"""Build the four-arm traffic-light network from editable XML source files."""

from pathlib import Path
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent / "environment" / "network"


def find_netconvert() -> str:
    found = shutil.which("netconvert")
    if found:
        return found
    candidates = []
    if os.environ.get("SUMO_HOME"):
        candidates.append(Path(os.environ["SUMO_HOME"]) / "bin" / "netconvert.exe")
    candidates.append(Path(r"C:\Program Files (x86)\SUMO\bin\netconvert.exe"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("找不到 netconvert；请确认 SUMO 已安装并设置 SUMO_HOME。")


def main() -> None:
    command = [
        find_netconvert(),
        "--node-files", str(ROOT / "nodes.nod.xml"),
        "--edge-files", str(ROOT / "edges.edg.xml"),
        "--output-file", str(ROOT / "network.net.xml"),
        "--tls.default-type", "static",
        "--tls.cycle.time", "90",
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    print(f"路网已生成：{ROOT / 'network.net.xml'}")


if __name__ == "__main__":
    main()
