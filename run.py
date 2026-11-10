"""Run a signalized intersection in SUMO and export per-step queue metrics."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
INBOUND = ("north_in", "south_in", "east_in", "west_in")


def locate_sumo_home() -> Path:
    candidates = []
    if os.environ.get("SUMO_HOME"):
        candidates.append(Path(os.environ["SUMO_HOME"]))
    installed = shutil.which("sumo")
    if installed:
        candidates.append(Path(installed).resolve().parent.parent)
    candidates.append(Path(r"C:\Program Files (x86)\SUMO"))
    for home in candidates:
        if (home / "tools" / "traci").is_dir():
            return home
    raise FileNotFoundError("找不到 SUMO tools；请设置 SUMO_HOME。")


def main() -> None:
    parser = argparse.ArgumentParser(description="SUMO 四向信号灯十字路口仿真")
    parser.add_argument("--gui", action="store_true", help="显示 SUMO GUI")
    parser.add_argument("--steps", type=int, default=1000, help="最多仿真多少秒，默认 1000")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps 必须大于 0")

    home = locate_sumo_home()
    sys.path.insert(0, str(home / "tools"))
    import traci  # type: ignore

    executable = home / "bin" / ("sumo-gui.exe" if args.gui else "sumo.exe")
    if not executable.is_file():
        raise FileNotFoundError(f"找不到 {executable}")
    network = ROOT / "environment" / "network" / "network.net.xml"
    if not network.is_file():
        raise FileNotFoundError("缺少 network.net.xml，请先运行 python build_network.py")

    results = ROOT / "results" / "csv"
    results.mkdir(parents=True, exist_ok=True)
    csv_path = results / "queues.csv"
    command = [str(executable), "-c", str(ROOT / "environment" / "network" / "intersection.sumocfg"),
               "--seed", str(args.seed), "--no-step-log", "true"]
    if args.gui:
        command.extend(["--start", "--quit-on-end"])

    total_arrived = 0
    total_departed = 0
    queue_total = 0
    traci.start(command)
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time_s", "active_vehicles", "departed", "arrived",
                             "north_queue", "south_queue", "east_queue", "west_queue", "total_queue"])
            for _ in range(args.steps):
                traci.simulationStep()
                time_s = int(traci.simulation.getTime())
                departed = traci.simulation.getDepartedNumber()
                arrived = traci.simulation.getArrivedNumber()
                queues = [traci.edge.getLastStepHaltingNumber(edge) for edge in INBOUND]
                queue = sum(queues)
                total_departed += departed
                total_arrived += arrived
                queue_total += queue
                writer.writerow([time_s, traci.vehicle.getIDCount(), departed, arrived,
                                 *queues, queue])
                if time_s % 100 == 0:
                    print(f"t={time_s:4d}s  已进入={total_departed:4d}  已通过={total_arrived:4d}  当前排队={queue:3d}")
                if traci.simulation.getMinExpectedNumber() == 0:
                    break
    finally:
        traci.close()

    print(f"完成：进入 {total_departed} 辆，通过 {total_arrived} 辆。")
    print(f"平均排队车辆数：{queue_total / max(time_s, 1):.2f}")
    print(f"逐秒数据：{csv_path}")


if __name__ == "__main__":
    main()
