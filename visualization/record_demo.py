"""Record only the SUMO viewport. A scripted signal creates a red-light wait.

This is a staged environment demonstration, not a trained RL policy.
"""
from pathlib import Path
import argparse
import sys
import tempfile
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from environment.sumo_env import IntersectionEnv
from evaluation.safety_log import SafetyLog

parser = argparse.ArgumentParser()
parser.add_argument('--density', choices=['low', 'medium', 'high'], default='medium')
parser.add_argument('--output', default=str(ROOT/'results'/'figures'/'intersection_demo.mp4'))
args = parser.parse_args()
output = Path(args.output).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
env = IntersectionEnv(render_mode='human', max_steps=160, traffic_density=args.density,
                      gui_delay_ms=50, gui_quit_on_end=True)
log = SafetyLog(ROOT/'results'/'csv'/'demo_safety.csv')
try:
    _, before = env.reset(seed=42)
    conn = env.connection
    views = conn.gui.getIDList()
    if not views:
        raise RuntimeError('SUMO-GUI 没有可用的视图')
    view = views[0]
    conn.gui.trackVehicle(view, 'ego')
    conn.gui.setZoom(view, 800)
    # Demonstration-only setup: hold the whole intersection red, then return
    # to its original fixed-cycle signal program. Safety checks remain enabled.
    tls = conn.trafficlight.getIDList()[0]
    program = conn.trafficlight.getProgram(tls)
    conn.trafficlight.setRedYellowGreenState(tls, 'r' * len(conn.trafficlight.getRedYellowGreenState(tls)))
    with tempfile.TemporaryDirectory(prefix='sumo_demo_') as folder:
        folder = Path(folder)
        frames = []
        stopped_red = False
        waiting_steps = 0
        resumed = False
        for step in range(150):
            if waiting_steps >= 4 and not resumed:
                conn.trafficlight.setProgram(tls, program)
                resumed = True
            frame = folder / f'{step:04d}.png'
            conn.gui.screenshot(view, str(frame), width=1280, height=720)
            _, reward, terminated, truncated, info = env.step(4)
            log.record(step+1, before, info)
            if frame.exists(): frames.append((frame, info, resumed))
            if (info.get('signal') or '').lower() == 'r' and info.get('speed_m_s', 99) < 0.2:
                stopped_red = True
                waiting_steps += 1
            before = info
            if terminated or truncated: break
        if not frames:
            raise RuntimeError('没有生成 SUMO-GUI 截图')
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), 2, (1280,720))
        if not writer.isOpened(): raise RuntimeError('视频编码器无法打开')
        try:
            for path, info, resumed in frames:
                image = cv2.imread(str(path))
                if image is None: continue
                stage = 'Fixed-cycle signal restored' if resumed else 'Red light: ego approaches and waits'
                cv2.rectangle(image, (0,0), (1280,62), (20,20,20), -1)
                cv2.putText(image, stage, (20,40), cv2.FONT_HERSHEY_SIMPLEX, .9, (255,255,255), 2)
                cv2.putText(image, 'EGO (tracked at center)', (20,700), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,220), 2)
                writer.write(image)
        finally: writer.release()
    print('VIDEO', output, 'frames',len(frames),'waited_at_red',stopped_red,
          'signal_restored',resumed,'passed',info['passed_intersection'],flush=True)
    if not stopped_red or not resumed or not info['passed_intersection']:
        raise RuntimeError('演示没有完整出现停车、恢复信号和通过，请检查视频与场景')
finally:
    log.close()
    env.close()
