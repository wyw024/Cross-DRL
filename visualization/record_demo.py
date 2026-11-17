
"""Record a staged SUMO intersection demonstration.

The scripted signal creates a red-light stop, restores the original
signal program, and records the ego vehicle passing the intersection.
This is an environment demonstration, not a trained RL policy.
"""

from pathlib import Path
import argparse
import sys
import time
import shutil
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.sumo_env import IntersectionEnv
from evaluation.safety_log import SafetyLog


# ------------------------------------------------------------
# Screenshot helper
# ------------------------------------------------------------
def wait_for_screenshot(path, timeout=3.0):
    """Wait until SUMO-GUI finishes writing a screenshot."""
    start = time.time()
    last_size = -1
    stable_count = 0

    while time.time() - start < timeout:
        try:
            if path.exists():
                size = path.stat().st_size

                if size > 0 and size == last_size:
                    stable_count += 1
                    if stable_count >= 2:
                        return True
                else:
                    stable_count = 0
                    last_size = size

        except OSError:
            pass

        time.sleep(0.05)

    return False


def read_image(path):
    """Read an image while supporting Windows paths containing Chinese characters."""
    try:
        image_data = np.fromfile(str(path), dtype=np.uint8)
        if image_data.size == 0:
            return None
        return cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--density', choices=['low', 'medium', 'high', 'extreme'], default='medium')
parser.add_argument('--output', default=str(ROOT / 'results' / 'figures' / 'intersection_demo.mp4'))
args = parser.parse_args()

output = Path(args.output).resolve()
output.parent.mkdir(parents=True, exist_ok=True)

# Use a persistent screenshot directory during recording.
frame_dir = output.parent / 'demo_frames'

if frame_dir.exists():
    shutil.rmtree(frame_dir, ignore_errors=True)

frame_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Video settings
# ------------------------------------------------------------
VIDEO_FPS = 10
FINAL_HOLD_SECONDS = 2
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720


# ------------------------------------------------------------
# Initialize SUMO environment
# ------------------------------------------------------------
env = IntersectionEnv(
    render_mode='human',
    max_steps=160,
    traffic_density=args.density,
    gui_delay_ms=50,
    gui_quit_on_end=True
)

log = SafetyLog(ROOT / 'results' / 'csv' / 'demo_safety.csv')


try:
    _, before = env.reset(seed=42)
    conn = env.connection

    # --------------------------------------------------------
    # GUI configuration
    # --------------------------------------------------------
    views = conn.gui.getIDList()

    if not views:
        raise RuntimeError('SUMO-GUI 没有可用的视图')

    view = views[0]

    conn.gui.trackVehicle(view, 'ego')
    conn.gui.setZoom(view, 800)

    # --------------------------------------------------------
    # Traffic signal configuration
    # --------------------------------------------------------
    tls_ids = conn.trafficlight.getIDList()

    if not tls_ids:
        raise RuntimeError('SUMO 场景中没有交通信号灯')

    tls = tls_ids[0]

    program = conn.trafficlight.getProgram(tls)
    original_state = conn.trafficlight.getRedYellowGreenState(tls)

    conn.trafficlight.setRedYellowGreenState(
        tls,
        'r' * len(original_state)
    )

    # --------------------------------------------------------
    # Simulation and screenshot recording
    # --------------------------------------------------------
    frames = []
    stopped_red = False
    waiting_steps = 0
    resumed = False
    info = before
    final_frame = None

    
    for step in range(150):

        # Restore the original signal program after the ego vehicle stops.
        if waiting_steps >= 4 and not resumed:
            conn.trafficlight.setProgram(tls, program)
            resumed = True
            print('[INFO] Original traffic signal program restored.', flush=True)

        # Request screenshot BEFORE advancing the simulation.
        # SUMO will generate this screenshot at the next simulationStep().
        frame = frame_dir / f'{step:04d}.png'

        try:
            conn.gui.screenshot(
                view,
                str(frame),
                width=VIDEO_WIDTH,
                height=VIDEO_HEIGHT
            )
        except Exception as e:
            raise RuntimeError(f'SUMO-GUI 截图请求失败，step={step + 1}: {e}') from e

        # Advance simulation: this triggers the pending screenshot.
        try:
            _, reward, terminated, truncated, info = env.step(4)
        except Exception as e:
            raise RuntimeError(f'SUMO 仿真中断，step={step + 1}: {e}') from e

        log.record(step + 1, before, info)

        current_signal = info.get('current_signal')
        speed = info.get('speed_m_s', 99)
        passed = info.get('passed_intersection', False)

        print(
            f"step={step + 1}, terminated={terminated}, "
            f"truncated={truncated}, signal={current_signal}, "
            f"speed={speed:.2f}, passed={passed}",
            flush=True
        )

        # Detect a complete stop under the red light.
        if current_signal == 'red' and speed < 0.2:
            stopped_red = True
            waiting_steps += 1

        # The screenshot request has now been processed by simulationStep().
        if wait_for_screenshot(frame, timeout=3.0):
            frames.append((frame, resumed))
            print(f'[INFO] Screenshot saved: {frame.name}', flush=True)

            if passed:
                final_frame = frame
        else:
            print(
                f'[WARNING] Screenshot not found after simulationStep: {frame}',
                flush=True
            )

        before = info

        if terminated or truncated:
            break


    # --------------------------------------------------------
    # Screenshot diagnostics
    # --------------------------------------------------------
    if not frames:
        print('[DEBUG] 截图目录：', frame_dir.resolve())
        print('[DEBUG] 目录是否存在：', frame_dir.exists())
        print(
            '[DEBUG] 目录内容：',
            list(frame_dir.glob('*')) if frame_dir.exists() else []
        )
        raise RuntimeError(
            '没有生成 SUMO-GUI 截图，请检查截图路径和 SUMO-GUI 状态'
        )

    print(f'[INFO] 成功获取截图：{len(frames)} 张', flush=True)

    # --------------------------------------------------------
    # Video generation
    # --------------------------------------------------------
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*'mp4v'),
        VIDEO_FPS,
        (VIDEO_WIDTH, VIDEO_HEIGHT)
    )

    if not writer.isOpened():
        raise RuntimeError('视频编码器无法打开')

    valid_frames = 0
    final_image = None

    try:
        for path, frame_resumed in frames:

            image = read_image(path)

            if image is None:
                print(f'[WARNING] 无法读取截图：{path}', flush=True)
                continue

            # Ensure a consistent image size.
            if image.shape[1] != VIDEO_WIDTH or image.shape[0] != VIDEO_HEIGHT:
                image = cv2.resize(
                    image,
                    (VIDEO_WIDTH, VIDEO_HEIGHT),
                    interpolation=cv2.INTER_AREA
                )

            # Add stage information.
            stage = (
                'Fixed-cycle signal restored'
                if frame_resumed
                else 'Red light: ego approaches and waits'
            )

            cv2.rectangle(
                image,
                (0, 0),
                (VIDEO_WIDTH, 62),
                (20, 20, 20),
                -1
            )

            cv2.putText(
                image,
                stage,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                image,
                'EGO (tracked at center)',
                (20, 700),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 220),
                2
            )

            writer.write(image)
            valid_frames += 1

            # Save the last passing frame.
            if path == final_frame:
                final_image = image.copy()

        # Hold the final passing scene for a few seconds.
        if final_image is not None:
            hold_frames = VIDEO_FPS * FINAL_HOLD_SECONDS

            for _ in range(hold_frames):
                writer.write(final_image)
                valid_frames += 1

    finally:
        writer.release()

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------
    print(
        '\n========== DEMO SUMMARY ==========',
        '\nVIDEO:', output,
        '\nValid frames:', valid_frames,
        '\nFPS:', VIDEO_FPS,
        '\nDuration:', round(valid_frames / VIDEO_FPS, 2), 'seconds',
        '\nWaited at red:', stopped_red,
        '\nSignal restored:', resumed,
        '\nPassed intersection:', info.get('passed_intersection'),
        '\n==================================',
        flush=True
    )

    if valid_frames == 0:
        raise RuntimeError('没有成功读取任何截图，视频无法生成。')

    if not stopped_red or not resumed or not info.get('passed_intersection'):
        raise RuntimeError(
            '演示没有完整出现停车、恢复信号和通过，请检查视频与场景'
        )

finally:
    log.close()
    env.close()

    # Remove screenshots only after SUMO and OpenCV have released their handles.
    time.sleep(0.5)
    shutil.rmtree(frame_dir, ignore_errors=True)
