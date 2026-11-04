"""Capture the hero GIF, videos and screenshots from the built web app with headless Chromium.

Frames are rendered at fixed simulated time steps (not real time) by driving the app's clock through
``window.__drs``.  Requires ``web/dist`` (``make web``) and ffmpeg.  Software WebGL (SwiftShader) is
used, so rendering is slow but deterministic.

Outputs in ``docs/media/``: ``hero.gif``, ``split_morning_peak.mp4``, ``incident.mp4`` and four PNG
screenshots (aerial golden hour split view, street level, night, heatmap mode).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

from drs.config import REPO_ROOT

MEDIA = REPO_ROOT / "docs" / "media"
BASE = "/dhaka-rl-signals/"
CHROMIUM = os.environ.get("PW_CHROMIUM", "/opt/pw-browsers/chromium")


def start_preview(port: int) -> subprocess.Popen[bytes]:
    proc = subprocess.Popen(
        ["npx", "vite", "preview", "--port", str(port), "--strictPort"], cwd=REPO_ROOT / "web", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(3)
    return proc


def wait_ready(page: Page) -> None:
    page.wait_for_selector("canvas", timeout=120_000)
    page.wait_for_function("() => window.__drs && window.__drs.frames > 5", timeout=120_000)


def set_state(page: Page, **kwargs: object) -> None:
    page.evaluate("(p) => window.__drs.store.getState().set(p)", kwargs)


def set_time(page: Page, t: float) -> None:
    page.evaluate("(t) => { window.__drs.clock.time = t; window.__drs.store.getState().set({ time: t }); }", t)


def pause(page: Page) -> None:
    page.evaluate("() => { window.__drs.clock.playing = false; window.__drs.store.getState().set({ playing: false }); }")


def render_frames(page: Page) -> None:
    before = page.evaluate("() => window.__drs.frames")
    page.wait_for_function(f"() => window.__drs.frames >= {before} + 2", timeout=30_000)


def capture_sequence(page: Page, out_dir: Path, t0: float, seconds: float, fps: int, camera_keys: list[tuple[float, str]]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(seconds * fps)
    sim_speed = 2.0  # simulated seconds per real second in the video
    for i in range(n):
        t = t0 + i / fps * sim_speed
        for at, key in camera_keys:
            if abs(i / fps - at) < 1e-6:
                page.keyboard.press(key)
        set_time(page, t)
        render_frames(page)
        page.screenshot(path=str(out_dir / f"f{i:05d}.png"))
    return n


def encode_video(frames: Path, out: Path, fps: int) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)],
        check=True,
    )


def encode_gif(frames: Path, out: Path, fps: int, width: int, every: int) -> None:
    palette = frames / "palette.png"
    vf = f"select='not(mod(n,{every}))',setpts=N/({fps}*TB),scale={width}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(frames / "f%05d.png"), "-vf", f"{vf},palettegen=max_colors=128", str(palette)], check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(frames / "f%05d.png"), "-i", str(palette), "-lavfi", f"{vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4", "-loop", "0", str(out)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4181)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--hero-seconds", type=float, default=32)
    parser.add_argument("--incident-seconds", type=float, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--only-screenshots", action="store_true")
    args = parser.parse_args()
    MEDIA.mkdir(parents=True, exist_ok=True)
    preview = start_preview(args.port)
    tmp = Path(tempfile.mkdtemp(prefix="drs_media_"))
    report: dict[str, object] = {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                executable_path=CHROMIUM if Path(CHROMIUM).exists() else None,
                args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
            )
            page = browser.new_page(viewport={"width": args.width, "height": args.height}, device_scale_factor=1)
            page.goto(f"http://127.0.0.1:{args.port}{BASE}")
            wait_ready(page)
            set_state(page, explorer=True, scenario="morning_peak")
            page.wait_for_timeout(500)
            wait_ready(page)
            pause(page)
            # screenshots
            set_state(page, camera="aerial", night=False, colourMode="realistic", heatmap=False)
            set_time(page, 1500)
            page.wait_for_timeout(1500)
            render_frames(page)
            page.screenshot(path=str(MEDIA / "aerial_golden_hour_split.png"))
            set_state(page, camera="street")
            page.wait_for_timeout(1500)
            render_frames(page)
            page.screenshot(path=str(MEDIA / "street_level.png"))
            set_state(page, night=True, viewMode="single", activeSide="right")
            page.wait_for_timeout(1500)
            render_frames(page)
            page.screenshot(path=str(MEDIA / "night_bloom.png"))
            set_state(page, night=False, viewMode="split", camera="top", colourMode="heat", heatmap=True)
            page.wait_for_timeout(1500)
            render_frames(page)
            page.screenshot(path=str(MEDIA / "heatmap_mode.png"))
            set_state(page, camera="aerial", colourMode="realistic", heatmap=False)
            page.wait_for_timeout(1500)
            report["screenshots"] = 4
            if not args.only_screenshots:
                t0 = time.time()
                frames = tmp / "hero"
                n = capture_sequence(page, frames, 1200, args.hero_seconds, args.fps, [(0.0, "1"), (args.hero_seconds / 2, "3")])
                encode_video(frames, MEDIA / "split_morning_peak.mp4", args.fps)
                encode_gif(frames, MEDIA / "hero.gif", 10, 720, max(1, args.fps // 10))
                report["hero_frames"] = n
                report["hero_wall_s"] = round(time.time() - t0, 1)
                set_state(page, scenario="incident", camera="aerial")
                page.wait_for_timeout(2000)
                wait_ready(page)
                pause(page)
                inc = tmp / "incident"
                n2 = capture_sequence(page, inc, 1560, args.incident_seconds, args.fps, [])
                encode_video(inc, MEDIA / "incident.mp4", args.fps)
                report["incident_frames"] = n2
            browser.close()
    finally:
        preview.terminate()
        shutil.rmtree(tmp, ignore_errors=True)
    for p in sorted(MEDIA.iterdir()):
        report[p.name] = p.stat().st_size
    (MEDIA / "capture_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
