#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, math, re, glob
from typing import List, Optional, Tuple, Iterable

import cv2
import numpy as np
import imageio.v3 as iio  # imageio v3 writer supports libx264 (yuv420p)
import imageio

# ---------------------------- helpers ----------------------------

def _videos_dir(root: Path) -> Path:
    d = root / "videos"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _natsorted_jpgs(rgb_dir: Path, pattern: str = "*.jpg") -> List[Path]:
    # Simple natural-ish sort: rely on zero-padded names or numeric stems
    paths = [Path(p) for p in glob.glob(str(rgb_dir / pattern))]
    def key(p: Path):
        m = re.findall(r"\d+", p.stem)
        return int(m[-1]) if m else p.name
    return sorted(paths, key=key)

def _list_episodes(root: Path) -> List[Path]:
    eps = [p for p in root.glob("episode_*") if p.is_dir()]
    def key(p: Path):
        m = re.findall(r"\d+", p.name)
        return int(m[0]) if m else 10**9
    return sorted(eps, key=key)

def _list_cameras(ep_dir: Path) -> List[int]:
    cams = []
    for d in ep_dir.glob("camera_*"):
        if (d / "rgb").is_dir():
            try:
                cams.append(int(d.name.split("_", 1)[1]))
            except Exception:
                pass
    return sorted(cams)

def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --------------------- 1) single videos (H.264) ---------------------

def make_single_videos(root: Path, fps: int = 30, cam_idx: Optional[int] = None,
                       pattern: str = "*.jpg", eps_idx: Optional[int] = None) -> List[Tuple[int, Path]]:
    """
    Create H.264 per-episode videos:
      videos/eps_{04d}_cam_{idx}.mp4
    Skips files that already exist. Returns list of (cam_idx, video_path).
    """
    videos = _videos_dir(root)
    outputs: List[Tuple[int, Path]] = []
    episodes = _list_episodes(root)
    if not episodes:
        print("[warn] no episode_* found")
        return outputs

    # discover all cameras (respect cam_idx filter)
    cam_set = set()
    for ep in episodes:
        for c in _list_cameras(ep):
            if cam_idx is None or c == cam_idx:
                cam_set.add(c)
    cams = sorted(cam_set)
    if not cams:
        print(f"[warn] no cameras found (filter cam_idx={cam_idx})")
        return outputs

    if eps_idx is not None:
        ep_name = f"episode_{eps_idx:04d}"
        episodes = [ep for ep in episodes if ep.name == ep_name]

    for ep in episodes:

        m = re.findall(r"\d+", ep.name)
        ep_num = int(m[0]) if m else 0
        for c in cams:
            rgb_dir = ep / f"camera_{c}" / "rgb"
            if not rgb_dir.is_dir():
                continue
            out_mp4 = videos / f"eps_{ep_num:04d}_cam_{c}.mp4"
            if out_mp4.exists():
                outputs.append((c, out_mp4))
                continue

            jpgs = _natsorted_jpgs(rgb_dir, pattern)
            if not jpgs:
                continue

            first = cv2.imread(str(jpgs[0]))
            if first is None:
                continue
            H, W = first.shape[:2]

            def frames() -> Iterable[np.ndarray]:
                for p in jpgs:
                    img = cv2.imread(str(p))
                    if img is None:
                        continue
                    if img.shape[:2] != (H, W):
                        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    yield _bgr_to_rgb(img)

            # H.264 + yuv420p for Windows/macOS preview compatibility
            with imageio.get_writer(
                str(out_mp4),
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=None,   # avoid forced resizing to 16x blocks
            ) as w:
                for f in frames():       # f must be RGB (H,W,3) uint8
                    w.append_data(f)
                    
            print(f"[ok] {out_mp4}")
            outputs.append((c, out_mp4))
    return outputs

# --------- 2) collage videos (on-the-fly from JPGs, with scale) ---------

def make_collage_videos(root: Path, fps: int = 30, cols: int = 7,
                        scale: float = 0.25, pattern: str = "*.jpg") -> None:
    """
    For each camera index:
      - Gather episodes with camera_{idx}/rgb/*.jpg
      - Downscale by 'scale'
      - Grid (cols x rows)
      - Shorter episodes loop their last frame so all end together
      - Output: videos/collage_cam_{idx}.mp4
    """
    videos = _videos_dir(root)
    episodes = _list_episodes(root)
    if not episodes:
        print("[warn] no episode_* found")
        return

    cam_set = set()
    for ep in episodes:
        cam_set.update(_list_cameras(ep))
    cams = sorted(cam_set)
    if not cams:
        print("[warn] no cameras found for collage")
        return

    for c in cams:
        out_mp4 = videos / f"collage_cam_{c}.mp4"
        if out_mp4.exists():
            print(f"[skip] {out_mp4} (exists)")
            continue

        # episode -> jpgs (for this camera)
        ep_jpgs: List[List[Path]] = []
        for ep in episodes:
            rgb_dir = ep / f"camera_{c}" / "rgb"
            if rgb_dir.is_dir():
                lst = _natsorted_jpgs(rgb_dir, pattern)
                if lst:
                    ep_jpgs.append(lst)
        if not ep_jpgs:
            continue

        # Determine cell size from the first readable frame
        first_frame = None
        for lst in ep_jpgs:
            f0 = cv2.imread(str(lst[0]))
            if f0 is not None:
                if scale != 1.0:
                    f0 = cv2.resize(f0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                first_frame = _bgr_to_rgb(f0)
                break
        if first_frame is None:
            print(f"[skip] camera {c}: no readable frames")
            continue

        H, W = first_frame.shape[:2]
        rows = math.ceil(len(ep_jpgs) / cols)
        canvas_h, canvas_w = rows * H, cols * W

        lens = [len(lst) for lst in ep_jpgs]
        max_len = max(lens)

        def collage_frames():
            for t in range(max_len):
                canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                for k, n in enumerate(lens):
                    r, col = divmod(k, cols)
                    y0, y1 = r * H, (r + 1) * H
                    x0, x1 = col * W, (col + 1) * W
                    idx = min(t, n - 1)  # loop last frame once episode ends
                    p = ep_jpgs[k][idx]
                    img = cv2.imread(str(p))
                    if img is None:
                        continue
                    if scale != 1.0:
                        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    if img.shape[:2] != (H, W):
                        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    canvas[y0:y1, x0:x1] = _bgr_to_rgb(img)
                yield canvas

        with imageio.get_writer(
            str(out_mp4),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        ) as w:
            for f in collage_frames():
                w.append_data(f)

        print(f"[ok] {out_mp4}")


# ------------------------------- CLI -------------------------------

def main():
    ap = argparse.ArgumentParser("Make per-episode singles and per-camera collage videos (H.264)")
    ap.add_argument("root", help="Path to root/ containing episode_XXXX/")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--cam-idx", type=int, default=None, help="For singles: only this camera index")
    ap.add_argument("--cols", type=int, default=3, help="Collage columns")
    ap.add_argument("--scale", type=float, default=1.0, help="Collage downscale factor")
    ap.add_argument("--collage", action="store_true", help="Only make collage videos")
    ap.add_argument("--single", action="store_true", help="Only make single videos")
    ap.add_argument("--eps_idx", type=int, default=None, help="(not used) For singles: only this episode index")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    assert (args.single or args.collage) and args.single != args.collage, "Please specify --single or --collage"

    if args.single:
        make_single_videos(root, fps=args.fps, cam_idx=args.cam_idx, eps_idx=args.eps_idx)

    if args.collage:
        make_collage_videos(root, fps=args.fps, cols=args.cols, scale=args.scale)

if __name__ == "__main__":
    main()
