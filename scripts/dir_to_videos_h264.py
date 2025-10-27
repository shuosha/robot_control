import os, glob, argparse
import cv2
import imageio
from typing import List, Tuple


def parse_args():
    p = argparse.ArgumentParser("Make VS Code previewable MP4s (H.264 yuv420p) from episode folders")
    p.add_argument("-d", "--data_dir", required=True, help="Root containing episode_*")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--pattern", default="*.jpg", help="Image glob inside camera_*/rgb/")
    return p.parse_args()


def list_imgs(dir_path: str, pattern: str) -> List[str]:
    files = glob.glob(os.path.join(dir_path, pattern))
    return sorted(files)


def pad_to_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    top = (target_h - h) // 2
    bottom = target_h - h - top
    return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def write_mp4_h264(path: str, fps: int, size: Tuple[int, int], frames_bgr_iter):
    """Write H.264 (libx264) + yuv420p using imageio (legacy-safe API)."""
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None
    )
    for bgr in frames_bgr_iter:
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb)
    writer.close()


def find_cameras(ep_dir: str) -> List[str]:
    """Return list of camera_* dirs (that have rgb/) sorted by numeric index."""
    cand = [d for d in glob.glob(os.path.join(ep_dir, "camera_*")) if os.path.isdir(d)]
    cams = []
    for d in cand:
        if os.path.isdir(os.path.join(d, "rgb")):
            cams.append(d)

    def idx(cam_path: str) -> int:
        name = os.path.basename(cam_path)
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return 1_000_000  # push non-numeric to the end (deterministic)

    cams.sort(key=idx)
    return cams


def main():
    a = parse_args()
    episodes = sorted(d for d in glob.glob(os.path.join(a.data_dir, "episode_*")) if os.path.isdir(d))
    if not episodes:
        print(f"No episodes under {a.data_dir}")
        return

    for ep in episodes:
        ep_name = os.path.basename(ep)
        cams = find_cameras(ep)

        if len(cams) == 0:
            raise RuntimeError(f"[error] {ep_name}: found no camera_* folders with rgb/")
        if len(cams) > 2:
            raise RuntimeError(f"[error] {ep_name}: found >2 camera_* folders ({len(cams)}).")

        # ---------- Single-camera ----------
        if len(cams) == 1:
            rgb_dir = os.path.join(cams[0], "rgb")
            imgs = list_imgs(rgb_dir, a.pattern)
            if not imgs:
                print(f"[skip] {ep_name}: no frames in {rgb_dir}")
                continue

            first = cv2.imread(imgs[0])
            if first is None:
                print(f"[skip] {ep_name}: bad first frame in {rgb_dir}")
                continue

            H, W = first.shape[:2]
            out_path = os.path.join(ep, f"vis_{ep_name}_{os.path.basename(cams[0])}.mp4")

            def gen():
                for p in imgs:
                    im = cv2.imread(p)
                    if im is None:
                        continue
                    if im.shape[:2] != (H, W):
                        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
                    yield im

            write_mp4_h264(out_path, a.fps, (W, H), gen())
            print(f"[ok] {out_path}")
            continue

        # ---------- Two-camera (side-by-side) ----------
        rgb_a, rgb_b = os.path.join(cams[0], "rgb"), os.path.join(cams[1], "rgb")
        A, B = list_imgs(rgb_a, a.pattern), list_imgs(rgb_b, a.pattern)
        if not A or not B:
            print(f"[skip] {ep_name}: empty camera dirs ({rgb_a} or {rgb_b})")
            continue

        n = min(len(A), len(B))
        A, B = A[:n], B[:n]
        fa, fb = cv2.imread(A[0]), cv2.imread(B[0])
        if fa is None or fb is None:
            print(f"[skip] {ep_name}: bad first frames in {rgb_a} or {rgb_b}")
            continue

        target_h = max(fa.shape[0], fb.shape[0])
        W_out = fa.shape[1] + fb.shape[1]
        H_out = target_h
        out_path = os.path.join(ep, f"vis_{ep_name}_{os.path.basename(cams[0])}_{os.path.basename(cams[1])}.mp4")

        def gen2():
            for pa, pb in zip(A, B):
                ia, ib = cv2.imread(pa), cv2.imread(pb)
                if ia is None or ib is None:
                    continue
                ia = pad_to_height(ia, target_h)
                ib = pad_to_height(ib, target_h)
                yield cv2.hconcat([ia, ib])

        write_mp4_h264(out_path, a.fps, (W_out, H_out), gen2())
        print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
