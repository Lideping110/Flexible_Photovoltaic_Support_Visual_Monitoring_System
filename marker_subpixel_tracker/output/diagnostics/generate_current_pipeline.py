# -*- coding: utf-8 -*-
"""Step-by-step diagnostics of the CURRENT project pipeline + far-target fix test.

Mirrors marker_subpixel_tracker/src/main.py / pipeline.py exactly:
ByteTrack -> TargetLock -> crop ROI -> grayscale -> structure-tensor
lambda_min -> anchor-window peak selection -> cornerSubPix -> map back.

The actual selection logic is imported from the PROJECT (features.detect_marker_center
and pipeline.localize_one) so this script verifies the ported fix, not a copy.

Two run modes:
  * default (no --video): single-frame visual diagnostic. Steps 1..7 are drawn
    for the chosen image (--image, default id1_cal00_00_original.jpg). A video
    path also works: frame 30 is read in memory (no new file is written, the
    sandbox only allows updating existing outputs).
  * --video PATH: quantitative test on a far clip. Each frame runs the NEW
    proportional-window + disambiguation + jump-gate selection (via
    localize_one) next to the OLD fixed-20px selection (old_select), and reports
    inter-frame jump statistics (the metric that exposed the 9m flip bug).
"""
import argparse
import cv2
import numpy as np
import yaml
from pathlib import Path

from marker_subpixel_tracker.src.tracker import ByteTrackTracker, TargetLock
from marker_subpixel_tracker.src.detector import crop_box
from marker_subpixel_tracker.src.features import (
    structure_tensor_lambda_min, detect_marker_center, refine_subpixel,
    predict_anchor, local_peaks,
)
from marker_subpixel_tracker.src.pipeline import localize_one

OUT = Path("D:/摄像头拉流/marker_subpixel_tracker/output/diagnostics")
OUT.mkdir(parents=True, exist_ok=True)
VIDEO = "D:/摄像头拉流/babiao/babiao_undistort.mp4"
IMAGES = "D:/摄像头拉流/marker_subpixel_tracker/output/diagnostics/id1_cal00_00_original.jpg"
WEIGHTS = "D:/摄像头拉流/marker26s_det.pt"

cfg = yaml.safe_load(open("D:/摄像头拉流/marker_subpixel_tracker/config/config.yaml",
                          encoding="utf-8"))
model = cfg["model"]
tracking = cfg["tracking"]
loc = cfg["localization"]
SIGMA = float(loc.get("structure_sigma", 1.5))
THRESH = float(loc.get("min_lambda_min", 5000.0))
# proportional-window params (mirror config; used for the visual window only)
WIN_RATIO = float(loc.get("win_ratio", 0.25))
WIN_MAX = float(loc.get("win_max", 20))
SUBWIN = int(loc.get("subpixel_window", 5))


def old_select(response, anchor, radius=20, min_lambda=THRESH):
    """Reference: the OLD fixed-20px window argmax (no disambiguation, no gate)."""
    h, w = response.shape
    px, py = float(anchor[0]), float(anchor[1])
    x1, x2 = max(0, int(px) - radius), min(w, int(px) + radius + 1)
    y1, y2 = max(0, int(py) - radius), min(h, int(py) + radius + 1)
    if x2 <= x1 or y2 <= y1:
        return None, None
    win = response[y1:y2, x1:x2]
    if float(np.max(win)) < min_lambda:
        return None, None
    wy, wx = np.unravel_index(int(np.argmax(win)), win.shape)
    return wx + x1, wy + y1


# ============================ helpers ============================
def save(name, img):
    cv2.imwrite(str(OUT / name), img)
    print("saved", name, img.shape)


def canvas_title(img, title):
    h, w = img.shape[:2]
    canvas = np.full((h + 64, w, 3), 34, np.uint8)
    canvas[64:, :, :] = img
    cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 235, 255), 2, cv2.LINE_AA)
    return canvas


def to_heatmap(response):
    v = np.nan_to_num(response.astype(np.float32), nan=0.0)
    lo = float(np.percentile(v, 2))
    hi = float(np.percentile(v, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    v = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    v8 = (v * 255).astype(np.uint8)
    return cv2.applyColorMap(v8, cv2.COLORMAP_JET)


# ============================ single-frame visual ============================
def run_single(image_path):
    frame = cv2.imread(str(image_path))
    if frame is None:
        cap = cv2.VideoCapture(str(image_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise SystemExit(f"cannot read image/video: {image_path}")

    tracker = ByteTrackTracker(WEIGHTS, model["confidence"], model["class_id"],
                               model["device"], tracker=tracking["tracker"],
                               persist=tracking["persist"])
    tracks = tracker.update(frame)
    lock = TargetLock(target_id=tracking["target_id"],
                      reacquire_after=tracking["reacquire_after_frames"])
    track = lock.select(tracks, 0)

    s1 = frame.copy()
    x1, y1, x2, y2 = np.round(track["box"]).astype(int)
    cv2.rectangle(s1, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(s1, f"step1  ByteTrack box  id={track['id']} conf={track['conf']:.2f}",
                (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)
    s1 = cv2.resize(s1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    save("cur_01_input_frame.jpg", s1)

    box = np.asarray(track["box"], dtype=float)
    roi, (ox, oy) = crop_box(frame, box)
    s2 = canvas_title(cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC),
                      f"step2  crop ROI (no expand)  {roi.shape[1]}x{roi.shape[0]}  offset=({ox},{oy})")
    save("cur_02_roi_crop.jpg", s2)

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    s3 = canvas_title(cv2.cvtColor(cv2.resize(roi_gray, None, fx=4, fy=4,
                                              interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR),
                      "step3  grayscale")
    save("cur_03_grayscale.jpg", s3)

    response = structure_tensor_lambda_min(roi_gray, sigma=SIGMA)
    h, w = response.shape
    sbox = min(w, h)
    radius = int(min(WIN_MAX, max(3, WIN_RATIO * sbox)))  # NEW proportional radius
    hm = to_heatmap(response)
    hm_big = cv2.resize(hm, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    peaks = local_peaks(response, min_val=1000.0, min_dist=5)
    for i, (v, px, py) in enumerate(peaks[:8]):
        color = (0, 0, 255) if i == 0 else (255, 255, 255)
        cv2.circle(hm_big, (px * 4, py * 4), 10, color, 1)
        cv2.putText(hm_big, f"{v:.0f}", (px * 4 + 6, py * 4 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    s4 = canvas_title(hm_big, "step4  lambda_min response heatmap")
    save("cur_04_lambda_response.jpg", s4)

    # step5 NEW anchor-window peak selection (via PROJECT detect_marker_center)
    anchor = np.array([w / 2.0, h / 2.0], dtype=np.float32)  # first frame = ROI center
    loc_center, loc_lambda, loc_cont = detect_marker_center(roi_gray, loc, anchor)
    sel_cx, sel_cy = float(loc_center[0]), float(loc_center[1])
    s5 = cv2.resize(hm, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.drawMarker(s5, (int(anchor[0] * 4), int(anchor[1] * 4)), (0, 255, 255),
                   cv2.MARKER_CROSS, 22, 2)
    x1w, x2w = max(0, int(anchor[0]) - radius), min(w, int(anchor[0]) + radius + 1)
    y1w, y2w = max(0, int(anchor[1]) - radius), min(h, int(anchor[1]) + radius + 1)
    cv2.rectangle(s5, (x1w * 4, y1w * 4), (x2w * 4 - 4, y2w * 4 - 4),
                  (0, 255, 255), 1)
    cv2.circle(s5, (int(sel_cx * 4), int(sel_cy * 4)), 11, (0, 255, 0), 2)
    if peaks:
        gx, gy = peaks[0][1], peaks[0][2]
        if not (x1w <= gx < x2w and y1w <= gy < y2w):
            cv2.circle(s5, (gx * 4, gy * 4), 11, (0, 0, 255), 2)
            cv2.putText(s5, "global max (outside window)",
                        (gx * 4 + 8, gy * 4 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255), 1, cv2.LINE_AA)
    tag = "step5  NEW proportional window (%.0f%%*sbox=%dpx, max %d): pick peak" % (
        WIN_RATIO * 100, radius, WIN_MAX)
    s5 = canvas_title(s5, tag)
    save("cur_05_anchor_window.jpg", s5)

    # step6 subpixel refinement (already done inside detect_marker_center)
    sub = np.array([sel_cx, sel_cy], dtype=np.float32)
    shift = 0.0  # subpixel refinement is internal; show the resolved point
    zoom = 8
    half = 14
    ix1, iy1 = max(0, int(sel_cx) - half), max(0, int(sel_cy) - half)
    ix2, iy2 = min(w, int(sel_cx) + half), min(h, int(sel_cy) + half)
    crop_g = roi_gray[iy1:iy2, ix1:ix2]
    crop_big = cv2.resize(crop_g, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    crop_big = cv2.cvtColor(crop_big, cv2.COLOR_GRAY2BGR)
    cv2.drawMarker(crop_big, (int((sel_cx - ix1) * zoom), int((sel_cy - iy1) * zoom)),
                   (0, 255, 255), cv2.MARKER_CROSS, 14, 1)
    cv2.circle(crop_big, (int((sub[0] - ix1) * zoom), int((sub[1] - iy1) * zoom)),
               4, (0, 0, 255), -1)
    cv2.drawMarker(crop_big, (int((sub[0] - ix1) * zoom), int((sub[1] - iy1) * zoom)),
                   (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    s6 = canvas_title(crop_big,
                      f"step6  selected center  ({sel_cx:.2f},{sel_cy:.2f})  "
                      f"continuous={loc_cont}")
    save("cur_06_subpixel.jpg", s6)

    # step7 map back to full frame
    center = np.array([sel_cx, sel_cy], dtype=np.float32) + np.array([ox, oy], dtype=np.float32)
    s7 = frame.copy()
    pxg, pyg = int(round(center[0])), int(round(center[1]))
    cv2.drawMarker(s7, (pxg, pyg), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
    cv2.circle(s7, (pxg, pyg), 6, (0, 0, 255), 2)
    cv2.putText(s7, f"center = ({center[0]:.2f}, {center[1]:.2f})",
                (pxg + 14, pyg - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2, cv2.LINE_AA)
    s7 = cv2.resize(s7, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    save("cur_07_final_center.jpg", s7)

    print("\n=== RESULT (PROJECT detect_marker_center) ===")
    print(f"ROI size            : {w}x{h}  (short side sbox={sbox})")
    print(f"window radius       : {radius}px  (= {WIN_RATIO}*{sbox}, capped at {WIN_MAX})")
    print(f"selected center     : ({sel_cx:.1f},{sel_cy:.1f}) ROI-local")
    print(f"continuous          : {loc_cont}  lambda_min={loc_lambda:.0f}")
    print(f"top-5 lambda peaks (global):")
    for v, px, py in peaks[:5]:
        print(f"  {v:9.0f}  at ({px},{py})")


# ============================ video jump test ============================
def run_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")
    tracker = ByteTrackTracker(WEIGHTS, model["confidence"], model["class_id"],
                               model["device"], tracker=tracking["tracker"],
                               persist=tracking["persist"])
    lock = TargetLock(target_id=tracking["target_id"],
                      reacquire_after=tracking["reacquire_after_frames"])

    prev_box = prev_center = None
    prev_old = prev_new = None
    old_jumps, new_jumps = [], []
    centers_new, centers_old = [], []
    n, ok = 0, True
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        tracks = tracker.update(frame)
        track = lock.select(tracks, 0)
        if track is None:
            continue
        box = np.asarray(track["box"], dtype=np.float32)
        # NEW (project): proportional window + disambiguation + jump gate
        center, lam, cont = localize_one(frame, box, loc, prev_center, prev_box)
        if center is not None:
            centers_new.append((float(center[0]), float(center[1])))
        # OLD reference: fixed-20px window argmax (no disambiguation, no gate)
        roi, (ox, oy) = crop_box(frame, box)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        response = structure_tensor_lambda_min(roi_gray, sigma=SIGMA)
        rw, rh = roi_gray.shape[1], roi_gray.shape[0]
        if prev_box is not None and prev_center is not None:
            anchor_g = predict_anchor(prev_box, prev_center, box)
            anchor = anchor_g - np.array([ox, oy], dtype=np.float32)
        else:
            anchor = np.array([rw / 2.0, rh / 2.0], dtype=np.float32)
        ocx, ocy = old_select(response, anchor, radius=20)

        if center is not None and prev_new is not None:
            new_jumps.append(np.hypot(center[0] - prev_new[0], center[1] - prev_new[1]))
        if ocx is not None and prev_old is not None:
            old_jumps.append(np.hypot(ocx - prev_old[0], ocy - prev_old[1]))
        if ocx is not None:
            centers_old.append((float(ocx), float(ocy)))

        prev_box = box
        prev_center = center if center is not None else prev_center
        prev_old = (ocx, ocy) if ocx is not None else prev_old
        prev_new = center
    cap.release()

    def stat(j):
        if not j:
            return "n/a"
        arr = np.array(j)
        big = int((arr > 0.5).sum())
        return (f"n={len(arr)}  max={arr.max():.2f}px  mean={arr.mean():.3f}px  "
                f"jumps>0.5px={big}  (RMS={np.sqrt((arr**2).mean()):.3f}px)")
    print("\n=== RESULT (video jump test, via PROJECT localize_one) ===")
    print(f"frames processed    : {n}")
    print(f"OLD  fixed-20px     : {stat(old_jumps)}")
    print(f"NEW  proportional+gate: {stat(new_jumps)}")
    def span(c):
        if len(c) < 2:
            return "n/a (too few points)"
        a = np.array(c)
        return (f"x span={a[:,0].max()-a[:,0].min():.1f}px  "
                f"y span={a[:,1].max()-a[:,1].min():.1f}px  "
                f"(follow amplitude; ~0 => stuck, large => tracking)")
    print(f"OLD center follow   : {span(centers_old)}")
    print(f"NEW center follow   : {span(centers_new)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=IMAGES, help="single-frame visual diagnostic image")
    ap.add_argument("--video", default=None, help="run quantitative far-clip jump test")
    args = ap.parse_args()
    if args.video:
        run_video(args.video)
    else:
        run_single(args.image)


if __name__ == "__main__":
    main()
