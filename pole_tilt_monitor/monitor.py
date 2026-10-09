import csv
import os
from collections import defaultdict, deque

import numpy as np

from .geometry import (
    load_calibration, load_intrinsics, pixel_ray_world,
    ray_plane_intersection, top_point_from_length,
    tilt_from_vector, displacement_from_vector
)
from .vision import (clean_mask, skeleton_variants,
                    ordered_skeleton_points, local_centerline_points,
                    line_endpoints)


class EMA:
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        x = np.asarray(x, dtype=float)
        if self.value is None:
            self.value = x.copy()
        else:
            self.value = self.alpha*x + (1-self.alpha)*self.value
        return self.value.copy()


class PoleMonitor:
    def __init__(self, calib_path, pole_length, ground_z,
                 endpoint_ratio=0.10,
                 ransac_residual_px=2.5,
                 ransac_iterations=300,
                 baseline_frames=60,
                 ema_alpha=0.25,
                 quality_threshold=0.60,
                 quality_weights=None,
                 min_ray_plane_angle_deg=5.0,
                 max_centerline_residual_px=4.0,
                 max_endpoint_spread_px=4.0,
                 max_length_error_ratio=0.03,
                 body_trim_ratio=0.10):

        self.K, self.dist, self.R, self.t, self.Cw = load_calibration(calib_path)
        self.L = float(pole_length)
        self.ground_z = float(ground_z)
        self.endpoint_ratio = endpoint_ratio
        self.ransac_residual_px = ransac_residual_px
        self.ransac_iterations = ransac_iterations
        self.baseline_frames = int(baseline_frames)
        self.quality_threshold = float(quality_threshold)
        self.quality_weights = quality_weights or {
            "inlier_ratio": 0.30, "centerline_residual": 0.25,
            "top_stability": 0.12, "bottom_stability": 0.12,
            "ray_ground_angle": 0.10, "length_constraint": 0.11,
        }
        self.min_ray_plane_angle_deg = float(min_ray_plane_angle_deg)
        self.max_centerline_residual_px = float(max_centerline_residual_px)
        self.max_endpoint_spread_px = float(max_endpoint_spread_px)
        self.max_length_error_ratio = float(max_length_error_ratio)
        self.body_trim_ratio = float(body_trim_ratio)

        self.history = defaultdict(lambda: deque(maxlen=self.baseline_frames))
        self.baseline = {}
        self.ema = defaultdict(lambda: EMA(ema_alpha))
        self.endpoint_history = defaultdict(lambda: deque(maxlen=10))
        self.line_history = defaultdict(lambda: deque(maxlen=self.baseline_frames))
        # Video frame that best represents the final baseline line.
        self.baseline_frame_id = {}
        self.baseline_quality = {}

    def _quality(self, ep, rb, V):
        fit = ep["fit"]
        inlier = float(np.clip(fit.get("inlier_ratio", 0.0), 0, 1))
        residual = float(fit.get("residual_rms", np.inf))
        residual_score = float(np.exp(-max(0.0, residual) /
                                      max(self.max_centerline_residual_px, 1e-6)))
        top_score = float(np.exp(-ep.get("top_spread_px", np.inf) /
                                 max(self.max_endpoint_spread_px, 1e-6)))
        bottom_score = float(np.exp(-ep.get("bottom_spread_px", np.inf) /
                                    max(self.max_endpoint_spread_px, 1e-6)))
        # Angle between ray and ground plane; near-parallel rays are unstable.
        ray_plane_angle = np.degrees(np.arcsin(np.clip(abs(float(rb[2])), 0, 1)))
        ray_score = float(np.clip(ray_plane_angle /
                                  max(self.min_ray_plane_angle_deg, 1e-6), 0, 1))
        length_error = abs(float(np.linalg.norm(V) - self.L)) / max(self.L, 1e-9)
        length_score = float(np.exp(-length_error /
                                    max(self.max_length_error_ratio, 1e-6)))
        components = {
            "ransac_inlier_ratio": inlier,
            "centerline_residual_score": residual_score,
            "centerline_residual_px": residual,
            "top_point_stability": top_score,
            "bottom_point_stability": bottom_score,
            "top_endpoint_deviation_px": float(
                ep.get("top_endpoint_deviation_px", np.inf)),
            "bottom_endpoint_deviation_px": float(
                ep.get("bottom_endpoint_deviation_px", np.inf)),
            "skeleton_mask_height_coverage": float(
                ep.get("skeleton_mask_height_coverage", 0.0)),
            "ray_ground_angle_deg": float(ray_plane_angle),
            "ray_ground_score": ray_score,
            "top_length_constraint_error_ratio": float(length_error),
            "top_length_constraint_score": length_score,
        }
        vals = {
            "inlier_ratio": inlier, "centerline_residual": residual_score,
            "top_stability": top_score, "bottom_stability": bottom_score,
            "ray_ground_angle": ray_score, "length_constraint": length_score,
        }
        total_w = sum(float(self.quality_weights.get(k, 0.0)) for k in vals)
        quality = (sum(float(self.quality_weights.get(k, 0.0)) * v
                       for k, v in vals.items()) / max(total_w, 1e-9))
        return float(np.clip(quality, 0, 1)), components

    def measure(self, mask):
        mask = clean_mask(mask)
        variants = skeleton_variants(mask)
        candidates = []
        if variants["thinning"] is not None:
            candidates.append(("thinning", variants["thinning"]))
        candidates.append(("fallback", variants["fallback"]))
        for method, skel in candidates:
            result = self._measure_with_skeleton(mask, skel)
            if result is not None:
                result["skeleton_method"] = method
                return result
        return None

    def _measure_with_skeleton(self, mask, skel):
        # Use the ordered longest skeleton path as the geometric observation.
        # Mask centers may supplement a missing end, but never refit the shaft.
        pts = ordered_skeleton_points(skel)
        if len(pts) < 20:
            return None
        mask_centerline = local_centerline_points(mask, skel)

        ep = line_endpoints(
            pts,
            endpoint_ratio=self.endpoint_ratio,
            residual_px=self.ransac_residual_px,
            iterations=self.ransac_iterations,
            body_trim_ratio=self.body_trim_ratio,
            mask_points=mask_centerline
        )
        if ep is None:
            return None

        top_uv = ep["top"]
        bottom_uv = ep["bottom"]

        rt = pixel_ray_world(
            top_uv[0], top_uv[1], self.K, self.dist, self.R
        )
        rb = pixel_ray_world(
            bottom_uv[0], bottom_uv[1], self.K, self.dist, self.R
        )

        # 地面/支撑平面：Z=ground_z
        Pb, lambda_b = ray_plane_intersection(
            self.Cw, rb,
            np.array([0,0,1.0]),
            -self.ground_z
        )
        if Pb is None:
            return None

        Pt, lambda_t = top_point_from_length(
            self.Cw, rt, Pb, self.L
        )
        if Pt is None:
            return None

        V = Pt-Pb
        quality, quality_components = self._quality(ep, rb, V)
        lr, fb, total = tilt_from_vector(V)
        dx, dy, dh = displacement_from_vector(V)

        return {
            "top": top_uv,
            "bottom": bottom_uv,
            "Pb": Pb, "Pt": Pt, "V": V,
            "lr": lr, "fb": fb, "total": total,
            "dx": dx, "dy": dy, "dh": dh,
            "lambda_b": lambda_b, "lambda_t": lambda_t
            ,"quality": quality, "quality_components": quality_components,
            "valid": bool(quality >= self.quality_threshold),
            "status": "OK" if quality >= self.quality_threshold else "INVALID",
        }

    def process(self, pole_id, mask, frame_id=None):
        m = self.measure(mask)
        if m is None:
            return None
        m["frame_id"] = frame_id

        # Temporal endpoint stability is evaluated per tracked pole.  It is
        # deliberately part of the gate so a suddenly jumping endpoint is
        # reported as INVALID instead of becoming a precise-looking outlier.
        hist = self.endpoint_history[pole_id]
        hist.append(np.r_[m["top"], m["bottom"]])
        if len(hist) >= 3:
            spread = float(np.mean(np.std(np.asarray(hist), axis=0)))
            temporal_score = float(np.exp(-spread /
                                          max(self.max_endpoint_spread_px, 1e-6)))
            qc = m["quality_components"]
            qc["temporal_endpoint_spread_px"] = spread
            qc["top_point_stability"] = min(qc["top_point_stability"], temporal_score)
            qc["bottom_point_stability"] = min(qc["bottom_point_stability"], temporal_score)
            # Recompute weighted score after temporal evidence is available.
            vals = {"inlier_ratio": qc["ransac_inlier_ratio"],
                    "centerline_residual": qc["centerline_residual_score"],
                    "top_stability": qc["top_point_stability"],
                    "bottom_stability": qc["bottom_point_stability"],
                    "ray_ground_angle": qc["ray_ground_score"],
                    "length_constraint": qc["top_length_constraint_score"]}
            weights = self.quality_weights
            m["quality"] = float(np.clip(sum(weights.get(k, 0.0)*v for k, v in vals.items()) /
                                          max(sum(weights.get(k, 0.0) for k in vals), 1e-9), 0, 1))
            m["valid"] = m["quality"] >= self.quality_threshold
            m["status"] = "OK" if m["valid"] else "INVALID"

        # Invalid geometry is observable to callers but never contaminates
        # baseline or temporal filtering.
        if not m.get("valid", False):
            m.update({"baseline_ready": pole_id in self.baseline,
                      "delta_lr": float("nan"), "delta_fb": float("nan"),
                      "delta_total": float("nan"), "delta_dx": float("nan"),
                      "delta_dy": float("nan"), "delta_dh": float("nan")})
            return m

        current = np.array([
            m["lr"],m["fb"],m["total"],
            m["dx"],m["dy"],m["dh"]
        ], dtype=float)

        # 每根立柱单独建立基准
        if pole_id not in self.baseline:
            self.history[pole_id].append(current)
            if len(self.history[pole_id]) >= self.baseline_frames:
                self.baseline[pole_id] = np.median(
                    np.asarray(self.history[pole_id]), axis=0
                )
            delta = np.zeros(6)
            ready = pole_id in self.baseline
        else:
            delta = current - self.baseline[pole_id]
            ready = True

        # 对角度变化做EMA
        smooth = self.ema[pole_id].update(delta[:3])

        m.update({
            "delta_lr": float(smooth[0]),
            "delta_fb": float(smooth[1]),
            "delta_total": float(smooth[2]),
            "delta_dx": float(delta[3]),
            "delta_dy": float(delta[4]),
            "delta_dh": float(delta[5]),
            "baseline_ready": ready
        })
        return m


class ImageRelativePoleMonitor(PoleMonitor):
    """
    Extrinsic-free monitor using undistorted image/camera coordinates.

    This mode measures apparent pole direction in the fixed camera frame and
    compares it with a per-track baseline.  It is useful for a first version
    when field extrinsics are unavailable, but it is not a world-frame 3D
    tilt measurement and cannot recover front/back tilt independently.
    """
    def __init__(self, calib_path, endpoint_ratio=0.10,
                 ransac_residual_px=2.5, ransac_iterations=300,
                 baseline_frames=1, ema_alpha=0.25,
                 quality_threshold=0.60, quality_weights=None,
                 max_centerline_residual_px=4.0,
                 max_endpoint_spread_px=4.0,
                 body_trim_ratio=0.10):
        self.K, self.dist = load_intrinsics(calib_path)
        self.endpoint_ratio = endpoint_ratio
        self.ransac_residual_px = ransac_residual_px
        self.ransac_iterations = ransac_iterations
        self.baseline_frames = int(baseline_frames)
        self.quality_threshold = float(quality_threshold)
        image_weights = quality_weights or {
            "inlier_ratio": 0.40, "centerline_residual": 0.30,
            "top_stability": 0.15, "bottom_stability": 0.15,
            "ray_ground_angle": 0.0, "length_constraint": 0.0,
        }
        # World-only evidence must never inflate an extrinsic-free score.
        self.quality_weights = dict(image_weights)
        self.quality_weights["ray_ground_angle"] = 0.0
        self.quality_weights["length_constraint"] = 0.0
        self.max_centerline_residual_px = float(max_centerline_residual_px)
        self.max_endpoint_spread_px = float(max_endpoint_spread_px)
        self.body_trim_ratio = float(body_trim_ratio)
        self.history = defaultdict(lambda: deque(maxlen=self.baseline_frames))
        self.baseline = {}
        self.ema = defaultdict(lambda: EMA(ema_alpha))
        self.endpoint_history = defaultdict(lambda: deque(maxlen=10))
        self.line_history = defaultdict(lambda: deque(maxlen=self.baseline_frames))
        # Frame id that best represents the final baseline line.
        self.baseline_frame_id = {}
        self.baseline_quality = {}

    def measure(self, mask):
        mask = clean_mask(mask)
        variants = skeleton_variants(mask)
        candidates = []
        if variants["thinning"] is not None:
            candidates.append(("thinning", variants["thinning"]))
        candidates.append(("fallback", variants["fallback"]))
        for method, skel in candidates:
            result = self._measure_with_skeleton(mask, skel)
            if result is not None:
                result["skeleton_method"] = method
                return result
        return None

    def _measure_with_skeleton(self, mask, skel):
        pts = ordered_skeleton_points(skel)
        if len(pts) < 20:
            return None
        mask_centerline = local_centerline_points(mask, skel)
        ep = line_endpoints(pts, endpoint_ratio=self.endpoint_ratio,
                            residual_px=self.ransac_residual_px,
                            iterations=self.ransac_iterations,
                            body_trim_ratio=self.body_trim_ratio,
                            mask_points=mask_centerline)
        if ep is None:
            return None
        # The full frame has already been undistorted before inference. Keep
        # the fitted endpoints in this rectified pixel coordinate system.
        top_xy = np.asarray(ep["top"], dtype=np.float64)
        bot_xy = np.asarray(ep["bottom"], dtype=np.float64)
        dv = top_xy - bot_xy
        vertical = -float(dv[1])
        if vertical <= 1e-8:
            return None
        lr = float(np.degrees(np.arctan2(float(dv[0]), vertical)))
        total = abs(lr)
        fit = ep["fit"]
        residual = float(fit.get("residual_rms", np.inf))
        qc = {
            "ransac_inlier_ratio": float(fit.get("inlier_ratio", 0.0)),
            "centerline_residual_px": residual,
            "centerline_residual_score": float(np.exp(-residual / max(self.max_centerline_residual_px, 1e-6))),
            "top_point_stability": float(np.exp(-ep.get("top_spread_px", np.inf) / max(self.max_endpoint_spread_px, 1e-6))),
            "bottom_point_stability": float(np.exp(-ep.get("bottom_spread_px", np.inf) / max(self.max_endpoint_spread_px, 1e-6))),
            "top_endpoint_deviation_px": float(ep.get("top_endpoint_deviation_px", np.inf)),
            "bottom_endpoint_deviation_px": float(ep.get("bottom_endpoint_deviation_px", np.inf)),
            "skeleton_mask_height_coverage": float(ep.get("skeleton_mask_height_coverage", 0.0)),
            # Not observable without extrinsics; excluded by default weights.
            "ray_ground_angle_deg": float("nan"), "ray_ground_score": 1.0,
            "top_length_constraint_error_ratio": float("nan"),
            "top_length_constraint_score": 1.0,
        }
        vals = {"inlier_ratio": qc["ransac_inlier_ratio"],
                "centerline_residual": qc["centerline_residual_score"],
                "top_stability": qc["top_point_stability"],
                "bottom_stability": qc["bottom_point_stability"],
                "ray_ground_angle": 1.0, "length_constraint": 1.0}
        weights = self.quality_weights
        quality = sum(weights.get(k, 0.0) * v for k, v in vals.items()) / max(sum(weights.get(k, 0.0) for k in vals), 1e-9)
        return {
            "top": ep["top"], "bottom": ep["bottom"],
            "Pb": np.full(3, np.nan), "Pt": np.full(3, np.nan),
            "V": np.array([dv[0], dv[1], 0.0]),
            "lr": lr, "fb": 0.0, "total": total,
            "dx": float(dv[0]), "dy": float(dv[1]), "dh": abs(float(dv[0])),
            "lambda_b": float("nan"), "lambda_t": float("nan"),
            "quality": float(np.clip(quality, 0, 1)),
            "quality_components": qc,
            "valid": bool(quality >= self.quality_threshold),
            "status": "OK" if quality >= self.quality_threshold else "INVALID",
            "measurement_mode": "image_relative",
            "line_angle_deg": lr,
        }

    def process(self, pole_id, mask, frame_id=None):
        m = self.measure(mask)
        if m is None:
            return None
        m["frame_id"] = frame_id

        # Temporal endpoint stability is evaluated before accepting a frame.
        hist = self.endpoint_history[pole_id]
        hist.append(np.r_[m["top"], m["bottom"]])
        if len(hist) >= 3:
            spread = float(np.mean(np.std(np.asarray(hist), axis=0)))
            temporal_score = float(np.exp(-spread / max(self.max_endpoint_spread_px, 1e-6)))
            qc = m["quality_components"]
            qc["temporal_endpoint_spread_px"] = spread
            qc["top_point_stability"] = min(qc["top_point_stability"], temporal_score)
            qc["bottom_point_stability"] = min(qc["bottom_point_stability"], temporal_score)
            vals = {"inlier_ratio": qc["ransac_inlier_ratio"],
                    "centerline_residual": qc["centerline_residual_score"],
                    "top_stability": qc["top_point_stability"],
                    "bottom_stability": qc["bottom_point_stability"],
                    "ray_ground_angle": 1.0, "length_constraint": 1.0}
            weights = self.quality_weights
            m["quality"] = float(np.clip(sum(weights.get(k, 0.0)*v for k, v in vals.items()) /
                                          max(sum(weights.get(k, 0.0) for k in vals), 1e-9), 0, 1))
            m["valid"] = m["quality"] >= self.quality_threshold
            m["status"] = "OK" if m["valid"] else "INVALID"

        if not m.get("valid", False):
            m.update({"baseline_ready": pole_id in self.baseline,
                      "delta_lr": float("nan"), "delta_fb": float("nan"),
                      "delta_total": float("nan"), "delta_dx": float("nan"),
                      "delta_dy": float("nan"), "delta_dh": float("nan")})
            return m

        # Build a stable reference line from the first valid frame(s). The
        # baseline bottom remains fixed thereafter, preventing bottom jitter
        # from being interpreted as pole motion.
        line = np.r_[m["top"], m["bottom"]]
        if pole_id not in self.baseline:
            frame_id = m.get("frame_id")
            self.line_history[pole_id].append({"frame_id": frame_id,
                                                "line": line.copy(),
                                                "quality": float(m["quality"])})
            if len(self.line_history[pole_id]) >= self.baseline_frames:
                records = list(self.line_history[pole_id])
                lines = np.asarray([r["line"] for r in records])
                b = np.median(lines, axis=0)
                bvec = b[:2] - b[2:4]
                bang = float(np.degrees(np.arctan2(bvec[0], -bvec[1])))
                self.baseline[pole_id] = np.r_[b, bang]
                nearest = int(np.argmin(np.linalg.norm(lines - b, axis=1)))
                self.baseline_frame_id[pole_id] = records[nearest]["frame_id"]
                self.baseline_quality[pole_id] = records[nearest]["quality"]
            ready = pole_id in self.baseline
            delta = np.zeros(6)
        else:
            b = self.baseline[pole_id]
            fixed_bottom = b[2:4]
            vec = m["top"] - fixed_bottom
            current_angle = float(np.degrees(np.arctan2(vec[0], -vec[1])))
            baseline_angle = float(b[4])
            delta = np.array([current_angle - baseline_angle, 0.0,
                              abs(current_angle) - abs(baseline_angle),
                              vec[0] - (b[0] - b[2]),
                              vec[1] - (b[1] - b[3]), 0.0])
            ready = True

        smooth = self.ema[pole_id].update(delta[:3])
        m.update({"lr": float(m["lr"]), "fb": 0.0,
                  "total": abs(float(m["lr"])),
                  "delta_lr": float(smooth[0]), "delta_fb": float(smooth[1]),
                  "delta_total": float(smooth[2]), "delta_dx": float(delta[3]),
                  "delta_dy": float(delta[4]), "delta_dh": float(delta[5]),
                  "baseline_ready": ready, "fixed_bottom":
                  self.baseline[pole_id][2:4].copy() if ready else None})
        return m


def create_csv(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "w", newline="", encoding="utf-8-sig")
    w = csv.writer(f)
    w.writerow([
        "frame","time_s","pole_id",
        "lr_deg","fb_deg","total_deg",
        "delta_lr_deg","delta_fb_deg","delta_total_deg",
        "dx_m","dy_m","horizontal_displacement_m",
        "delta_dx_m","delta_dy_m","delta_horizontal_displacement_m",
        "Pb_x","Pb_y","Pb_z","Pt_x","Pt_y","Pt_z",
        "quality","status","ransac_inlier_ratio","centerline_residual_px",
        "top_point_stability","bottom_point_stability","ray_ground_angle_deg",
        "top_length_constraint_error_ratio","temporal_endpoint_spread_px"
    ])
    return f,w


def write_csv(w, frame_id, fps, pole_id, m):
    Pb,Pt = m["Pb"],m["Pt"]
    w.writerow([
        frame_id,
        frame_id/fps if fps>0 else 0,
        pole_id,
        m["lr"],m["fb"],m["total"],
        m["delta_lr"],m["delta_fb"],m["delta_total"],
        m["dx"],m["dy"],m["dh"],
        m["delta_dx"],m["delta_dy"],m["delta_dh"],
        Pb[0],Pb[1],Pb[2],Pt[0],Pt[1],Pt[2],
        m.get("quality", float("nan")), m.get("status", "INVALID"),
        m.get("quality_components", {}).get("ransac_inlier_ratio", float("nan")),
        m.get("quality_components", {}).get("centerline_residual_px", float("nan")),
        m.get("quality_components", {}).get("top_point_stability", float("nan")),
        m.get("quality_components", {}).get("bottom_point_stability", float("nan")),
        m.get("quality_components", {}).get("ray_ground_angle_deg", float("nan")),
        m.get("quality_components", {}).get("top_length_constraint_error_ratio", float("nan")),
        m.get("quality_components", {}).get("temporal_endpoint_spread_px", float("nan"))
    ])
