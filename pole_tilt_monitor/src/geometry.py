import numpy as np
import cv2
import yaml


def load_calibration(path):
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    K = np.asarray(d["K"], dtype=np.float64)
    dist = np.asarray(d["dist"], dtype=np.float64).reshape(-1, 1)
    R = np.asarray(d["R"], dtype=np.float64)
    t = np.asarray(d["t"], dtype=np.float64).reshape(3, 1)

    # OpenCV: Xc = R Xw + t
    Cw = (-R.T @ t).reshape(3)
    return K, dist, R, t, Cw


def load_intrinsics(path):
    """Load only K/dist for relative image-space measurements."""
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    K = np.asarray(d["K"], dtype=np.float64)
    dist = np.asarray(d.get("dist", []), dtype=np.float64).reshape(-1, 1)
    return K, dist


def undistort_pixel(u, v, K, dist):
    """Return normalized undistorted coordinates (x, y) for one pixel."""
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, K, dist, P=None)
    return und[0, 0].astype(np.float64)


def undistort_frame(frame, K, dist):
    """Undistort the complete image before detection/segmentation."""
    return cv2.undistort(frame, K, dist)


def pixel_ray_world(u, v, K, dist, R):
    # 去畸变，得到归一化相机坐标
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, K, dist, P=None)
    x, y = und[0, 0]

    rc = np.array([x, y, 1.0], dtype=np.float64)
    rc /= np.linalg.norm(rc)

    # 相机射线 -> 世界射线
    rw = R.T @ rc
    rw /= np.linalg.norm(rw)
    return rw


def ray_plane_intersection(Cw, rw, normal, d, eps=1e-10):
    # 平面 n^T X + d = 0
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    denom = float(normal @ rw)
    if abs(denom) < eps:
        return None, None

    lam = -(float(normal @ Cw) + d) / denom
    if lam <= 0:
        return None, None

    return Cw + lam * rw, lam


def top_point_from_length(Cw, rt, Pb, L, eps=1e-10):
    # |C + lambda*r - Pb|^2 = L^2
    a = Cw - Pb
    A = float(rt @ rt)
    B = 2.0 * float(rt @ a)
    C = float(a @ a) - L * L

    disc = B * B - 4 * A * C
    if disc < -eps:
        return None, None

    disc = max(0.0, disc)
    s = np.sqrt(disc)
    roots = [(-B - s) / (2*A), (-B + s) / (2*A)]
    valid = [x for x in roots if x > eps]
    if not valid:
        return None, None

    # 取相机前方较远的交点
    lam = max(valid)
    return Cw + lam * rt, lam


def tilt_from_vector(V):
    V = np.asarray(V, dtype=np.float64).reshape(3)
    if V[2] < 0:
        V = -V

    lr = np.degrees(np.arctan2(V[0], V[2]))
    fb = np.degrees(np.arctan2(V[1], V[2]))
    total = np.degrees(np.arctan2(np.hypot(V[0], V[1]), V[2]))
    return float(lr), float(fb), float(total)


def displacement_from_vector(V):
    V = np.asarray(V, dtype=np.float64).reshape(3)
    if V[2] < 0:
        V = -V
    dx, dy = float(V[0]), float(V[1])
    return dx, dy, float(np.hypot(dx, dy))
