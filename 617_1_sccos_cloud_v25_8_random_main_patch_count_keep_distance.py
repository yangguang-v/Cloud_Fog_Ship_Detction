# -*- coding: utf-8 -*-
r"""
617_1_sccos_cloud_v25_8_random_main_patch_count_keep_distance.py

SCCOS-style synthetic cloud augmentation for DIOR-Ship images.

v25.8 核心：
1. 主云不再使用“大区域 + 纹理填充”；
2. 改为多尺度 cloudlet assemblage：巴掌大、苹果大、小絮状云块拼接；
3. 在 v25.1 基础上，主云链默认增加 1–2 组 cloudlet 主块；
4. 周边零星散云和全图自由散云略微增加，但仍避免雾幕；
5. 默认关闭 veil / cirrus，避免变成雾幕。

推荐运行：
python .\617_1_sccos_cloud_v25_8_random_main_patch_count_keep_distance.py --image_dir D:\Downloadss\DIOR\DIOR-Ship619 --out_dir D:\Downloadss\DIOR\DIOR-Ship-SCCOS617\images_region_dense_v25_8 --num_per_image 1 --light_ratio 0.04 --medium_ratio 0.52 --heavy_ratio 0.44 --cloud_density 1.20 --fragment_density 3.80 --max_cloud_area 0.50 --outside_density 0.33 --outside_alpha 0.15 --secondary_density 0.26 --free_density 0.12 --transition_strength 0.17 --min_main_cloud_patches 1 --max_main_cloud_patches 4 --main_patch_min_sep_scale 0.18 --veil_density 0.00 --veil_alpha 0.00 --veil_prob 0.00 --cirrus_density 0.00 --cirrus_alpha 0.00 --cirrus_prob 0.00 --brush_scale 0.94 --seed 2026
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


def normalize01(x, eps=1e-6):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)


def soft_normalize01(x, eps=1e-6):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 3), np.percentile(x, 97)
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def odd_int(x):
    x = int(round(float(x)))
    x = max(1, x)
    return x if x % 2 == 1 else x + 1


def fractal_value_noise(h, w, rng, start_grid=4, octaves=5, persistence=0.55, lacunarity=2.0):
    noise = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    amp_sum = 0.0
    gh = max(2, int(start_grid))
    gw = max(2, int(start_grid * w / max(h, 1)))

    for _ in range(octaves):
        grid = rng.random((gh, gw)).astype(np.float32)
        layer = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        noise += amp * layer
        amp_sum += amp
        amp *= persistence
        gh = max(2, int(round(gh * lacunarity)))
        gw = max(2, int(round(gw * lacunarity)))

    noise /= max(amp_sum, 1e-6)
    return normalize01(noise)


def ridge_noise(h, w, rng, start_grid=8, octaves=4, persistence=0.55, lacunarity=2.0):
    n = fractal_value_noise(h, w, rng, start_grid, octaves, persistence, lacunarity)
    r = 1.0 - np.abs(2.0 * n - 1.0)
    return soft_normalize01(r)


def shift_float_image(img, dx, dy):
    h, w = img.shape[:2]
    mat = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img.astype(np.float32), mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def warp_float_image(img, rng, strength=10.0, grid=4):
    h, w = img.shape[:2]
    dx = fractal_value_noise(h, w, rng, start_grid=grid, octaves=3, persistence=0.60)
    dy = fractal_value_noise(h, w, rng, start_grid=grid, octaves=3, persistence=0.60)
    dx = (dx * 2.0 - 1.0) * strength
    dy = (dy * 2.0 - 1.0) * strength
    blur_k = odd_int(max(21, int(min(h, w) * 0.04)))
    dx = cv2.GaussianBlur(dx, (blur_k, blur_k), 0)
    dy = cv2.GaussianBlur(dy, (blur_k, blur_k), 0)
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    map_x = xx + dx.astype(np.float32)
    map_y = yy + dy.astype(np.float32)
    return cv2.remap(img.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def motion_blur_float(img, ksize, angle_deg):
    ksize = odd_int(max(3, ksize))
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    cv2.line(kernel, (0, ksize // 2), (ksize - 1, ksize // 2), 1.0, thickness=1, lineType=cv2.LINE_AA)
    center = (ksize / 2.0 - 0.5, ksize / 2.0 - 0.5)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, mat, (ksize, ksize))
    kernel = kernel / (kernel.sum() + 1e-6)
    return cv2.filter2D(img.astype(np.float32), -1, kernel)


def build_severity_plan(total_num, rng, light_ratio=0.05, medium_ratio=0.65, heavy_ratio=0.30):
    ratios = np.array([light_ratio, medium_ratio, heavy_ratio], dtype=np.float64)
    ratios = ratios / ratios.sum()
    counts = np.floor(ratios * total_num).astype(int)
    remain = total_num - counts.sum()
    frac = ratios * total_num - counts
    order = np.argsort(-frac)
    for i in range(remain):
        counts[order[i % 3]] += 1
    plan = ["light"] * counts[0] + ["medium"] * counts[1] + ["heavy"] * counts[2]
    rng.shuffle(plan)
    return plan


def get_sccos_params(severity, rng, cloud_density=1.08, fragment_density=3.20):
    """
    v22: top-origin auto-brush main cloud + sparser peripheral scattered clouds.
    Main changes:
    - make the cloud core denser and thicker
    - keep the overall footprint slightly smaller and more compact
    - further suppress fog-like outskirts and transition haze
    - keep SCCOS-style bright fragmented cumulus appearance
    """
    d = float(np.clip(cloud_density, 0.72, 1.48))
    fd = float(np.clip(fragment_density, 0.90, 4.20))

    if severity == "light":
        return {
            "small_count": int(rng.integers(220, 360) * d * fd),
            "small_r_min": 0.0028,
            "small_r_max": 0.0118,
            "cluster_cover": rng.uniform(0.012, 0.030) * min(d, 1.10),
            "large_prob": 0.0,
            "combine_small": rng.uniform(0.82, 0.96),
            "combine_cluster": rng.uniform(0.05, 0.12),
            "combine_large": 0.0,
            "warp_strength": rng.uniform(4.0, 7.5),
            "fray_strength": rng.uniform(0.18, 0.30),
            "porous_strength": rng.uniform(0.16, 0.26),
            "alpha_scale": rng.uniform(0.66, 0.86),
            "edge_soften": int(rng.integers(3, 6)),
            "tiny_frag_alpha": rng.uniform(0.10, 0.18),
            "clump_count": max(2, int(rng.integers(2, 4) * d)),
            "clump_r_min": 0.022,
            "clump_r_max": 0.050,
            "clump_alpha": rng.uniform(0.26, 0.40),
            "white_base": rng.uniform(0.87, 0.94),
            "white_gain": rng.uniform(0.09, 0.16),
            "screen_gain": rng.uniform(0.92, 0.98),
            "shadow_strength": rng.uniform(0.001, 0.006),
            "ambient_shadow": rng.uniform(0.002, 0.010),
            "light_alpha_cut_low": rng.uniform(0.030, 0.050),
            "light_alpha_cut_high": rng.uniform(0.085, 0.130),
        }

    if severity == "medium":
        return {
            "small_count": int(rng.integers(520, 860) * d * fd),
            "small_r_min": 0.0031,
            "small_r_max": 0.0138,
            "cluster_cover": rng.uniform(0.040, 0.092) * min(d, 1.12),
            "large_prob": rng.uniform(0.02, 0.10),
            "combine_small": rng.uniform(0.60, 0.76),
            "combine_cluster": rng.uniform(0.16, 0.28),
            "combine_large": rng.uniform(0.00, 0.04),
            "warp_strength": rng.uniform(5.0, 9.0),
            "fray_strength": rng.uniform(0.20, 0.34),
            "porous_strength": rng.uniform(0.14, 0.24),
            "alpha_scale": rng.uniform(1.04, 1.18),
            "edge_soften": int(rng.integers(3, 7)),
            "tiny_frag_alpha": rng.uniform(0.12, 0.22),
            "clump_count": max(4, int(rng.integers(4, 7) * d)),
            "clump_r_min": 0.030,
            "clump_r_max": 0.072,
            "clump_alpha": rng.uniform(0.38, 0.56),
            "white_base": rng.uniform(0.91, 0.98),
            "white_gain": rng.uniform(0.10, 0.18),
            "screen_gain": rng.uniform(0.96, 1.00),
            "shadow_strength": rng.uniform(0.008, 0.022),
            "ambient_shadow": rng.uniform(0.012, 0.030),
        }

    return {
        "small_count": int(rng.integers(920, 1480) * d * fd),
        "small_r_min": 0.0033,
        "small_r_max": 0.0142,
        "cluster_cover": rng.uniform(0.058, 0.130) * min(d, 1.10),
        "large_prob": rng.uniform(0.03, 0.12),
        "combine_small": rng.uniform(0.52, 0.68),
        "combine_cluster": rng.uniform(0.18, 0.30),
        "combine_large": rng.uniform(0.00, 0.05),
        "warp_strength": rng.uniform(5.0, 9.8),
        "fray_strength": rng.uniform(0.24, 0.38),
        "porous_strength": rng.uniform(0.18, 0.30),
        "alpha_scale": rng.uniform(1.04, 1.18),
        "edge_soften": int(rng.integers(3, 7)),
        "tiny_frag_alpha": rng.uniform(0.14, 0.24),
        "clump_count": max(6, int(rng.integers(6, 10) * d)),
        "clump_r_min": 0.034,
        "clump_r_max": 0.082,
        "clump_alpha": rng.uniform(0.44, 0.64),
        "white_base": rng.uniform(0.92, 0.99),
        "white_gain": rng.uniform(0.11, 0.20),
        "screen_gain": rng.uniform(0.98, 1.02),
        "shadow_strength": rng.uniform(0.016, 0.036),
        "ambient_shadow": rng.uniform(0.018, 0.038),
    }


def draw_rotated_soft_stamp(canvas, x, y, rx, ry, angle_deg, value, rng):
    h, w = canvas.shape[:2]
    pad = int(max(rx, ry) * 2.6) + 4
    x0 = max(0, int(x) - pad)
    x1 = min(w, int(x) + pad + 1)
    y0 = max(0, int(y) - pad)
    y1 = min(h, int(y) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    cx = float(x)
    cy = float(y)
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dx = xx - cx
    dy = yy - cy
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy
    rx = max(float(rx), 1.0)
    ry = max(float(ry), 1.0)
    g = np.exp(-0.5 * ((xr / rx) ** 2 + (yr / ry) ** 2))
    local_noise = rng.random(g.shape).astype(np.float32)
    local_noise = cv2.GaussianBlur(local_noise, (5, 5), 0)
    local_noise = 0.82 + 0.36 * local_noise
    stamp = np.clip(g * local_noise, 0.0, 1.0)
    patch = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = np.maximum(patch, stamp * value)


def draw_small_fragment_stroke(canvas, x, y, r, angle_deg, rng, value=1.0, brush_scale=1.0):
    theta = np.deg2rad(angle_deg)
    dx = np.cos(theta)
    dy = np.sin(theta)
    nx = -dy
    ny = dx
    bscale = float(np.clip(brush_scale, 0.75, 2.20))
    length = r * rng.uniform(2.4, 5.2) * bscale
    width = r * rng.uniform(0.55, 1.10) * (0.86 + 0.32 * bscale)
    steps = max(4, int(length / max(width * 0.68, 1.0)))
    phase = rng.uniform(0, 2 * np.pi)

    for i in range(steps):
        u = i / max(steps - 1, 1)
        t = u - 0.5
        px = x + dx * t * length
        py = y + dy * t * length
        bend = np.sin(u * np.pi * rng.uniform(0.8, 1.8) + phase) * width * rng.uniform(0.20, 0.65)
        px += nx * bend + rng.normal(0, width * 0.35)
        py += ny * bend + rng.normal(0, width * 0.35)
        local_rx = width * rng.uniform(0.88, 1.95)
        local_ry = width * rng.uniform(0.62, 1.38)
        draw_rotated_soft_stamp(canvas, px, py, local_rx, local_ry, angle_deg + rng.normal(0, 22), value * rng.uniform(0.48, 0.96), rng)


def make_small_cumulus_field(h, w, rng, params, region_gate=None, brush_scale=1.0):
    canvas = np.zeros((h, w), dtype=np.float32)
    min_side = min(h, w)
    coarse_prob = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(3, 6)), octaves=4, persistence=0.62)
    coarse_prob = 0.25 + 0.75 * coarse_prob

    if region_gate is not None:
        gate_soft = cv2.GaussianBlur(region_gate.astype(np.float32), (odd_int(max(7, int(min_side * 0.022))), odd_int(max(7, int(min_side * 0.022)))), 0)
        gate_soft = np.clip(region_gate + 0.18 * gate_soft, 0.0, 1.0)
        coarse_prob = coarse_prob * np.power(0.010 + gate_soft, 1.35)

    prob = coarse_prob.reshape(-1).astype(np.float64)
    prob_sum = prob.sum()
    prob = None if prob_sum <= 1e-8 else prob / prob_sum

    area_scale = np.sqrt((h * w) / (800.0 * 800.0))
    n_clouds = max(80, int(params["small_count"] * area_scale))
    starts = rng.choice(h * w, size=n_clouds, replace=True, p=prob)

    for ind in starts:
        y, x = divmod(int(ind), w)
        r = rng.uniform(params["small_r_min"], params["small_r_max"]) * min_side * (0.92 + 0.22 * np.clip(brush_scale, 0.8, 2.0))
        angle = rng.uniform(0, 180)
        val = rng.uniform(0.58, 1.0)
        if rng.random() < 0.70:
            draw_small_fragment_stroke(canvas, x, y, r, angle, rng, value=val, brush_scale=brush_scale)
        else:
            rx = r * rng.uniform(0.55, 1.18)
            ry = r * rng.uniform(0.50, 1.12)
            cv2.ellipse(canvas, (int(x), int(y)), (max(1, int(rx)), max(1, int(ry))), angle, 0, 360, float(val), thickness=-1, lineType=cv2.LINE_AA)

        for _ in range(int(rng.integers(2, 7))):
            dx = rng.normal(0, 0.82 * r)
            dy = rng.normal(0, 0.82 * r)
            xx = int(np.clip(x + dx, 0, w - 1))
            yy = int(np.clip(y + dy, 0, h - 1))
            rr = r * rng.uniform(0.20, 0.58)
            aa = angle + rng.normal(0, 55)
            if rng.random() < 0.60:
                draw_small_fragment_stroke(canvas, xx, yy, rr, aa, rng, value=val * rng.uniform(0.40, 0.88), brush_scale=brush_scale)
            else:
                crx = rr * rng.uniform(0.55, 1.22)
                cry = rr * rng.uniform(0.55, 1.22)
                cv2.ellipse(canvas, (xx, yy), (max(1, int(crx)), max(1, int(cry))), rng.uniform(0, 180), 0, 360, float(val * rng.uniform(0.45, 0.88)), thickness=-1, lineType=cv2.LINE_AA)

    canvas = cv2.GaussianBlur(canvas, (odd_int(max(3, int(min_side * 0.0046))), odd_int(max(3, int(min_side * 0.0046)))), 0)
    canvas = cv2.GaussianBlur(canvas, (odd_int(max(5, int(min_side * 0.0075))), odd_int(max(5, int(min_side * 0.0075)))), 0)

    tex = ridge_noise(h, w, rng, start_grid=int(rng.integers(20, 38)), octaves=4, persistence=0.54)
    canvas = canvas * (0.76 + 0.50 * tex)
    canvas = warp_float_image(canvas, rng, strength=params["warp_strength"] * 0.42, grid=4)

    if region_gate is not None:
        support = np.clip(region_gate + 0.18 * cv2.GaussianBlur(region_gate.astype(np.float32), (odd_int(max(7, int(min_side * 0.020))), odd_int(max(7, int(min_side * 0.020)))), 0), 0.0, 1.0)
        canvas *= support

    return np.clip(canvas, 0.0, 1.0)


def make_cluster_field(h, w, rng, params):
    base1 = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(5, 8)), octaves=5, persistence=0.58)
    base2 = ridge_noise(h, w, rng, start_grid=int(rng.integers(10, 16)), octaves=4, persistence=0.55)
    base = normalize01(0.62 * base1 + 0.38 * base2)
    thr = np.quantile(base, 1.0 - params["cluster_cover"])
    mask = smoothstep(thr - 0.02, thr + 0.08, base)
    mask = warp_float_image(mask, rng, strength=params["warp_strength"], grid=3)
    mask = np.clip(mask, 0.0, 1.0)
    k = odd_int(max(7, int(min(h, w) * 0.017)))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    fine = ridge_noise(h, w, rng, start_grid=int(rng.integers(20, 36)), octaves=4, persistence=0.52)
    mask = mask * (0.82 + 0.36 * fine)
    return np.clip(mask, 0.0, 1.0)


def make_large_thick_field(h, w, rng, params, severity):
    """
    A loose low-frequency thick support field.

    In v12 this layer could look like a single smooth opaque blob. In v13 we
    keep it only as a broad-density scaffold, then let dense fragment masses
    provide the visible heavy-cloud texture.
    """
    if rng.random() > params["large_prob"]:
        return np.zeros((h, w), dtype=np.float32)

    base = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(2, 4)), octaves=5, persistence=0.63)
    rid = ridge_noise(h, w, rng, start_grid=int(rng.integers(6, 10)), octaves=4, persistence=0.56)
    fine = ridge_noise(h, w, rng, start_grid=int(rng.integers(14, 24)), octaves=3, persistence=0.52)
    m = normalize01(0.52 * base + 0.26 * rid + 0.22 * fine)

    cover = rng.uniform(0.080, 0.150) if severity == "medium" else rng.uniform(0.16, 0.28)
    thr = np.quantile(m, 1.0 - cover)
    large = smoothstep(thr - 0.025, thr + 0.080, m)
    large = warp_float_image(large, rng, strength=params["warp_strength"] * 0.95, grid=3)

    tex1 = ridge_noise(h, w, rng, start_grid=int(rng.integers(12, 22)), octaves=4, persistence=0.54)
    tex2 = motion_blur_float(tex1, odd_int(int(rng.integers(11, 23))), rng.uniform(0, 180))
    tex3 = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(24, 40)), octaves=3, persistence=0.50)
    tex = soft_normalize01(0.46 * tex1 + 0.28 * tex2 + 0.26 * tex3)
    large = large * (0.76 + 0.34 * tex)

    pore = ridge_noise(h, w, rng, start_grid=int(rng.integers(18, 30)), octaves=4, persistence=0.52)
    edge = np.clip(large * (1.0 - large) * 4.0, 0.0, 1.0)
    pore_cut = smoothstep(0.60, 0.88, pore) * (0.22 * large + 0.42 * edge)
    large = np.clip(large - 0.18 * pore_cut, 0.0, 1.0)

    wisps = motion_blur_float(edge * tex, odd_int(int(rng.integers(9, 19))), rng.uniform(0, 180))
    large = np.clip(large + 0.07 * soft_normalize01(wisps) * edge, 0.0, 1.0)

    k = odd_int(max(9, int(min(h, w) * 0.014)))
    large = cv2.GaussianBlur(large, (k, k), 0)

    tex4 = ridge_noise(h, w, rng, start_grid=int(rng.integers(24, 44)), octaves=3, persistence=0.50)
    large = np.clip(large * (0.82 + 0.25 * tex4), 0.0, 1.0)
    return np.clip(large, 0.0, 1.0)


def draw_region_stroke_for_gate(canvas, x, y, rx, ry, angle_deg, value, rng, brush_scale=1.0):
    theta = np.deg2rad(angle_deg)
    dx = np.cos(theta)
    dy = np.sin(theta)
    nx = -dy
    ny = dx
    bscale = float(np.clip(brush_scale, 0.75, 2.20))
    length = max(rx, ry) * rng.uniform(2.8, 5.8) * bscale
    width = min(rx, ry) * rng.uniform(0.70, 1.25) * (0.84 + 0.30 * bscale)
    steps = max(9, int(length / max(width * 0.52, 1.0)))
    phase = rng.uniform(0, 2 * np.pi)
    curve_amp = width * rng.uniform(0.40, 1.05)

    for i in range(steps):
        u = i / max(steps - 1, 1)
        t = u - 0.5
        px = x + dx * t * length
        py = y + dy * t * length
        bend = np.sin(u * np.pi * rng.uniform(1.0, 2.2) + phase) * curve_amp
        px += nx * bend + rng.normal(0, width * 0.28)
        py += ny * bend + rng.normal(0, width * 0.28)
        local_rx = width * rng.uniform(1.18, 2.70)
        local_ry = width * rng.uniform(0.72, 1.58)
        draw_rotated_soft_stamp(canvas, px, py, local_rx, local_ry, angle_deg + rng.normal(0, 24), value * rng.uniform(0.38, 0.92), rng)




def make_top_origin_brush_gate(h, w, rng, severity, target_cover, brush_scale=1.0):
    """
    v22 helper:
    Build the main cloud support so it starts from the upper part of the image
    and grows downward with controlled automatic brush strokes. This keeps the
    main cloud shape more like a guided SCCOS cloud mass rather than a random
    central blob.
    """
    min_side = min(h, w)
    gate = np.zeros((h, w), dtype=np.float32)

    if severity == "light":
        n_strokes = int(rng.integers(2, 4))
        path_len = min_side * rng.uniform(0.22, 0.36)
        width0 = min_side * rng.uniform(0.045, 0.080)
    elif severity == "medium":
        n_strokes = int(rng.integers(3, 5))
        path_len = min_side * rng.uniform(0.28, 0.46)
        width0 = min_side * rng.uniform(0.055, 0.095)
    else:
        n_strokes = int(rng.integers(3, 6))
        path_len = min_side * rng.uniform(0.34, 0.56)
        width0 = min_side * rng.uniform(0.065, 0.115)

    anchor_x = w * rng.uniform(0.24, 0.76)
    anchor_y = h * rng.uniform(-0.03, 0.07)
    base_angle = 90.0 + rng.uniform(-26.0, 26.0)

    for _ in range(n_strokes):
        stroke_angle = base_angle + rng.normal(0, 10)
        theta = np.deg2rad(stroke_angle)
        dir_x = np.cos(theta)
        dir_y = np.sin(theta)
        norm_x = -dir_y
        norm_y = dir_x

        start_x = anchor_x + rng.normal(0, min_side * 0.040)
        start_y = anchor_y + rng.normal(0, min_side * 0.018)
        stroke_len = path_len * rng.uniform(0.86, 1.18)
        steps = int(rng.integers(18, 34))
        phase = rng.uniform(0, 2 * np.pi)
        curve_amp = min_side * rng.uniform(0.018, 0.060) * (0.92 + 0.18 * np.clip(brush_scale, 0.8, 2.0))

        for i in range(steps):
            u = i / max(steps - 1, 1)
            taper = 0.48 + 0.62 * np.sin(np.pi * np.clip(u, 0.0, 1.0))
            px = start_x + dir_x * stroke_len * u
            py = start_y + dir_y * stroke_len * u
            bend = np.sin(u * np.pi * rng.uniform(0.9, 1.5) + phase) * curve_amp
            px += norm_x * bend + rng.normal(0, width0 * 0.18)
            py += norm_y * bend + rng.normal(0, width0 * 0.18)

            local_rx = width0 * (0.85 + 1.00 * taper) * rng.uniform(0.85, 1.60)
            local_ry = width0 * (0.62 + 0.82 * taper) * rng.uniform(0.65, 1.25)
            local_val = rng.uniform(0.42, 0.92)
            draw_rotated_soft_stamp(gate, px, py, local_rx, local_ry, stroke_angle + rng.normal(0, 18), local_val, rng)

            if rng.random() < 0.32:
                side = rng.choice([-1.0, 1.0])
                off = width0 * rng.uniform(0.45, 1.45)
                sx = px + norm_x * off * side + rng.normal(0, width0 * 0.14)
                sy = py + norm_y * off * side + rng.normal(0, width0 * 0.14)
                draw_rotated_soft_stamp(
                    gate,
                    sx,
                    sy,
                    local_rx * rng.uniform(0.45, 0.88),
                    local_ry * rng.uniform(0.42, 0.80),
                    stroke_angle + rng.normal(0, 28),
                    local_val * rng.uniform(0.36, 0.72),
                    rng,
                )

    # small head near the top entrance, to make the cloud visually start from above
    for _ in range(int(rng.integers(1, 3))):
        cx = anchor_x + rng.normal(0, min_side * 0.045)
        cy = h * rng.uniform(0.02, 0.14)
        rx = min_side * rng.uniform(0.07, 0.13)
        ry = min_side * rng.uniform(0.05, 0.10)
        draw_rotated_soft_stamp(gate, cx, cy, rx, ry, base_angle + rng.normal(0, 18), rng.uniform(0.46, 0.82), rng)

    k = odd_int(max(13, int(min_side * 0.026)))
    gate = cv2.GaussianBlur(gate, (k, k), 0)
    gate = soft_normalize01(gate)

    # apply gentle texture so the gate is not a uniform stroke ribbon
    tex1 = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(4, 8)), octaves=4, persistence=0.60)
    tex2 = ridge_noise(h, w, rng, start_grid=int(rng.integers(12, 22)), octaves=3, persistence=0.52)
    gate = gate * (0.76 + 0.24 * tex1 + 0.14 * tex2)

    thr = np.quantile(gate, 1.0 - target_cover)
    gate = smoothstep(thr - 0.050, thr + 0.150, gate)

    holes = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(6, 10)), octaves=3, persistence=0.56)
    edge = np.clip(gate * (1.0 - gate) * 4.0, 0.0, 1.0)
    gate = gate * (1.0 - 0.14 * smoothstep(0.62, 0.90, holes) * (0.44 * edge + 0.12 * gate))

    k2 = odd_int(max(5, int(min_side * 0.010)))
    gate = cv2.GaussianBlur(gate, (k2, k2), 0)
    gate = np.clip(gate, 0.0, 1.0)
    gate[gate < 0.003] = 0.0
    return gate




def make_top_fragment_assemblage_gate(h, w, rng, severity, target_cover, brush_scale=1.0):
    """
    v23 helper:
    Build the main SCCOS cloud support as many small/medium cloud pieces
    assembled together, instead of a single large continuous patch. The overall
    cluster still tends to originate from the upper part of the image, but the
    body is composed of palm-sized / apple-sized bright cumulus fragments plus
    a few surrounding chips.
    """
    min_side = min(h, w)
    gate = np.zeros((h, w), dtype=np.float32)

    if severity == "light":
        n_centers = int(rng.integers(2, 4))
        pieces_low, pieces_high = 14, 24
        spread_main = min_side * rng.uniform(0.045, 0.085)
        path_len = min_side * rng.uniform(0.18, 0.28)
    elif severity == "medium":
        n_centers = int(rng.integers(3, 5))
        pieces_low, pieces_high = 22, 36
        spread_main = min_side * rng.uniform(0.055, 0.110)
        path_len = min_side * rng.uniform(0.22, 0.38)
    else:
        n_centers = int(rng.integers(4, 6))
        pieces_low, pieces_high = 30, 48
        spread_main = min_side * rng.uniform(0.070, 0.135)
        path_len = min_side * rng.uniform(0.28, 0.46)

    anchor_x = w * rng.uniform(0.22, 0.78)
    anchor_y = h * rng.uniform(-0.02, 0.09)
    base_angle = 90.0 + rng.uniform(-24.0, 24.0)
    theta = np.deg2rad(base_angle)
    dir_x = np.cos(theta)
    dir_y = np.sin(theta)
    norm_x = -dir_y
    norm_y = dir_x

    centers = []
    for i in range(n_centers):
        if n_centers == 1:
            u = 0.0
        else:
            u = i / (n_centers - 1)
        along = path_len * (0.10 + 0.90 * u)
        sideways = spread_main * rng.uniform(-0.95, 0.95) + np.sin(u * np.pi * rng.uniform(0.8, 1.4) + rng.uniform(0, 2*np.pi)) * spread_main * 0.40
        cx = anchor_x + dir_x * along + norm_x * sideways + rng.normal(0, min_side * 0.018)
        cy = anchor_y + dir_y * along + norm_y * sideways + rng.normal(0, min_side * 0.018)
        centers.append((cx, cy, base_angle + rng.normal(0, 18), spread_main * rng.uniform(0.75, 1.20)))

    # Dense cloud pieces in each local cluster
    for (cx0, cy0, local_angle, local_spread) in centers:
        local_n = int(rng.integers(pieces_low, pieces_high))
        local_theta = np.deg2rad(local_angle)
        ldx, ldy = np.cos(local_theta), np.sin(local_theta)
        lnx, lny = -ldy, ldx
        for _ in range(local_n):
            # mostly small pieces, some medium pieces, very few larger pieces
            coin = rng.random()
            if coin < 0.64:
                rx = min_side * rng.uniform(0.014, 0.030)
                ry = min_side * rng.uniform(0.012, 0.026)
            elif coin < 0.93:
                rx = min_side * rng.uniform(0.022, 0.042)
                ry = min_side * rng.uniform(0.016, 0.035)
            else:
                rx = min_side * rng.uniform(0.032, 0.056)
                ry = min_side * rng.uniform(0.022, 0.046)

            off_a = rng.normal(0, local_spread * 0.85)
            off_b = rng.normal(0, local_spread * 0.55)
            px = cx0 + ldx * off_a + lnx * off_b
            py = cy0 + ldy * off_a + lny * off_b
            ang = local_angle + rng.normal(0, 34)
            val = rng.uniform(0.36, 0.86)
            draw_rotated_soft_stamp(gate, px, py, rx, ry, ang, val, rng)

            # small satellites around some main pieces, to form assembled fragments
            if rng.random() < 0.42:
                sat_n = int(rng.integers(1, 4))
                for _ in range(sat_n):
                    sx = px + rng.normal(0, rx * 1.35)
                    sy = py + rng.normal(0, ry * 1.35)
                    srx = rx * rng.uniform(0.32, 0.70)
                    sry = ry * rng.uniform(0.32, 0.72)
                    sval = val * rng.uniform(0.40, 0.82)
                    draw_rotated_soft_stamp(gate, sx, sy, srx, sry, ang + rng.normal(0, 28), sval, rng)

    # A few loose surrounding chips near the main assemblage, but not too dense
    surround_n = int(rng.integers(12, 28) * (0.9 if severity == 'light' else 1.0 if severity == 'medium' else 1.15))
    for _ in range(surround_n):
        base_cx, base_cy, local_angle, local_spread = centers[int(rng.integers(0, len(centers)))]
        rad = local_spread * rng.uniform(1.2, 2.4)
        ang0 = rng.uniform(0, 2 * np.pi)
        px = base_cx + np.cos(ang0) * rad + rng.normal(0, min_side * 0.015)
        py = base_cy + np.sin(ang0) * rad + rng.normal(0, min_side * 0.015)
        rx = min_side * rng.uniform(0.010, 0.026)
        ry = min_side * rng.uniform(0.009, 0.022)
        draw_rotated_soft_stamp(gate, px, py, rx, ry, local_angle + rng.normal(0, 45), rng.uniform(0.22, 0.52), rng)

    # Gentle local blend, but keep piece-assembly feeling
    k = odd_int(max(5, int(min_side * 0.010)))
    gate = cv2.GaussianBlur(gate, (k, k), 0)
    gate = soft_normalize01(gate)

    tex1 = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(5, 9)), octaves=4, persistence=0.60)
    tex2 = ridge_noise(h, w, rng, start_grid=int(rng.integers(14, 24)), octaves=3, persistence=0.52)
    gate = gate * (0.82 + 0.20 * tex1 + 0.10 * tex2)

    thr = np.quantile(gate, 1.0 - target_cover)
    gate = smoothstep(thr - 0.035, thr + 0.110, gate)

    # small gaps between assembled pieces
    chip = ridge_noise(h, w, rng, start_grid=int(rng.integers(18, 30)), octaves=3, persistence=0.52)
    hole = smoothstep(0.64, 0.90, chip)
    edge = np.clip(gate * (1.0 - gate) * 4.0, 0.0, 1.0)
    gate = gate * (1.0 - 0.12 * hole * (0.48 * edge + 0.08 * gate))

    k2 = odd_int(max(3, int(min_side * 0.005)))
    gate = cv2.GaussianBlur(gate, (k2, k2), 0)
    gate = np.clip(gate, 0.0, 1.0)
    gate[gate < 0.003] = 0.0
    return gate



def make_cloud_region_gate(h, w, rng, severity, max_cloud_area=0.42, brush_scale=1.0):
    """
    Natural support field for the main cloud.

    v23 change:
    Prefer a SCCOS-style assemblage of many small/medium cloud pieces, guided
    from the upper part of the image downward. The main cloud should look like
    many differently sized fragments assembled together, not one giant smooth
    patch. The gate is only a probability/support field, not a hard boundary.
    """
    min_side = min(h, w)
    max_cloud_area = float(np.clip(max_cloud_area, 0.10, 0.70))

    if severity == "light":
        target_cover = rng.uniform(0.08, 0.16)
        n_core = int(rng.integers(2, 4))
    elif severity == "medium":
        target_cover = rng.uniform(0.12, 0.24)
        n_core = int(rng.integers(3, 6))
    else:
        target_cover = rng.uniform(0.18, 0.30)
        n_core = int(rng.integers(4, 7))

    target_cover = float(min(target_cover, max_cloud_area))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gate = np.zeros((h, w), dtype=np.float32)

    def add_soft_ellipse(canvas, cx, cy, rx, ry, angle_deg, value=1.0, sharpness=0.72):
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx = xx - float(cx)
        dy = yy - float(cy)
        xr = cos_t * dx + sin_t * dy
        yr = -sin_t * dx + cos_t * dy
        rx0 = max(float(rx), 2.0)
        ry0 = max(float(ry), 2.0)
        d = (xr / rx0) ** 2 + (yr / ry0) ** 2
        # smooth elliptical density, not a hard ellipse
        e = np.exp(-0.5 * d)
        e = np.power(np.clip(e, 0.0, 1.0), sharpness)
        canvas[:] = np.maximum(canvas, e.astype(np.float32) * float(value))

    mode = rng.choice(["fragment_assemblage", "top_brush", "band", "compact", "multi_lobe"], p=[0.78, 0.12, 0.06, 0.03, 0.01])

    if mode == "fragment_assemblage":
        gate = make_top_fragment_assemblage_gate(h, w, rng, severity, target_cover, brush_scale=brush_scale)

    elif mode == "top_brush":
        gate = make_top_origin_brush_gate(h, w, rng, severity, target_cover, brush_scale=brush_scale)

    elif mode == "band":
        base_angle = rng.uniform(0, 180)
        theta = np.deg2rad(base_angle)
        dir_x, dir_y = np.cos(theta), np.sin(theta)
        norm_x, norm_y = -dir_y, dir_x
        cx0 = w * rng.uniform(0.25, 0.75)
        cy0 = h * rng.uniform(0.25, 0.75)
        length = min_side * rng.uniform(0.26, 0.52)
        n_seg = int(rng.integers(3, 7))
        for i in range(n_seg):
            u = (i / max(n_seg - 1, 1)) - 0.5
            cx = cx0 + dir_x * u * length + norm_x * rng.normal(0, min_side * 0.035)
            cy = cy0 + dir_y * u * length + norm_y * rng.normal(0, min_side * 0.035)
            rx = min_side * rng.uniform(0.070, 0.150)
            ry = min_side * rng.uniform(0.045, 0.095)
            if rng.random() < 0.50:
                rx *= rng.uniform(1.15, 1.65)
            add_soft_ellipse(gate, cx, cy, rx, ry, base_angle + rng.normal(0, 18), rng.uniform(0.55, 1.00), sharpness=rng.uniform(0.62, 0.82))

    elif mode == "compact":
        cx0 = w * rng.uniform(0.20, 0.80)
        cy0 = h * rng.uniform(0.20, 0.80)
        base_angle = rng.uniform(0, 180)
        add_soft_ellipse(
            gate,
            cx0,
            cy0,
            min_side * rng.uniform(0.145, 0.255),
            min_side * rng.uniform(0.095, 0.180),
            base_angle,
            rng.uniform(0.70, 1.00),
            sharpness=rng.uniform(0.65, 0.85),
        )
        for _ in range(n_core):
            ang = rng.uniform(0, 2 * np.pi)
            dist = min_side * rng.uniform(0.030, 0.150)
            cx = cx0 + np.cos(ang) * dist + rng.normal(0, min_side * 0.030)
            cy = cy0 + np.sin(ang) * dist + rng.normal(0, min_side * 0.030)
            rx = min_side * rng.uniform(0.060, 0.140)
            ry = min_side * rng.uniform(0.045, 0.110)
            add_soft_ellipse(gate, cx, cy, rx, ry, base_angle + rng.normal(0, 55), rng.uniform(0.40, 0.86), sharpness=rng.uniform(0.62, 0.88))

    else:  # multi_lobe
        n_centers = int(rng.integers(2, 4))
        base_angle = rng.uniform(0, 180)
        center_x = w * rng.uniform(0.22, 0.78)
        center_y = h * rng.uniform(0.22, 0.78)
        for _ in range(n_centers):
            cx0 = center_x + rng.normal(0, min_side * 0.13)
            cy0 = center_y + rng.normal(0, min_side * 0.13)
            local_angle = base_angle + rng.normal(0, 45)
            add_soft_ellipse(
                gate,
                cx0,
                cy0,
                min_side * rng.uniform(0.085, 0.180),
                min_side * rng.uniform(0.055, 0.125),
                local_angle,
                rng.uniform(0.50, 0.95),
                sharpness=rng.uniform(0.62, 0.85),
            )
            if rng.random() < 0.75:
                cx = cx0 + rng.normal(0, min_side * 0.060)
                cy = cy0 + rng.normal(0, min_side * 0.060)
                add_soft_ellipse(
                    gate,
                    cx,
                    cy,
                    min_side * rng.uniform(0.050, 0.120),
                    min_side * rng.uniform(0.040, 0.095),
                    local_angle + rng.normal(0, 50),
                    rng.uniform(0.35, 0.75),
                    sharpness=rng.uniform(0.60, 0.86),
                )

    # Smooth the union before applying noise so the global support stays natural.
    k0 = odd_int(max(15, int(min_side * 0.030)))
    gate = cv2.GaussianBlur(gate, (k0, k0), 0)
    gate = soft_normalize01(gate)

    # Multi-scale texture breaks the smooth support, but does not create odd limbs.
    noise_low = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(4, 8)), octaves=5, persistence=0.60)
    noise_mid = ridge_noise(h, w, rng, start_grid=int(rng.integers(11, 20)), octaves=4, persistence=0.52)
    noise_flow = motion_blur_float(noise_mid, odd_int(int(rng.integers(11, 25))), rng.uniform(0, 180))
    tex = soft_normalize01(0.46 * noise_low + 0.34 * noise_mid + 0.20 * noise_flow)

    gate = gate * (0.68 + 0.44 * tex)

    # Quantile threshold controls approximate coverage without producing hard edges.
    thr = np.quantile(gate, 1.0 - target_cover)
    gate = smoothstep(thr - 0.055, thr + 0.165, gate)

    # Interior holes and edge chips. Keep them modest to avoid strange shapes.
    holes_large = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(5, 9)), octaves=3, persistence=0.58)
    holes_ridge = ridge_noise(h, w, rng, start_grid=int(rng.integers(16, 30)), octaves=3, persistence=0.50)
    hole_mask = 0.55 * smoothstep(0.58, 0.90, holes_large) + 0.45 * smoothstep(0.62, 0.92, holes_ridge)

    edge = np.clip(gate * (1.0 - gate) * 4.0, 0.0, 1.0)
    if severity == "light":
        hole_strength = 0.22
    elif severity == "medium":
        hole_strength = 0.18
    else:
        hole_strength = 0.16

    gate = gate * (1.0 - hole_strength * hole_mask * (0.55 * edge + 0.18 * gate))

    # Final wide feather. Values below this are not zeroed too aggressively; later
    # build_feather_support will create natural outside spill.
    k1 = odd_int(max(7, int(min_side * 0.012)))
    gate = cv2.GaussianBlur(gate, (k1, k1), 0)
    gate = np.clip(gate, 0.0, 1.0)
    gate[gate < 0.003] = 0.0
    return gate



def make_secondary_cloud_gate(
    h,
    w,
    rng,
    severity,
    secondary_density=0.55,
    brush_scale=1.0,
):
    """
    Natural secondary support fields.

    v18: no random-stroke region drawing. Secondary regions are compact soft
    ellipses, so they no longer create strange arms or claw-like cloud regions.
    """
    min_side = min(h, w)
    secondary_density = float(np.clip(secondary_density, 0.0, 1.50))

    gate = np.zeros((h, w), dtype=np.float32)
    if secondary_density <= 1e-6:
        return gate

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def add_soft_ellipse(canvas, cx, cy, rx, ry, angle_deg, value=1.0, sharpness=0.75):
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx = xx - float(cx)
        dy = yy - float(cy)
        xr = cos_t * dx + sin_t * dy
        yr = -sin_t * dx + cos_t * dy
        rx0 = max(float(rx), 2.0)
        ry0 = max(float(ry), 2.0)
        d = (xr / rx0) ** 2 + (yr / ry0) ** 2
        e = np.exp(-0.5 * d)
        e = np.power(np.clip(e, 0.0, 1.0), sharpness)
        canvas[:] = np.maximum(canvas, e.astype(np.float32) * float(value))

    if severity == "light":
        n_regions = int(rng.integers(1, 2))
        target_cover = rng.uniform(0.020, 0.052) * secondary_density
    elif severity == "medium":
        n_regions = int(rng.integers(1, 3))
        target_cover = rng.uniform(0.036, 0.090) * secondary_density
    else:
        n_regions = int(rng.integers(1, 4))
        target_cover = rng.uniform(0.050, 0.120) * secondary_density

    target_cover = float(np.clip(target_cover, 0.006, 0.160))

    for _ in range(n_regions):
        cx0 = w * rng.uniform(0.08, 0.92)
        cy0 = h * rng.uniform(0.08, 0.92)
        base_angle = rng.uniform(0, 180)

        rx = min_side * rng.uniform(0.045, 0.115)
        ry = min_side * rng.uniform(0.030, 0.085)
        add_soft_ellipse(gate, cx0, cy0, rx, ry, base_angle, rng.uniform(0.45, 0.92), sharpness=rng.uniform(0.64, 0.88))

        if rng.random() < 0.60:
            cx = cx0 + rng.normal(0, min_side * 0.060)
            cy = cy0 + rng.normal(0, min_side * 0.060)
            add_soft_ellipse(
                gate,
                cx,
                cy,
                min_side * rng.uniform(0.030, 0.080),
                min_side * rng.uniform(0.022, 0.060),
                base_angle + rng.normal(0, 65),
                rng.uniform(0.28, 0.66),
                sharpness=rng.uniform(0.62, 0.90),
            )

    k = odd_int(max(9, int(min_side * 0.024)))
    gate = cv2.GaussianBlur(gate, (k, k), 0)
    gate = soft_normalize01(gate)

    noise1 = fractal_value_noise(
        h,
        w,
        rng,
        start_grid=int(rng.integers(6, 11)),
        octaves=4,
        persistence=0.60,
    )

    noise2 = ridge_noise(
        h,
        w,
        rng,
        start_grid=int(rng.integers(14, 28)),
        octaves=3,
        persistence=0.52,
    )

    gate = gate * (0.62 + 0.26 * noise1 + 0.18 * noise2)

    thr = np.quantile(gate, 1.0 - target_cover)
    gate = smoothstep(thr - 0.052, thr + 0.145, gate)

    holes = fractal_value_noise(
        h,
        w,
        rng,
        start_grid=int(rng.integers(5, 9)),
        octaves=3,
        persistence=0.58,
    )
    holes = smoothstep(0.58, 0.88, holes)
    gate = gate * (1.0 - 0.20 * holes * smoothstep(0.08, 0.86, gate))

    k2 = odd_int(max(5, int(min_side * 0.008)))
    gate = cv2.GaussianBlur(gate, (k2, k2), 0)

    gate = np.clip(gate, 0.0, 1.0)
    gate[gate < 0.003] = 0.0
    return gate


def limit_alpha_coverage(alpha, max_cloud_area=0.42):
    max_cloud_area = float(np.clip(max_cloud_area, 0.10, 0.75))
    area = float((alpha > 0.025).mean())
    if area <= max_cloud_area:
        return alpha
    q = np.clip(1.0 - max_cloud_area, 0.0, 0.99)
    thr = float(np.quantile(alpha, q))
    keep = smoothstep(thr, thr + 0.060, alpha)
    return np.clip(alpha * keep, 0.0, 1.0)


def make_puffy_clump_field(h, w, rng, params, severity, brush_scale=1.0):
    field = np.zeros((h, w), dtype=np.float32)
    min_side = min(h, w)
    region = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(2, 5)), octaves=4, persistence=0.62)
    prob = region.reshape(-1).astype(np.float64) + 1e-8
    prob = prob / prob.sum()
    centers = rng.choice(h * w, size=params["clump_count"], replace=True, p=prob)
    global_wind = rng.uniform(0, 180)

    for ind in centers:
        cy, cx = divmod(int(ind), w)
        base_r = rng.uniform(params["clump_r_min"], params["clump_r_max"]) * min_side * (0.92 + 0.20 * np.clip(brush_scale, 0.8, 2.0))

        if severity == "light":
            n_strokes, steps_min, steps_max, length_scale = int(rng.integers(2, 4)), 7, 13, rng.uniform(1.8, 3.0)
        elif severity == "medium":
            n_strokes, steps_min, steps_max, length_scale = int(rng.integers(3, 6)), 10, 18, rng.uniform(2.2, 3.8)
        else:
            n_strokes, steps_min, steps_max, length_scale = int(rng.integers(4, 8)), 12, 24, rng.uniform(2.8, 4.8)

        for _ in range(n_strokes):
            stroke_angle = global_wind + rng.normal(0, 28)
            sx = cx + rng.normal(0, base_r * 0.45)
            sy = cy + rng.normal(0, base_r * 0.45)
            steps = int(rng.integers(steps_min, steps_max))
            total_len = base_r * length_scale * rng.uniform(0.70, 1.25)
            step_len = total_len / max(steps, 1)
            curve_phase = rng.uniform(0, 2 * np.pi)
            curve_amp = base_r * rng.uniform(0.18, 0.48)
            brush_rx = base_r * rng.uniform(0.22, 0.42)
            brush_ry = base_r * rng.uniform(0.10, 0.24)
            stroke_value = rng.uniform(0.55, 1.0)
            theta = np.deg2rad(stroke_angle)
            dir_x = np.cos(theta)
            dir_y = np.sin(theta)
            norm_x = -dir_y
            norm_y = dir_x

            for t in range(steps):
                u = t / max(steps - 1, 1)
                px = sx + dir_x * step_len * t
                py = sy + dir_y * step_len * t
                bend = np.sin(u * np.pi * rng.uniform(0.8, 1.6) + curve_phase)
                px += norm_x * bend * curve_amp + rng.normal(0, base_r * 0.10)
                py += norm_y * bend * curve_amp + rng.normal(0, base_r * 0.10)
                taper = 0.55 + 0.55 * np.sin(np.pi * u)
                local_rx = brush_rx * taper * rng.uniform(0.75, 1.35)
                local_ry = brush_ry * taper * rng.uniform(0.75, 1.45)
                if rng.random() < 0.18:
                    local_rx *= rng.uniform(0.75, 1.10)
                    local_ry *= rng.uniform(0.95, 1.35)
                else:
                    local_rx *= rng.uniform(1.10, 1.65)
                draw_rotated_soft_stamp(field, px, py, local_rx, local_ry, stroke_angle + rng.normal(0, 18), stroke_value * rng.uniform(0.65, 1.0), rng)

        satellite_num = int(rng.integers(4, 8)) if severity == "light" else int(rng.integers(7, 14)) if severity == "medium" else int(rng.integers(9, 18))
        for _ in range(satellite_num):
            sat_angle = global_wind + rng.normal(0, 55)
            dist = base_r * rng.uniform(0.55, 1.45)
            x = cx + np.cos(np.deg2rad(sat_angle)) * dist + rng.normal(0, base_r * 0.35)
            y = cy + np.sin(np.deg2rad(sat_angle)) * dist + rng.normal(0, base_r * 0.35)
            r = base_r * rng.uniform(0.08, 0.22)
            draw_rotated_soft_stamp(field, x, y, r * rng.uniform(1.0, 2.1), r * rng.uniform(0.55, 1.15), sat_angle + rng.normal(0, 30), rng.uniform(0.35, 0.80), rng)

    blur_k = odd_int(max(7, int(min_side * rng.uniform(0.007, 0.013))))
    field = cv2.GaussianBlur(field, (blur_k, blur_k), 0)
    tex1 = ridge_noise(h, w, rng, start_grid=int(rng.integers(14, 26)), octaves=4, persistence=0.54)
    tex2 = ridge_noise(h, w, rng, start_grid=int(rng.integers(26, 46)), octaves=3, persistence=0.50)
    tex3 = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(36, 64)), octaves=3, persistence=0.48)
    texture = soft_normalize01(0.55 * tex1 + 0.30 * tex2 + 0.15 * tex3)
    field = field * (0.72 + 0.58 * texture)
    pore = ridge_noise(h, w, rng, start_grid=int(rng.integers(22, 42)), octaves=4, persistence=0.50)
    edge = np.clip(field * (1.0 - field) * 4.0, 0.0, 1.0)
    cut = smoothstep(0.56, 0.88, pore) * (0.28 * field + 0.55 * edge)
    field = np.clip(field - 0.24 * cut, 0.0, 1.0)
    filament = motion_blur_float(edge * texture, odd_int(int(rng.integers(13, 27))), global_wind + rng.normal(0, 20))
    field = np.clip(field + 0.12 * soft_normalize01(filament) * edge, 0.0, 1.0)
    field = warp_float_image(field, rng, strength=params["warp_strength"] * 0.85, grid=4)
    final_k = odd_int(max(5, int(min_side * 0.0058)))
    field = cv2.GaussianBlur(field, (final_k, final_k), 0)
    return np.clip(field, 0.0, 1.0)


def refine_sccos_edges(mask, rng, params):
    edge = np.clip(mask * (1.0 - mask) * 4.0, 0.0, 1.0)
    fray1 = ridge_noise(mask.shape[0], mask.shape[1], rng, start_grid=int(rng.integers(20, 38)), octaves=4, persistence=0.52)
    fray2 = ridge_noise(mask.shape[0], mask.shape[1], rng, start_grid=int(rng.integers(30, 52)), octaves=3, persistence=0.50)
    cut = smoothstep(0.52, 0.88, fray1) * edge * params["fray_strength"]
    add = smoothstep(0.66, 0.92, fray2) * edge * (0.45 * params["fray_strength"])
    mask = np.clip(mask - cut + add, 0.0, 1.0)
    wisp = motion_blur_float(edge * fray2, odd_int(int(rng.integers(9, 21))), rng.uniform(0, 180))
    mask = np.clip(mask + 0.12 * params["fray_strength"] * soft_normalize01(wisp), 0.0, 1.0)
    k = odd_int(params["edge_soften"])
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return np.clip(mask, 0.0, 1.0)


def add_tiny_cloud_fragments(h, w, rng, params, severity, region_gate=None, fragment_density=2.30, brush_scale=1.0):
    frag = np.zeros((h, w), dtype=np.float32)
    min_side = min(h, w)
    fd = float(np.clip(fragment_density, 0.80, 2.80))
    low, high = (220, 420) if severity == "light" else (440, 760) if severity == "medium" else (620, 1050)
    n = int(np.sqrt((h * w) / (800.0 * 800.0)) * rng.integers(low, high) * fd)
    n = max(120, n)

    if region_gate is not None:
        prob_map = np.clip(region_gate, 0.0, 1.0)
        prob_map = cv2.GaussianBlur(prob_map.astype(np.float32), (odd_int(max(7, int(min_side * 0.018))), odd_int(max(7, int(min_side * 0.018)))), 0)
        prob_map = np.clip(region_gate + 0.16 * prob_map, 0.0, 1.0)
        prob = prob_map.reshape(-1).astype(np.float64)
        prob = None if prob.sum() <= 1e-8 else prob / prob.sum()
    else:
        prob = None

    inds = rng.choice(h * w, size=n, replace=True, p=prob)
    for ind in inds:
        y, x = divmod(int(ind), w)
        r = rng.uniform(0.0014, 0.0056) * min_side
        angle = rng.uniform(0, 180)
        if rng.random() < 0.55:
            draw_small_fragment_stroke(frag, x, y, r, angle, rng, value=rng.uniform(0.36, 0.85), brush_scale=brush_scale)
        else:
            rx = r * rng.uniform(0.50, 1.20)
            ry = r * rng.uniform(0.50, 1.20)
            cv2.ellipse(frag, (int(x), int(y)), (max(1, int(rx)), max(1, int(ry))), angle, 0, 360, float(rng.uniform(0.42, 1.0)), thickness=-1, lineType=cv2.LINE_AA)

    k = odd_int(max(3, int(min_side * 0.0036)))
    frag = cv2.GaussianBlur(frag, (k, k), 0)
    if region_gate is not None:
        support = np.clip(region_gate + 0.18 * cv2.GaussianBlur(region_gate.astype(np.float32), (odd_int(max(7, int(min_side * 0.018))), odd_int(max(7, int(min_side * 0.018)))), 0), 0.0, 1.0)
        frag *= support
    return np.clip(frag, 0.0, 1.0)


def make_bright_fragmented_cumulus_core(
    h,
    w,
    rng,
    params,
    severity,
    region_gate=None,
    fragment_density=3.20,
    brush_scale=1.0,
):
    """
    v19 main addition:
    Build SCCOS-style bright fragmented cumulus core.
    The cloud should look like many bright white cloudlets packed together,
    with holes and broken edges, instead of a fog sheet.
    """
    core = np.zeros((h, w), dtype=np.float32)
    min_side = min(h, w)
    fd = float(np.clip(fragment_density, 0.90, 4.20))
    bscale = float(np.clip(brush_scale, 0.75, 2.00))

    if severity == "light":
        low, high = 320, 600
        rmin, rmax = 0.0018, 0.0052
        child_rng = (3, 6)
        grand_prob = 0.24
    elif severity == "medium":
        low, high = 900, 1500
        rmin, rmax = 0.0019, 0.0056
        child_rng = (4, 8)
        grand_prob = 0.34
    else:
        low, high = 1600, 2500
        rmin, rmax = 0.0020, 0.0060
        child_rng = (5, 10)
        grand_prob = 0.42

    area_scale = np.sqrt((h * w) / (800.0 * 800.0))
    n_centers = int(rng.integers(low, high) * area_scale * fd * (0.92 + 0.14 * bscale))
    n_centers = max(220, n_centers)

    if region_gate is not None:
        gate = np.clip(region_gate.astype(np.float32), 0.0, 1.0)
        gate_soft = cv2.GaussianBlur(gate, (odd_int(max(7, int(min_side * 0.012))), odd_int(max(7, int(min_side * 0.012)))), 0)
        gate_wide = cv2.GaussianBlur(gate, (odd_int(max(15, int(min_side * 0.030))), odd_int(max(15, int(min_side * 0.030)))), 0)
        tex1 = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(8, 14)), octaves=4, persistence=0.58)
        tex2 = ridge_noise(h, w, rng, start_grid=int(rng.integers(18, 30)), octaves=3, persistence=0.52)
        prob_map = np.clip(0.68 * gate + 0.16 * gate_soft + 0.08 * gate_wide + 0.06 * gate * tex1 + 0.02 * gate * tex2, 0.0, 1.0)
        prob_map = np.power(prob_map, 1.34)
        prob = prob_map.reshape(-1).astype(np.float64)
        prob = None if prob.sum() <= 1e-8 else prob / prob.sum()
    else:
        prob = None

    inds = rng.choice(h * w, size=n_centers, replace=True, p=prob)

    for ind in inds:
        y, x = divmod(int(ind), w)
        r = rng.uniform(rmin, rmax) * min_side * (0.95 + 0.15 * bscale)
        angle = rng.uniform(0, 180)
        val = rng.uniform(0.44, 0.88)

        if rng.random() < 0.78:
            draw_small_fragment_stroke(core, x, y, r, angle, rng, value=val, brush_scale=brush_scale * 0.92)
        else:
            rx = r * rng.uniform(0.55, 1.50)
            ry = r * rng.uniform(0.45, 1.05)
            draw_rotated_soft_stamp(core, x, y, rx, ry, angle, val, rng)

        child_n = int(rng.integers(child_rng[0], child_rng[1]))
        for _ in range(child_n):
            aa = angle + rng.normal(0, 65)
            dist = r * rng.uniform(0.10, 1.45)
            xx = x + np.cos(np.deg2rad(aa)) * dist + rng.normal(0, r * 0.45)
            yy = y + np.sin(np.deg2rad(aa)) * dist + rng.normal(0, r * 0.45)
            rr = r * rng.uniform(0.14, 0.44)
            vv = val * rng.uniform(0.38, 0.86)
            if rng.random() < 0.84:
                draw_small_fragment_stroke(core, xx, yy, rr, aa, rng, value=vv, brush_scale=brush_scale * 0.88)
            else:
                draw_rotated_soft_stamp(core, xx, yy, rr * rng.uniform(0.85, 1.55), rr * rng.uniform(0.50, 1.05), aa + rng.normal(0, 16), vv, rng)

            if rng.random() < grand_prob:
                gg_n = int(rng.integers(1, 4))
                for _ in range(gg_n):
                    aaa = aa + rng.normal(0, 75)
                    ddd = rr * rng.uniform(0.25, 1.25)
                    xxx = xx + np.cos(np.deg2rad(aaa)) * ddd + rng.normal(0, rr * 0.38)
                    yyy = yy + np.sin(np.deg2rad(aaa)) * ddd + rng.normal(0, rr * 0.38)
                    rrr = rr * rng.uniform(0.12, 0.30)
                    draw_rotated_soft_stamp(core, xxx, yyy, rrr * rng.uniform(0.75, 1.60), rrr * rng.uniform(0.50, 1.00), aaa, vv * rng.uniform(0.35, 0.72), rng)

    # keep the core fragmented, not foggy
    k1 = odd_int(max(3, int(min_side * 0.0018)))
    k2 = odd_int(max(3, int(min_side * 0.0030)))
    core = cv2.GaussianBlur(core, (k1, k1), 0)
    core = cv2.GaussianBlur(core, (k2, k2), 0)
    core = soft_normalize01(core)

    tex1 = ridge_noise(h, w, rng, start_grid=int(rng.integers(22, 38)), octaves=4, persistence=0.50)
    tex2 = ridge_noise(h, w, rng, start_grid=int(rng.integers(32, 58)), octaves=3, persistence=0.48)
    tex3 = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(36, 64)), octaves=3, persistence=0.48)
    texture = soft_normalize01(0.44 * tex1 + 0.34 * tex2 + 0.22 * tex3)
    core = core * (0.80 + 0.44 * texture)

    # create many holes so the cloud looks like connected small cloudlets
    pore1 = ridge_noise(h, w, rng, start_grid=int(rng.integers(24, 40)), octaves=4, persistence=0.50)
    pore2 = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(18, 32)), octaves=3, persistence=0.54)
    pore = 0.62 * smoothstep(0.50, 0.86, pore1) + 0.38 * smoothstep(0.56, 0.88, pore2)
    edge = np.clip(core * (1.0 - core) * 4.0, 0.0, 1.0)
    core = np.clip(core - (0.18 * edge + 0.08 * core) * pore, 0.0, 1.0)

    # re-brighten cloudlet cores, keep broken edge shape
    core = smoothstep(0.08, 0.76, core)
    pinhole = ridge_noise(h, w, rng, start_grid=int(rng.integers(40, 72)), octaves=3, persistence=0.48)
    core = np.clip(core - 0.08 * smoothstep(0.70, 0.94, pinhole) * core, 0.0, 1.0)

    if region_gate is not None:
        gate = np.clip(region_gate.astype(np.float32), 0.0, 1.0)
        gate_soft = cv2.GaussianBlur(gate, (odd_int(max(9, int(min_side * 0.018))), odd_int(max(9, int(min_side * 0.018)))), 0)
        gate_wide = cv2.GaussianBlur(gate, (odd_int(max(17, int(min_side * 0.042))), odd_int(max(17, int(min_side * 0.042)))), 0)
        support = np.clip(0.56 * gate + 0.24 * gate_soft + 0.16 * gate_wide, 0.0, 1.0)
        core *= support

    # modest warp only
    core = warp_float_image(core, rng, strength=params["warp_strength"] * 0.10, grid=4)
    core = np.clip(core, 0.0, 1.0)
    return core


def make_dense_fragment_mass(
    h,
    w,
    rng,
    params,
    severity,
    region_gate=None,
    fragment_density=2.30,
    brush_scale=1.0,
):
    """
    Build thick cloud as many fragmented small cloudlets packed together,
    avoiding one stiff opaque patch.
    """
    mass = np.zeros((h, w), dtype=np.float32)
    min_side = min(h, w)
    fd = float(np.clip(fragment_density, 0.80, 3.40))
    bscale = float(np.clip(brush_scale, 0.75, 2.30))

    if severity == "light":
        low, high = 300, 520
        rmin, rmax = 0.0013, 0.0042
        child_rng = (4, 8)
        grand_prob = 0.24
        val_min, val_max = 0.28, 0.62
    elif severity == "medium":
        low, high = 620, 1040
        rmin, rmax = 0.0015, 0.0050
        child_rng = (6, 10)
        grand_prob = 0.36
        val_min, val_max = 0.30, 0.68
    else:
        low, high = 1180, 1900
        rmin, rmax = 0.0017, 0.0056
        child_rng = (8, 13)
        grand_prob = 0.50
        val_min, val_max = 0.32, 0.72

    area_scale = np.sqrt((h * w) / (800.0 * 800.0))
    n_centers = int(rng.integers(low, high) * area_scale * fd * (0.90 + 0.20 * bscale))
    n_centers = max(160, n_centers)

    if region_gate is not None:
        gate0 = np.clip(region_gate.astype(np.float32), 0.0, 1.0)
        gate1 = cv2.GaussianBlur(gate0, (odd_int(max(7, int(min_side * 0.018))), odd_int(max(7, int(min_side * 0.018)))), 0)
        gate2 = cv2.GaussianBlur(gate0, (odd_int(max(15, int(min_side * 0.040))), odd_int(max(15, int(min_side * 0.040)))), 0)
        noise1 = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(6, 11)), octaves=4, persistence=0.58)
        noise2 = ridge_noise(h, w, rng, start_grid=int(rng.integers(18, 30)), octaves=3, persistence=0.52)
        prob_map = np.clip(0.64 * gate0 + 0.18 * gate1 + 0.06 * gate2 * noise1 + 0.12 * gate0 * noise2, 0.0, 1.0)
        prob_map = np.power(prob_map, 1.32)
        prob = prob_map.reshape(-1).astype(np.float64)
        prob = None if prob.sum() <= 1e-8 else prob / prob.sum()
    else:
        prob = None

    inds = rng.choice(h * w, size=n_centers, replace=True, p=prob)

    for ind in inds:
        y, x = divmod(int(ind), w)
        r = rng.uniform(rmin, rmax) * min_side * (0.94 + 0.22 * bscale)
        angle = rng.uniform(0, 180)
        val = rng.uniform(val_min, val_max)

        draw_small_fragment_stroke(mass, x, y, r, angle, rng, value=val, brush_scale=brush_scale)

        child_n = int(rng.integers(child_rng[0], child_rng[1]))
        for _ in range(child_n):
            aa = angle + rng.normal(0, 64)
            dist = r * rng.uniform(0.10, 1.40)
            xx = x + np.cos(np.deg2rad(aa)) * dist + rng.normal(0, r * 0.44)
            yy = y + np.sin(np.deg2rad(aa)) * dist + rng.normal(0, r * 0.44)
            rr = r * rng.uniform(0.14, 0.42)
            vv = val * rng.uniform(0.28, 0.68)
            if rng.random() < 0.90:
                draw_small_fragment_stroke(mass, xx, yy, rr, aa, rng, value=vv, brush_scale=brush_scale * 0.98)
            else:
                draw_rotated_soft_stamp(mass, xx, yy, rr * rng.uniform(0.8, 1.3), rr * rng.uniform(0.5, 0.9), aa + rng.normal(0, 18), vv, rng)

            if rng.random() < grand_prob:
                gg_n = int(rng.integers(1, 4))
                for _ in range(gg_n):
                    aaa = aa + rng.normal(0, 70)
                    ddd = rr * rng.uniform(0.22, 1.40)
                    xxx = xx + np.cos(np.deg2rad(aaa)) * ddd + rng.normal(0, rr * 0.42)
                    yyy = yy + np.sin(np.deg2rad(aaa)) * ddd + rng.normal(0, rr * 0.42)
                    rrr = rr * rng.uniform(0.12, 0.34)
                    draw_small_fragment_stroke(mass, xxx, yyy, rrr, aaa, rng, value=vv * rng.uniform(0.40, 0.70), brush_scale=brush_scale * 0.96)

    k1 = odd_int(max(3, int(min_side * 0.0026)))
    k2 = odd_int(max(5, int(min_side * 0.0042)))
    mass = cv2.GaussianBlur(mass, (k1, k1), 0)
    mass = cv2.GaussianBlur(mass, (k2, k2), 0)
    mass = soft_normalize01(mass)
    mass = mass ** 0.96

    tex1 = ridge_noise(h, w, rng, start_grid=int(rng.integers(20, 34)), octaves=4, persistence=0.52)
    tex2 = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(28, 46)), octaves=3, persistence=0.50)
    tex3 = motion_blur_float(tex1, odd_int(int(rng.integers(7, 15))), rng.uniform(0, 180))
    texture = soft_normalize01(0.50 * tex1 + 0.18 * tex2 + 0.32 * tex3)
    mass = mass * (0.70 + 0.30 * texture)

    pore = ridge_noise(h, w, rng, start_grid=int(rng.integers(24, 42)), octaves=4, persistence=0.50)
    edge = np.clip(mass * (1.0 - mass) * 4.0, 0.0, 1.0)
    chips = smoothstep(0.50, 0.86, pore) * (0.28 * mass + 0.50 * edge)
    mass = np.clip(mass - 0.22 * chips, 0.0, 1.0)

    filament = motion_blur_float(edge * texture, odd_int(int(rng.integers(9, 17))), rng.uniform(0, 180))
    mass = np.clip(mass + 0.05 * soft_normalize01(filament) * edge, 0.0, 1.0)

    mass = warp_float_image(mass, rng, strength=params["warp_strength"] * 0.18, grid=4)

    if region_gate is not None:
        rg = np.clip(region_gate.astype(np.float32), 0.0, 1.0)
        rg_soft = cv2.GaussianBlur(rg, (odd_int(max(9, int(min_side * 0.020))), odd_int(max(9, int(min_side * 0.020)))), 0)
        rg_wide = cv2.GaussianBlur(rg, (odd_int(max(17, int(min_side * 0.050))), odd_int(max(17, int(min_side * 0.050)))), 0)
        rg_soft = normalize01(rg_soft)
        rg_wide = normalize01(rg_wide)
        edge_tex = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(10, 16)), octaves=3, persistence=0.56)
        support = np.clip(0.50 * rg + 0.24 * rg_soft + 0.22 * rg_wide * (0.70 + 0.30 * edge_tex), 0.0, 1.0)
        mass *= support

    k3 = odd_int(max(3, int(min_side * 0.0030)))
    mass = cv2.GaussianBlur(mass, (k3, k3), 0)
    mass = np.clip(mass, 0.0, 1.0)
    mass = np.minimum(mass, cv2.GaussianBlur(mass, (odd_int(max(5, int(min_side * 0.006))), odd_int(max(5, int(min_side * 0.006)))), 0) * 1.08)
    return np.clip(mass, 0.0, 1.0)


def add_outside_sparse_fragments(
    h,
    w,
    rng,
    params,
    severity,
    region_gate,
    fragment_density=2.30,
    outside_density=0.38,
    brush_scale=1.0,
):
    """
    Generate sparse small cloud fragments outside the main cloud region.

    Main cloud is still constrained by region_gate. This layer adds a small
    amount of peripheral and distant fragments so the result is not only one
    isolated irregular cloud patch.
    """
    frag = np.zeros((h, w), dtype=np.float32)
    min_side = min(h, w)

    outside_density = float(np.clip(outside_density, 0.0, 1.80))
    fd = float(np.clip(fragment_density, 0.80, 4.20))

    if outside_density <= 1e-6:
        return frag

    near_k = odd_int(max(11, int(min_side * 0.046)))
    far_k = odd_int(max(25, int(min_side * 0.135)))

    gate_near = cv2.GaussianBlur(region_gate.astype(np.float32), (near_k, near_k), 0)
    gate_far = cv2.GaussianBlur(region_gate.astype(np.float32), (far_k, far_k), 0)

    gate_near = normalize01(gate_near)
    gate_far = normalize01(gate_far)

    # Peripheral halo around the main cloud region.
    halo = np.clip(gate_far - 0.30 * gate_near, 0.0, 1.0)
    halo = normalize01(halo)

    # Keep outside fragments away from the core of the main cloud.
    outside_mask = 1.0 - smoothstep(0.20, 0.68, region_gate)
    outside_mask = np.clip(outside_mask, 0.0, 1.0)

    bg_noise = fractal_value_noise(
        h,
        w,
        rng,
        start_grid=int(rng.integers(5, 9)),
        octaves=4,
        persistence=0.58,
    )

    sparse_noise = ridge_noise(
        h,
        w,
        rng,
        start_grid=int(rng.integers(18, 32)),
        octaves=3,
        persistence=0.52,
    )

    # near halo: small fragments around the main cloud.
    # free_sparse: scattered fragments farther away from the main cloud.
    free_sparse = bg_noise * sparse_noise * outside_mask
    free_sparse = smoothstep(0.56, 0.88, free_sparse)

    prob_map = (
        0.62 * halo
        + 0.16 * bg_noise * outside_mask
        + 0.10 * sparse_noise * outside_mask
        + 0.34 * free_sparse
    )
    prob_map *= outside_mask
    prob_map = np.clip(prob_map, 0.0, 1.0)

    prob = prob_map.reshape(-1).astype(np.float64)
    prob_sum = prob.sum()
    if prob_sum <= 1e-8:
        return frag

    prob = prob / prob_sum

    area_scale = np.sqrt((h * w) / (800.0 * 800.0))

    if severity == "light":
        low, high = 90, 170
    elif severity == "medium":
        low, high = 160, 300
    else:
        low, high = 260, 520

    n = int(rng.integers(low, high) * area_scale * fd * outside_density)
    n = max(8, n)

    inds = rng.choice(h * w, size=n, replace=True, p=prob)

    for ind in inds:
        y, x = divmod(int(ind), w)

        r = rng.uniform(0.0020, 0.0076) * min_side * (0.90 + 0.22 * np.clip(brush_scale, 0.8, 2.0))
        angle = rng.uniform(0, 180)

        if rng.random() < 0.92:
            draw_small_fragment_stroke(
                frag,
                x,
                y,
                r,
                angle,
                rng,
                value=rng.uniform(0.52, 1.00),
                brush_scale=brush_scale,
            )
        else:
            rx = r * rng.uniform(0.65, 1.55)
            ry = r * rng.uniform(0.45, 1.15)

            cv2.ellipse(
                frag,
                (int(x), int(y)),
                (max(1, int(rx)), max(1, int(ry))),
                angle,
                0,
                360,
                float(rng.uniform(0.52, 1.00)),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        if rng.random() < 0.58:
            child_n = int(rng.integers(1, 4))
            for _ in range(child_n):
                xx = int(np.clip(x + rng.normal(0, r * 1.8), 0, w - 1))
                yy = int(np.clip(y + rng.normal(0, r * 1.8), 0, h - 1))
                rr = r * rng.uniform(0.30, 0.70)

                draw_small_fragment_stroke(
                    frag,
                    xx,
                    yy,
                    rr,
                    angle + rng.normal(0, 50),
                    rng,
                    value=rng.uniform(0.36, 0.78),
                    brush_scale=brush_scale,
                )

    k = odd_int(max(3, int(min_side * 0.0038)))
    frag = cv2.GaussianBlur(frag, (k, k), 0)

    tex = ridge_noise(
        h,
        w,
        rng,
        start_grid=int(rng.integers(22, 40)),
        octaves=3,
        persistence=0.52,
    )

    frag = smoothstep(0.035, 0.86, frag) * (0.78 + 0.46 * tex)
    frag *= outside_mask

    return np.clip(frag, 0.0, 1.0)




def add_free_sparse_clouds(
    h,
    w,
    rng,
    severity,
    fragment_density=2.30,
    free_density=0.35,
    brush_scale=1.0,
):
    """
    Generate very sparse free clouds over the whole image.

    This layer is independent from region_gate. It creates distant thin fragments,
    so the final cloud distribution is not only main cloud + halo.
    """
    frag = np.zeros((h, w), dtype=np.float32)
    min_side = min(h, w)

    free_density = float(np.clip(free_density, 0.0, 1.60))
    fd = float(np.clip(fragment_density, 0.80, 4.20))

    if free_density <= 1e-6:
        return frag

    bg = fractal_value_noise(
        h,
        w,
        rng,
        start_grid=int(rng.integers(5, 9)),
        octaves=4,
        persistence=0.58,
    )

    rid = ridge_noise(
        h,
        w,
        rng,
        start_grid=int(rng.integers(18, 34)),
        octaves=3,
        persistence=0.52,
    )

    prob_map = smoothstep(0.56, 0.90, bg * rid)
    prob_map = prob_map * (0.65 + 0.35 * bg)
    prob_map = np.clip(prob_map, 0.0, 1.0)

    prob = prob_map.reshape(-1).astype(np.float64)
    prob_sum = prob.sum()
    if prob_sum <= 1e-8:
        return frag
    prob = prob / prob_sum

    area_scale = np.sqrt((h * w) / (800.0 * 800.0))

    if severity == "light":
        low, high = 30, 70
    elif severity == "medium":
        low, high = 55, 110
    else:
        low, high = 90, 180

    n = int(rng.integers(low, high) * area_scale * fd * free_density)
    n = max(5, n)

    inds = rng.choice(h * w, size=n, replace=True, p=prob)

    for ind in inds:
        y, x = divmod(int(ind), w)
        r = rng.uniform(0.0018, 0.0068) * min_side * (0.90 + 0.18 * np.clip(brush_scale, 0.8, 2.0))
        angle = rng.uniform(0, 180)

        draw_small_fragment_stroke(
            frag,
            x,
            y,
            r,
            angle,
            rng,
            value=rng.uniform(0.36, 0.78),
            brush_scale=brush_scale,
        )

        if rng.random() < 0.48:
            child_n = int(rng.integers(1, 3))
            for _ in range(child_n):
                xx = int(np.clip(x + rng.normal(0, r * 2.2), 0, w - 1))
                yy = int(np.clip(y + rng.normal(0, r * 2.2), 0, h - 1))
                rr = r * rng.uniform(0.25, 0.58)
                draw_small_fragment_stroke(
                    frag,
                    xx,
                    yy,
                    rr,
                    angle + rng.normal(0, 55),
                    rng,
                    value=rng.uniform(0.28, 0.68),
                    brush_scale=brush_scale,
                )

    k = odd_int(max(3, int(min_side * 0.0030)))
    frag = cv2.GaussianBlur(frag, (k, k), 0)

    tex = ridge_noise(
        h,
        w,
        rng,
        start_grid=int(rng.integers(24, 44)),
        octaves=3,
        persistence=0.50,
    )

    frag = smoothstep(0.035, 0.84, frag) * (0.76 + 0.42 * tex)
    return np.clip(frag, 0.0, 1.0)




def make_stratiform_veil_layer(
    h,
    w,
    rng,
    severity,
    veil_density=0.45,
    veil_alpha=0.12,
):
    """
    Generate a broad, thin, stratiform veil cloud layer.

    This layer is designed to mimic wide-area thin SCCOS-like cloud sheets:
    broad coverage, low opacity, soft transitions, and many internal holes.
    It is intentionally independent from region_gate, so it reduces the
    "limited-region" feeling caused by only using main/secondary gates.
    """
    veil_density = float(np.clip(veil_density, 0.0, 1.50))
    veil_alpha = float(np.clip(veil_alpha, 0.0, 0.35))

    if veil_density <= 1e-6 or veil_alpha <= 1e-6:
        return np.zeros((h, w), dtype=np.float32)

    min_side = min(h, w)

    base = fractal_value_noise(
        h,
        w,
        rng,
        start_grid=int(rng.integers(2, 5)),
        octaves=5,
        persistence=0.62,
    )

    ridge = ridge_noise(
        h,
        w,
        rng,
        start_grid=int(rng.integers(7, 13)),
        octaves=4,
        persistence=0.55,
    )

    sheet = normalize01(0.70 * base + 0.30 * ridge)

    if severity == "light":
        cover = rng.uniform(0.12, 0.24) * veil_density
        alpha_scale = rng.uniform(0.45, 0.70)
    elif severity == "medium":
        cover = rng.uniform(0.22, 0.42) * veil_density
        alpha_scale = rng.uniform(0.70, 0.95)
    else:
        cover = rng.uniform(0.32, 0.58) * veil_density
        alpha_scale = rng.uniform(0.85, 1.15)

    cover = float(np.clip(cover, 0.04, 0.72))

    thr = np.quantile(sheet, 1.0 - cover)
    sheet = smoothstep(thr - 0.080, thr + 0.190, sheet)

    # Elongate and soften the sheet so it does not look like isolated blobs.
    blur_angle = rng.uniform(0, 180)
    sheet = motion_blur_float(
        sheet,
        odd_int(int(rng.integers(35, 75))),
        blur_angle,
    )
    sheet = soft_normalize01(sheet)

    # Add holes and torn transparent areas.
    holes_large = fractal_value_noise(
        h,
        w,
        rng,
        start_grid=int(rng.integers(4, 8)),
        octaves=4,
        persistence=0.60,
    )
    holes_fine = ridge_noise(
        h,
        w,
        rng,
        start_grid=int(rng.integers(18, 34)),
        octaves=3,
        persistence=0.52,
    )

    hole_mask = 0.70 * smoothstep(0.55, 0.88, holes_large) + 0.30 * smoothstep(0.62, 0.92, holes_fine)
    sheet = sheet * (1.0 - 0.42 * hole_mask * smoothstep(0.10, 0.90, sheet))

    tex = ridge_noise(
        h,
        w,
        rng,
        start_grid=int(rng.integers(22, 42)),
        octaves=3,
        persistence=0.50,
    )
    sheet = sheet * (0.68 + 0.46 * tex)

    # Soft final feathering.
    k = odd_int(max(5, int(min_side * 0.010)))
    sheet = cv2.GaussianBlur(sheet, (k, k), 0)

    sheet = np.clip(sheet, 0.0, 1.0)

    # Keep veil thin. The visible "thickness" should come from broad coverage,
    # not from high opacity.
    veil = np.clip(sheet * veil_alpha * alpha_scale, 0.0, 0.32)

    return veil



def make_cirrus_wisp_layer(
    h,
    w,
    rng,
    severity,
    cirrus_density=0.38,
    cirrus_alpha=0.10,
):
    """
    Generate directional wispy cirrus-like streaks.

    Unlike a stratiform veil, this layer is composed of elongated broken strokes
    and torn filaments, so it should not look like a continuous fog curtain.
    """
    cirrus_density = float(np.clip(cirrus_density, 0.0, 1.50))
    cirrus_alpha = float(np.clip(cirrus_alpha, 0.0, 0.28))

    if cirrus_density <= 1e-6 or cirrus_alpha <= 1e-6:
        return np.zeros((h, w), dtype=np.float32)

    min_side = min(h, w)

    base = fractal_value_noise(
        h, w, rng,
        start_grid=int(rng.integers(4, 9)),
        octaves=4,
        persistence=0.60,
    )
    fine = ridge_noise(
        h, w, rng,
        start_grid=int(rng.integers(18, 36)),
        octaves=4,
        persistence=0.52,
    )

    # Directional stretching to create cloud filaments rather than a sheet.
    ang1 = rng.uniform(0, 180)
    ang2 = ang1 + rng.uniform(-22, 22)
    mb1 = motion_blur_float(base, odd_int(int(rng.integers(31, 71))), ang1)
    mb2 = motion_blur_float(fine, odd_int(int(rng.integers(17, 43))), ang2)
    field = soft_normalize01(0.56 * mb1 + 0.26 * mb2 + 0.18 * fine)

    if severity == "light":
        cover = rng.uniform(0.05, 0.12) * cirrus_density
        alpha_scale = rng.uniform(0.55, 0.78)
    elif severity == "medium":
        cover = rng.uniform(0.08, 0.18) * cirrus_density
        alpha_scale = rng.uniform(0.72, 0.96)
    else:
        cover = rng.uniform(0.12, 0.24) * cirrus_density
        alpha_scale = rng.uniform(0.86, 1.10)

    cover = float(np.clip(cover, 0.02, 0.32))
    thr = np.quantile(field, 1.0 - cover)
    wisps = smoothstep(thr - 0.05, thr + 0.10, field)

    # Tear the wisps so they are broken, airy and non-uniform.
    holes = fractal_value_noise(
        h, w, rng,
        start_grid=int(rng.integers(10, 22)),
        octaves=3,
        persistence=0.58,
    )
    tears = ridge_noise(
        h, w, rng,
        start_grid=int(rng.integers(22, 44)),
        octaves=3,
        persistence=0.50,
    )
    gap = 0.55 * smoothstep(0.48, 0.84, holes) + 0.45 * smoothstep(0.60, 0.90, tears)
    wisps = wisps * (1.0 - 0.52 * gap)

    # Create variable thickness and localized puffs along the streaks.
    micro = ridge_noise(
        h, w, rng,
        start_grid=int(rng.integers(28, 52)),
        octaves=3,
        persistence=0.48,
    )
    wisps = wisps * (0.70 + 0.38 * micro)

    # Soft but still filament-like.
    k = odd_int(max(5, int(min_side * 0.010)))
    wisps = cv2.GaussianBlur(wisps, (k, k), 0)
    wisps = np.clip(wisps, 0.0, 1.0)
    wisps = np.clip(wisps * cirrus_alpha * alpha_scale, 0.0, 0.24)
    return wisps



def build_feather_support(base_gate, rng):
    """
    Turn a hard limited region into a softer multi-scale support with outside spill.
    This is used to reduce the visible boundary of the constrained region.
    """
    base = np.clip(base_gate.astype(np.float32), 0.0, 1.0)
    h, w = base.shape
    min_side = min(h, w)

    soft = cv2.GaussianBlur(base, (odd_int(max(7, int(min_side * 0.026))), odd_int(max(7, int(min_side * 0.026)))), 0)
    wide = cv2.GaussianBlur(base, (odd_int(max(13, int(min_side * 0.055))), odd_int(max(13, int(min_side * 0.055)))), 0)
    far = cv2.GaussianBlur(base, (odd_int(max(21, int(min_side * 0.100))), odd_int(max(21, int(min_side * 0.100)))), 0)

    soft = normalize01(soft)
    wide = normalize01(wide)
    far = normalize01(far)

    n1 = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(6, 11)), octaves=4, persistence=0.58)
    n2 = ridge_noise(h, w, rng, start_grid=int(rng.integers(16, 28)), octaves=4, persistence=0.52)
    n3 = motion_blur_float(n1, odd_int(int(rng.integers(9, 17))), rng.uniform(0, 180))
    tex = soft_normalize01(0.44 * n1 + 0.28 * n2 + 0.28 * n3)

    outside_ring = np.clip(far - 0.62 * soft, 0.0, 1.0)
    outside_ring *= (1.0 - smoothstep(0.10, 0.64, base))
    outside_ring *= (0.58 + 0.42 * tex)

    support = np.clip(
        0.62 * base
        + 0.20 * soft
        + 0.10 * wide * (0.72 + 0.28 * tex)
        + 0.05 * outside_ring,
        0.0,
        1.0,
    )

    edge = np.clip(support * (1.0 - support) * 4.0, 0.0, 1.0)
    chips = smoothstep(0.54, 0.88, n2) * edge
    support = np.clip(support - 0.10 * chips, 0.0, 1.0)
    support = cv2.GaussianBlur(support, (odd_int(max(5, int(min_side * 0.010))), odd_int(max(5, int(min_side * 0.010)))), 0)

    return np.clip(base, 0.0, 1.0), soft, wide, far, np.clip(outside_ring, 0.0, 1.0), np.clip(support, 0.0, 1.0), tex



def paint_sccos_piece_cloudlet(body, core, x, y, rx, ry, angle_deg, rng,
                               value=1.0, core_value=1.0, hole_strength=0.18,
                               roughness=0.50):
    """
    Paint one SCCOS-style cloudlet. This is intentionally local:
    a small/medium cloud piece with a bright core, rough edge and small holes.
    v25 uses many of these cloudlets to assemble the main cloud, instead of
    filling one large region with texture.
    """
    h, w = body.shape[:2]
    pad = int(max(rx, ry) * 3.2) + 6
    x0 = max(0, int(x) - pad)
    x1 = min(w, int(x) + pad + 1)
    y0 = max(0, int(y) - pad)
    y1 = min(h, int(y) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx = xx - float(x)
    dy = yy - float(y)
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy
    rx = max(float(rx), 1.0)
    ry = max(float(ry), 1.0)

    d = (xr / rx) ** 2 + (yr / ry) ** 2
    g = np.exp(-0.5 * d)

    # Local cloudlet texture. Low-res random field gives cloudlet roughness.
    ph, pw = g.shape
    gh = max(3, min(10, ph // 7 + 2))
    gw = max(3, min(10, pw // 7 + 2))
    n1 = rng.random((gh, gw)).astype(np.float32)
    n1 = cv2.resize(n1, (pw, ph), interpolation=cv2.INTER_CUBIC)
    n1 = cv2.GaussianBlur(n1, (3, 3), 0)

    gh2 = max(4, min(18, ph // 4 + 2))
    gw2 = max(4, min(18, pw // 4 + 2))
    n2 = rng.random((gh2, gw2)).astype(np.float32)
    n2 = cv2.resize(n2, (pw, ph), interpolation=cv2.INTER_CUBIC)
    n2 = cv2.GaussianBlur(n2, (3, 3), 0)

    tex = np.clip(0.62 + roughness * (0.52 * n1 + 0.34 * n2), 0.0, 1.45)
    stamp = np.clip(g * tex, 0.0, 1.0)

    # Small holes/eroded bits. This prevents cloudlets merging into flat white blobs.
    holes = smoothstep(0.62, 0.92, n2) * (0.35 + 0.65 * np.clip(stamp * (1.0 - stamp) * 4.0, 0.0, 1.0))
    stamp = np.clip(stamp * (1.0 - hole_strength * holes), 0.0, 1.0)

    # Body has fuzzy piece; core is smaller and brighter.
    core_g = np.exp(-0.5 * ((xr / (rx * 0.58)) ** 2 + (yr / (ry * 0.58)) ** 2))
    core_stamp = np.clip(core_g * (0.72 + 0.36 * n1), 0.0, 1.0)
    core_stamp = np.clip(core_stamp * (1.0 - 0.12 * holes), 0.0, 1.0)

    body[y0:y1, x0:x1] = np.maximum(body[y0:y1, x0:x1], stamp * float(value))
    core[y0:y1, x0:x1] = np.maximum(core[y0:y1, x0:x1], core_stamp * float(core_value))


def paint_piece_group(body, core, cx, cy, rng, min_side, severity,
                      group_scale=1.0, angle=90.0, fragment_density=1.0,
                      brightness=1.0, loose=False):
    """Paint a cluster of palm/apple-sized cloudlets around one group center."""
    group_scale = float(group_scale)
    fd = float(np.clip(fragment_density, 0.6, 4.8))
    if severity == "light":
        n_big = int(rng.integers(1, 3))
        n_mid = int(rng.integers(4, 8))
        n_small = int(rng.integers(8, 16))
    elif severity == "medium":
        n_big = int(rng.integers(2, 4))
        n_mid = int(rng.integers(7, 13))
        n_small = int(rng.integers(14, 28))
    else:
        n_big = int(rng.integers(2, 5))
        n_mid = int(rng.integers(10, 18))
        n_small = int(rng.integers(20, 38))

    n_big = max(1, int(n_big * (0.75 + 0.10 * fd)))
    n_mid = max(2, int(n_mid * (0.85 + 0.16 * fd)))
    n_small = max(5, int(n_small * (0.82 + 0.14 * fd)))

    theta = np.deg2rad(angle + rng.normal(0, 14))
    dx, dy = np.cos(theta), np.sin(theta)
    nx, ny = -dy, dx
    spread_x = min_side * (0.035 if not loose else 0.060) * group_scale
    spread_y = min_side * (0.028 if not loose else 0.048) * group_scale

    def sample_pos(scale=1.0):
        u = rng.normal(0, spread_x * scale)
        v = rng.normal(0, spread_y * scale)
        return cx + dx * u + nx * v, cy + dy * u + ny * v

    # big/apple pieces
    for _ in range(n_big):
        x, y = sample_pos(0.85)
        rx = min_side * rng.uniform(0.024, 0.050) * group_scale
        ry = min_side * rng.uniform(0.018, 0.040) * group_scale
        val = rng.uniform(0.62, 0.94) * brightness
        paint_sccos_piece_cloudlet(body, core, x, y, rx, ry, angle + rng.normal(0, 38), rng,
                                   value=val, core_value=val * rng.uniform(0.88, 1.12),
                                   hole_strength=rng.uniform(0.12, 0.22), roughness=rng.uniform(0.35, 0.62))

    # medium/palm pieces
    for _ in range(n_mid):
        x, y = sample_pos(1.15)
        rx = min_side * rng.uniform(0.014, 0.034) * group_scale
        ry = min_side * rng.uniform(0.010, 0.028) * group_scale
        val = rng.uniform(0.48, 0.86) * brightness
        paint_sccos_piece_cloudlet(body, core, x, y, rx, ry, angle + rng.normal(0, 50), rng,
                                   value=val, core_value=val * rng.uniform(0.76, 1.05),
                                   hole_strength=rng.uniform(0.14, 0.28), roughness=rng.uniform(0.44, 0.72))

    # tiny fragments around the group
    for _ in range(n_small):
        x, y = sample_pos(1.55)
        rx = min_side * rng.uniform(0.0055, 0.0175) * group_scale
        ry = min_side * rng.uniform(0.0045, 0.0145) * group_scale
        val = rng.uniform(0.24, 0.62) * brightness
        paint_sccos_piece_cloudlet(body, core, x, y, rx, ry, angle + rng.normal(0, 75), rng,
                                   value=val, core_value=val * rng.uniform(0.54, 0.88),
                                   hole_strength=rng.uniform(0.18, 0.34), roughness=rng.uniform(0.55, 0.88))


def make_sccos_cloudlet_assemblage(h, w, rng, severity, params,
                                   max_cloud_area=0.50,
                                   fragment_density=3.80,
                                   outside_density=0.33,
                                   outside_alpha=0.15,
                                   secondary_density=0.26,
                                   free_density=0.12,
                                   transition_strength=0.17,
                                   brush_scale=0.95,
                                   min_main_cloud_patches=1,
                                   max_main_cloud_patches=4,
                                   main_patch_min_sep_scale=0.18):
    """
    v25.8 main generator:
    Build clouds as an assemblage of differently sized SCCOS-style cloudlets.
    For each output image, randomly choose the total number of main cloud
    patches from 1 to 4, and keep a modest distance between main cloud patches
    so separate main clouds remain visually distinguishable.
    """
    min_side = min(h, w)
    area_scale = np.sqrt((h * w) / (800.0 * 800.0))
    fd = float(np.clip(fragment_density, 1.0, 4.8))
    bscale = float(np.clip(brush_scale, 0.70, 1.50))
    min_main_cloud_patches = int(np.clip(min_main_cloud_patches, 1, 4))
    max_main_cloud_patches = int(np.clip(max_main_cloud_patches, min_main_cloud_patches, 4))
    total_main_cloud_patches = int(rng.integers(min_main_cloud_patches, max_main_cloud_patches + 1))
    extra_main_patch_count = max(0, total_main_cloud_patches - 1)
    main_patch_min_sep_scale = float(np.clip(main_patch_min_sep_scale, 0.10, 0.35))

    body = np.zeros((h, w), dtype=np.float32)
    core = np.zeros((h, w), dtype=np.float32)

    # Main top-origin chain of groups. Each group is a cloudlet cluster.
    # v25.8: keep the longer main chain and support richer main masses.
    extra_main_groups = int(rng.integers(4, 6))
    if severity == "light":
        n_groups = int(rng.integers(2, 4)) + extra_main_groups
        chain_len = min_side * rng.uniform(0.22, 0.40)
    elif severity == "medium":
        n_groups = int(rng.integers(3, 5)) + extra_main_groups
        chain_len = min_side * rng.uniform(0.30, 0.50)
    else:
        n_groups = int(rng.integers(4, 7)) + extra_main_groups
        chain_len = min_side * rng.uniform(0.38, 0.62)

    # Main cloud starts from top/upper side, but is made of discrete groups.
    start_x = w * rng.uniform(0.20, 0.80)
    start_y = h * rng.uniform(-0.04, 0.10)
    angle = 90.0 + rng.uniform(-34, 34)
    theta = np.deg2rad(angle)
    dir_x, dir_y = np.cos(theta), np.sin(theta)
    norm_x, norm_y = -dir_y, dir_x
    curve_phase = rng.uniform(0, 2 * np.pi)
    curve_amp = min_side * rng.uniform(0.020, 0.075)

    group_centers = []
    for i in range(n_groups):
        u = i / max(n_groups - 1, 1)
        along = chain_len * (0.10 + 0.92 * u)
        curve = np.sin(u * np.pi * rng.uniform(0.8, 1.5) + curve_phase) * curve_amp
        jitter_n = rng.normal(0, min_side * 0.026)
        jitter_d = rng.normal(0, min_side * 0.020)
        cx = start_x + dir_x * along + norm_x * (curve + jitter_n)
        cy = start_y + dir_y * along + norm_y * (curve + jitter_n) + jitter_d
        group_centers.append((cx, cy))

    for idx, (cx, cy) in enumerate(group_centers):
        u = idx / max(len(group_centers) - 1, 1)
        # middle groups slightly larger, ends smaller; avoids one giant rectangle/blob.
        local_scale = bscale * (0.78 + 0.36 * np.sin(np.pi * np.clip(u, 0, 1))) * rng.uniform(0.88, 1.18)
        paint_piece_group(body, core, cx, cy, rng, min_side, severity,
                          group_scale=local_scale, angle=angle + rng.normal(0, 22),
                          fragment_density=fd, brightness=rng.uniform(0.90, 1.12), loose=False)

    primary_chain_centers = list(group_centers)
    primary_main_center = (
        float(np.mean([c[0] for c in primary_chain_centers])),
        float(np.mean([c[1] for c in primary_chain_centers])),
    )
    main_patch_centers = [primary_main_center]

    # v25.8: add a few detached puffy groups around the primary chain.
    # When total_main_cloud_patches == 1, keep them slightly fewer so the image
    # still reads as one main cloud rather than multiple separate main pieces.
    if total_main_cloud_patches <= 1:
        extra_detached_groups = int(rng.integers(1, 3))
    elif total_main_cloud_patches == 2:
        extra_detached_groups = int(rng.integers(2, 4))
    else:
        extra_detached_groups = int(rng.integers(3, 5))
    for _ in range(extra_detached_groups):
        base_cx, base_cy = group_centers[int(rng.integers(0, len(group_centers)))]
        dist = min_side * rng.uniform(0.12, 0.30)
        aa = angle + rng.choice([-1, 1]) * rng.uniform(36, 120)
        cx = base_cx + np.cos(np.deg2rad(aa)) * dist + rng.normal(0, min_side * 0.022)
        cy = base_cy + np.sin(np.deg2rad(aa)) * dist + rng.normal(0, min_side * 0.022)
        det_scale = bscale * rng.uniform(0.62, 0.92)
        paint_piece_group(body, core, cx, cy, rng, min_side, severity,
                          group_scale=det_scale, angle=angle + rng.normal(0, 30),
                          fragment_density=fd * 0.86, brightness=rng.uniform(0.82, 1.00), loose=True)
        group_centers.append((cx, cy))

    # v25.8: randomly add 0–3 extra MAIN cloud patches so each image can have
    # 1, 2, 3, or 4 total main cloud pieces. Keep a modest distance between
    # main cloud patches so separate pieces remain visually distinguishable.
    patch_side_candidates = [-1, 1, -1, 1]
    rng.shuffle(patch_side_candidates)
    if total_main_cloud_patches <= 2:
        desired_min_patch_sep = min_side * (main_patch_min_sep_scale + 0.03)
    elif total_main_cloud_patches == 3:
        desired_min_patch_sep = min_side * main_patch_min_sep_scale
    else:
        desired_min_patch_sep = min_side * max(0.14, main_patch_min_sep_scale - 0.02)

    for patch_idx in range(extra_main_patch_count):
        patch_side = patch_side_candidates[patch_idx % len(patch_side_candidates)]
        best_candidate = None
        best_candidate_score = -1.0
        for _attempt in range(18):
            extra_patch_anchor = primary_chain_centers[int(rng.integers(0, len(primary_chain_centers)))]
            patch_angle = angle + patch_side * rng.uniform(60, 145) + rng.normal(0, 10)
            # Keep separate main patches a touch farther away than v25.7.
            patch_dist = min_side * rng.uniform(0.24 + 0.01 * patch_idx, 0.42 + 0.02 * patch_idx)
            cand_x = extra_patch_anchor[0] + np.cos(np.deg2rad(patch_angle)) * patch_dist + rng.normal(0, min_side * 0.026)
            cand_y = extra_patch_anchor[1] + np.sin(np.deg2rad(patch_angle)) * patch_dist + rng.normal(0, min_side * 0.026)
            min_sep = min(np.hypot(cand_x - cx0, cand_y - cy0) for cx0, cy0 in main_patch_centers)
            candidate = (extra_patch_anchor, patch_angle, cand_x, cand_y, min_sep)
            if min_sep > best_candidate_score:
                best_candidate = candidate
                best_candidate_score = min_sep
            if min_sep >= desired_min_patch_sep:
                best_candidate = candidate
                break

        extra_patch_anchor, patch_angle, patch_cx0, patch_cy0, chosen_sep = best_candidate
        patch_chain_angle = angle + rng.normal(0, 52)
        ptheta = np.deg2rad(patch_chain_angle)
        pdx, pdy = np.cos(ptheta), np.sin(ptheta)
        pnx, pny = -pdy, pdx
        patch_groups = int(rng.integers(2, 5))
        patch_len = min_side * rng.uniform(0.08, 0.22)
        patch_centers = []
        for j in range(patch_groups):
            pu = j / max(patch_groups - 1, 1) - 0.5
            px = patch_cx0 + pdx * patch_len * pu + pnx * rng.normal(0, min_side * 0.032)
            py = patch_cy0 + pdy * patch_len * pu + pny * rng.normal(0, min_side * 0.032)
            patch_scale = bscale * rng.uniform(0.72, 1.05)
            paint_piece_group(body, core, px, py, rng, min_side, severity,
                              group_scale=patch_scale, angle=patch_chain_angle + rng.normal(0, 28),
                              fragment_density=fd * 0.94, brightness=rng.uniform(0.84, 1.03), loose=True)
            patch_centers.append((px, py))
            group_centers.append((px, py))

        patch_center_mean = (
            float(np.mean([p[0] for p in patch_centers])),
            float(np.mean([p[1] for p in patch_centers])),
        )
        main_patch_centers.append(patch_center_mean)

        # Add a small amount of broken bridge chips toward each new patch, but
        # keep the bridge lighter than before so the gap remains visible.
        bridge_n = int(rng.integers(2, 6) * (0.78 + 0.32 * np.clip(transition_strength, 0.0, 1.0)))
        nearest_patch = patch_centers[int(rng.integers(0, len(patch_centers)))]
        for _ in range(max(0, bridge_n)):
            bu = rng.uniform(0.26, 0.70)
            bx = extra_patch_anchor[0] * (1.0 - bu) + nearest_patch[0] * bu + rng.normal(0, min_side * 0.026)
            by = extra_patch_anchor[1] * (1.0 - bu) + nearest_patch[1] * bu + rng.normal(0, min_side * 0.026)
            brx = min_side * rng.uniform(0.0052, 0.0150) * bscale
            bry = min_side * rng.uniform(0.0042, 0.0130) * bscale
            bval = rng.uniform(0.12, 0.30)
            paint_sccos_piece_cloudlet(body, core, bx, by, brx, bry, patch_chain_angle + rng.normal(0, 48), rng,
                                       value=bval, core_value=bval * 0.50,
                                       hole_strength=0.30, roughness=0.84)

    print(f">>> random total main cloud patches = {total_main_cloud_patches}", flush=True)
    print(f">>> desired main patch min sep = {desired_min_patch_sep / max(min_side, 1):.3f} * min_side", flush=True)

    # Add a few secondary small pieces near the main cloud, not giant secondary regions.
    # v25.8 slightly raises the default so the main mass has richer nearby pieces.
    sec_density = float(np.clip(secondary_density, 0.0, 1.0))
    if sec_density > 0:
        if severity == "light":
            n_sec = int(rng.integers(0, 2))
        elif severity == "medium":
            n_sec = int(rng.integers(1, 3))
        else:
            n_sec = int(rng.integers(1, 4))
        n_sec = int(round(n_sec * (1.00 + sec_density)))
        for _ in range(max(0, n_sec)):
            base_cx, base_cy = group_centers[int(rng.integers(0, len(group_centers)))]
            dist = min_side * rng.uniform(0.10, 0.26)
            aa = angle + rng.choice([-1, 1]) * rng.uniform(45, 115)
            cx = base_cx + np.cos(np.deg2rad(aa)) * dist + rng.normal(0, min_side * 0.025)
            cy = base_cy + np.sin(np.deg2rad(aa)) * dist + rng.normal(0, min_side * 0.025)
            paint_piece_group(body, core, cx, cy, rng, min_side, severity,
                              group_scale=bscale * rng.uniform(0.48, 0.78),
                              angle=angle + rng.normal(0, 45),
                              fragment_density=fd * 0.70,
                              brightness=rng.uniform(0.58, 0.82), loose=True)

    # Peripheral sparse chips: visible, but not a fog curtain.
    # v25.8: slightly more chips around the primary cloud.
    out_d = float(np.clip(outside_density, 0.0, 1.8))
    outside_alpha_scale = float(np.clip(outside_alpha / 0.15, 0.65, 1.35))
    n_out = int((rng.integers(30, 74) if severity == "light" else rng.integers(56, 130) if severity == "medium" else rng.integers(84, 176)) * out_d * area_scale * 1.13)
    for _ in range(max(0, n_out)):
        base_cx, base_cy = group_centers[int(rng.integers(0, len(group_centers)))]
        # annulus around main cloud
        dist = min_side * rng.uniform(0.14, 0.50)
        aa = rng.uniform(0, 360)
        x = base_cx + np.cos(np.deg2rad(aa)) * dist + rng.normal(0, min_side * 0.035)
        y = base_cy + np.sin(np.deg2rad(aa)) * dist + rng.normal(0, min_side * 0.035)
        rx = min_side * rng.uniform(0.005, 0.020) * bscale
        ry = min_side * rng.uniform(0.004, 0.016) * bscale
        val = rng.uniform(0.13, 0.40) * outside_alpha_scale
        paint_sccos_piece_cloudlet(body, core, x, y, rx, ry, rng.uniform(0, 180), rng,
                                   value=val, core_value=val * 0.55,
                                   hole_strength=rng.uniform(0.20, 0.38), roughness=rng.uniform(0.58, 0.95))

    # Few global tiny fragments, still sparse but a bit more present in v25.8.
    free_d = float(np.clip(free_density, 0.0, 1.2))
    n_free = int((rng.integers(14, 44) if severity == "light" else rng.integers(32, 76) if severity == "medium" else rng.integers(48, 108)) * free_d * area_scale * 1.15)
    for _ in range(max(0, n_free)):
        x = rng.uniform(0, w)
        y = rng.uniform(0, h)
        rx = min_side * rng.uniform(0.004, 0.014) * bscale
        ry = min_side * rng.uniform(0.003, 0.011) * bscale
        val = rng.uniform(0.08, 0.24)
        paint_sccos_piece_cloudlet(body, core, x, y, rx, ry, rng.uniform(0, 180), rng,
                                   value=val, core_value=val * 0.45,
                                   hole_strength=0.32, roughness=0.88)

    # Very light transition: a few pieces between groups, not a whole mask.
    trans = float(np.clip(transition_strength, 0.0, 1.0))
    if len(group_centers) >= 2 and trans > 0:
        for a, b in zip(group_centers[:-1], group_centers[1:]):
            n_bridge = int(rng.integers(4, 10) * trans * (0.8 + 0.2 * fd))
            for _ in range(max(0, n_bridge)):
                u = rng.uniform(0.15, 0.85)
                x = a[0] * (1-u) + b[0] * u + rng.normal(0, min_side * 0.028)
                y = a[1] * (1-u) + b[1] * u + rng.normal(0, min_side * 0.028)
                rx = min_side * rng.uniform(0.006, 0.018) * bscale
                ry = min_side * rng.uniform(0.005, 0.015) * bscale
                val = rng.uniform(0.16, 0.38)
                paint_sccos_piece_cloudlet(body, core, x, y, rx, ry, angle + rng.normal(0, 60), rng,
                                           value=val, core_value=val * 0.55,
                                           hole_strength=0.30, roughness=0.78)

    # Local softening only; avoid broad blur.
    k_body = odd_int(max(3, int(min_side * 0.0028)))
    k_core = odd_int(max(3, int(min_side * 0.0018)))
    body = cv2.GaussianBlur(body, (k_body, k_body), 0)
    core = cv2.GaussianBlur(core, (k_core, k_core), 0)

    # Break up accidental broad merges with global ridge holes.
    ridge = ridge_noise(h, w, rng, start_grid=int(rng.integers(20, 36)), octaves=4, persistence=0.50)
    cellular = ridge_noise(h, w, rng, start_grid=int(rng.integers(36, 62)), octaves=3, persistence=0.48)
    edge = np.clip(body * (1.0 - body) * 4.0, 0.0, 1.0)
    cut = smoothstep(0.60, 0.90, ridge) * (0.18 * body + 0.34 * edge)
    chip = smoothstep(0.66, 0.92, cellular) * (0.10 * body + 0.20 * edge)
    body = np.clip(body - cut - chip, 0.0, 1.0)
    core = np.clip(core - 0.10 * smoothstep(0.68, 0.94, cellular) * core, 0.0, 1.0)

    # Lift cores, preserve holes.
    body = np.clip(body ** 0.88, 0.0, 1.0)
    core = smoothstep(0.10, 0.72, core)

    # Ensure max area is respected. Use alpha-like preview.
    preview = np.clip(0.62 * body + 0.54 * core, 0.0, 1.0)
    preview = limit_alpha_coverage(preview, max_cloud_area=max_cloud_area)
    keep = smoothstep(0.015, 0.12, preview)
    body *= keep
    core *= keep

    return np.clip(body, 0.0, 1.0), np.clip(core, 0.0, 1.0)

def make_sccos_cloud_layer(h, w, rng, severity, cloud_density=1.20, max_cloud_area=0.50, fragment_density=3.80, outside_density=0.33, outside_alpha=0.15, secondary_density=0.26, free_density=0.12, transition_strength=0.17, veil_density=0.0, veil_alpha=0.0, veil_prob=0.0, cirrus_density=0.0, cirrus_alpha=0.0, cirrus_prob=0.0, brush_scale=0.94, min_main_cloud_patches=1, max_main_cloud_patches=4, main_patch_min_sep_scale=0.18):
    """
    v25.8: SCCOS-style bright fragmented cumulus from cloudlet assemblage.
    Each image randomly chooses 1–4 total main cloud patches and keeps a modest gap between main patches.
    """
    params = get_sccos_params(severity, rng, cloud_density=cloud_density, fragment_density=fragment_density)

    body, core = make_sccos_cloudlet_assemblage(
        h, w, rng, severity, params,
        max_cloud_area=max_cloud_area,
        fragment_density=fragment_density,
        outside_density=outside_density,
        outside_alpha=outside_alpha,
        secondary_density=secondary_density,
        free_density=free_density,
        transition_strength=transition_strength,
        brush_scale=brush_scale,
        min_main_cloud_patches=min_main_cloud_patches,
        max_main_cloud_patches=max_main_cloud_patches,
        main_patch_min_sep_scale=main_patch_min_sep_scale,
    )

    tex_a = ridge_noise(h, w, rng, start_grid=int(rng.integers(18, 34)), octaves=4, persistence=0.52)
    tex_b = fractal_value_noise(h, w, rng, start_grid=int(rng.integers(28, 56)), octaves=3, persistence=0.50)
    texture = soft_normalize01(0.58 * tex_a + 0.42 * tex_b)

    # Alpha: body + bright cores. Keep cloudlets separated.
    if severity == "light":
        alpha_gain = 0.66
        core_gain = 0.60
    elif severity == "medium":
        alpha_gain = 0.78
        core_gain = 0.76
    else:
        alpha_gain = 0.88
        core_gain = 0.92

    alpha = body * alpha_gain * (0.82 + 0.20 * texture)
    alpha = np.maximum(alpha, core * core_gain)
    alpha = np.clip(alpha * params["alpha_scale"], 0.0, 1.0)

    # Cut very faint haze; SCCOS cloud should be piece-like, not fog-like.
    alpha = alpha * smoothstep(0.020, 0.090, alpha)
    alpha = limit_alpha_coverage(alpha, max_cloud_area=max_cloud_area)

    # Optional veil/cirrus kept for CLI compatibility but effectively off by default.
    if rng.random() < veil_prob and veil_density > 1e-6 and veil_alpha > 1e-6:
        veil_layer = make_stratiform_veil_layer(h, w, rng, severity, veil_density=veil_density, veil_alpha=veil_alpha)
        alpha = np.clip(alpha + veil_layer * 0.35, 0.0, 1.0)
    if rng.random() < cirrus_prob and cirrus_density > 1e-6 and cirrus_alpha > 1e-6:
        cirrus_layer = make_cirrus_wisp_layer(h, w, rng, severity, cirrus_density=cirrus_density, cirrus_alpha=cirrus_alpha)
        alpha = np.clip(alpha + cirrus_layer * 0.25, 0.0, 1.0)

    # White/bright fragmented cloud color, with cores brighter than edges.
    bright = smoothstep(0.12, 0.80, np.maximum(core, body * 0.72))
    micro = ridge_noise(h, w, rng, start_grid=int(rng.integers(28, 54)), octaves=3, persistence=0.50)
    luma = params["white_base"] + params["white_gain"] * bright + 0.06 * texture + 0.04 * micro
    if severity == "light":
        luma = np.clip(luma, 0.80, 0.96)
    elif severity == "medium":
        luma = np.clip(luma, 0.84, 0.99)
    else:
        luma = np.clip(luma, 0.86, 1.0)

    cloud_rgb = np.stack([
        np.clip(luma * rng.uniform(0.99, 1.01), 0.0, 1.0),
        np.clip(luma * rng.uniform(1.00, 1.02), 0.0, 1.0),
        np.clip(luma * rng.uniform(1.01, 1.04), 0.0, 1.0),
    ], axis=2).astype(np.float32)

    # Small natural cloud shadow; not too dark.
    highlight_dir_x = int(rng.choice([-1, 1]) * rng.integers(2, 6))
    highlight_dir_y = int(rng.choice([-1, 1]) * rng.integers(2, 6))
    shadow = shift_float_image(alpha, -highlight_dir_x * 2, -highlight_dir_y * 2)
    shadow_k = odd_int(int(rng.integers(17, 35)))
    shadow = cv2.GaussianBlur(shadow, (shadow_k, shadow_k), 0)
    shadow = np.clip(shadow * params["shadow_strength"], 0.0, 0.08)

    return alpha, cloud_rgb, shadow, params


def blend_cloud_layer(img, alpha, cloud_rgb, shadow, params):
    """
    Blend the synthetic cloud layer into a normalized RGB image.

    This function was accidentally omitted in v25. v25.1 restores it.
    img       : float32 RGB image in [0, 1]
    alpha     : cloud alpha mask in [0, 1]
    cloud_rgb : float32 RGB cloud color in [0, 1]
    shadow    : float32 cloud shadow mask in [0, 1]
    params    : SCCOS parameter dict containing screen_gain and ambient_shadow
    """
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)[..., None]
    s = np.clip(shadow.astype(np.float32), 0.0, 1.0)[..., None]
    cloud_rgb = np.clip(cloud_rgb.astype(np.float32), 0.0, 1.0)

    # Slight cloud shadow first, but avoid making the image look smoky or dirty.
    img_shadowed = img * (1.0 - s * (1.0 - 0.55 * a))
    ambient_shadow = np.clip(a * float(params.get("ambient_shadow", 0.02)), 0.0, float(params.get("ambient_shadow", 0.02)))
    img_shadowed = img_shadowed * (1.0 - ambient_shadow)

    # Screen blend keeps thin cloud bright and transparent.
    screen_gain = float(params.get("screen_gain", 0.94))
    screen_out = 1.0 - (1.0 - img_shadowed) * (1.0 - cloud_rgb * a * screen_gain)

    # Normal alpha blend strengthens dense cloud cores.
    alpha_out = img_shadowed * (1.0 - a) + cloud_rgb * a

    # Thin edges use screen; dense cores use alpha blend.
    mix = np.clip((a - 0.10) / 0.38, 0.0, 1.0)
    out = screen_out * (1.0 - mix) + alpha_out * mix
    return np.clip(out, 0.0, 1.0)

def add_synthetic_clouds(image_rgb, rng, severity="medium", cloud_density=1.20, max_cloud_area=0.50, fragment_density=3.80, outside_density=0.33, outside_alpha=0.15, secondary_density=0.26, free_density=0.12, transition_strength=0.17, veil_density=0.0, veil_alpha=0.0, veil_prob=0.0, cirrus_density=0.0, cirrus_alpha=0.0, cirrus_prob=0.0, brush_scale=0.94, min_main_cloud_patches=1, max_main_cloud_patches=4, main_patch_min_sep_scale=0.18):
    h, w = image_rgb.shape[:2]
    out = image_rgb.astype(np.float32) / 255.0
    alpha, cloud_rgb, shadow, params = make_sccos_cloud_layer(
        h,
        w,
        rng,
        severity,
        cloud_density=cloud_density,
        max_cloud_area=max_cloud_area,
        fragment_density=fragment_density,
        outside_density=outside_density,
        outside_alpha=outside_alpha,
        secondary_density=secondary_density,
        free_density=free_density,
        transition_strength=transition_strength,
        veil_density=veil_density,
        veil_alpha=veil_alpha,
        veil_prob=veil_prob,
        cirrus_density=cirrus_density,
        cirrus_alpha=cirrus_alpha,
        cirrus_prob=cirrus_prob,
        brush_scale=brush_scale,
        min_main_cloud_patches=min_main_cloud_patches,
        max_main_cloud_patches=max_main_cloud_patches,
        main_patch_min_sep_scale=main_patch_min_sep_scale,
    )
    out = blend_cloud_layer(out, alpha, cloud_rgb, shadow, params)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def find_annotation(img_path, image_dir, ann_dir):
    rel = img_path.relative_to(image_dir)
    candidates = [
        ann_dir / rel.with_suffix(".xml"),
        ann_dir / rel.with_suffix(".txt"),
        ann_dir / f"{img_path.stem}.xml",
        ann_dir / f"{img_path.stem}.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def copy_annotation(src_ann, dst_ann, new_img_path):
    dst_ann.parent.mkdir(parents=True, exist_ok=True)
    if src_ann.suffix.lower() == ".xml":
        tree = ET.parse(src_ann)
        root = tree.getroot()
        filename_node = root.find("filename")
        if filename_node is not None:
            filename_node.text = new_img_path.name
        path_node = root.find("path")
        if path_node is not None:
            path_node.text = str(new_img_path)
        tree.write(dst_ann, encoding="utf-8", xml_declaration=True)
    else:
        shutil.copy2(src_ann, dst_ann)


def save_image(arr, out_path, jpeg_quality=95):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr)
    if out_path.suffix.lower() in [".jpg", ".jpeg"]:
        img.save(out_path, quality=jpeg_quality, subsampling=0)
    else:
        img.save(out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SCCOS-style bright fragmented cumulus cloud images (v25.8: each image randomly selects 1 to 4 total main cloud patches and keeps a modest distance between them).")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--ann_dir", type=str, default=None, help="Optional annotation directory.")
    parser.add_argument("--out_ann_dir", type=str, default=None, help="Optional output annotation directory.")
    parser.add_argument("--num_per_image", type=int, default=1)
    parser.add_argument("--light_ratio", type=float, default=0.05)
    parser.add_argument("--medium_ratio", type=float, default=0.65)
    parser.add_argument("--heavy_ratio", type=float, default=0.30)
    parser.add_argument("--cloud_density", type=float, default=1.20, help="主体云块密度，建议 1.10 到 1.34；v25.8 主体云块再略微增强。")
    parser.add_argument("--fragment_density", type=float, default=3.80, help="控制碎积云密度，建议 3.40 到 4.35；越大云块越密。v25.8 主云额外片区略增。")
    parser.add_argument("--max_cloud_area", type=float, default=0.50, help="控制云最大可见面积比例，建议 0.40 到 0.58；v25.8 为再增加的一片主云留更足面积余量。")
    parser.add_argument("--outside_density", type=float, default=0.33, help="外围零星散云密度，建议 0.20 到 0.46；v25.8 再略微增加周围散云。")
    parser.add_argument("--outside_alpha", type=float, default=0.15, help="外围零星散云透明度，建议 0.10 到 0.20。")
    parser.add_argument("--secondary_density", type=float, default=0.26, help="次云块密度，建议 0.18 到 0.36；v25.8 让主体附近再多一些小块。")
    parser.add_argument("--free_density", type=float, default=0.12, help="全图自由散云密度，建议 0.05 到 0.18；v25.8 保持背景碎云不过量。")
    parser.add_argument("--transition_strength", type=float, default=0.17, help="主云内部组块和新增主云片之间的少量桥接碎云，建议 0.12 到 0.26。")
    parser.add_argument("--min_main_cloud_patches", type=int, default=1, help="每张图像随机选择的主云总片数下限，建议 1 到 4。1 表示只保留 1 片主云。")
    parser.add_argument("--max_main_cloud_patches", type=int, default=4, help="每张图像随机选择的主云总片数上限，建议 1 到 4。4 表示最多 4 片主云。")
    parser.add_argument("--main_patch_min_sep_scale", type=float, default=0.18, help="主云片之间的最小中心间距比例，相对于图像短边；建议 0.14 到 0.24。")
    parser.add_argument("--veil_density", type=float, default=0.00, help="大范围薄层云密度，建议 0.00 到 0.80；默认关闭，避免形成一层雾化云幕。")
    parser.add_argument("--veil_alpha", type=float, default=0.00, help="大范围薄层云透明度；v19 默认关闭，避免一层雾化云幕。")
    parser.add_argument("--veil_prob", type=float, default=0.00, help="生成大范围薄层云的概率；v19 默认关闭。")
    parser.add_argument("--cirrus_density", type=float, default=0.00, help="卷云/云丝密度；v19 默认关闭，避免卷丝把画面拉成雾状纹理。")
    parser.add_argument("--cirrus_alpha", type=float, default=0.00, help="卷云/云丝透明度；v19 默认关闭。")
    parser.add_argument("--cirrus_prob", type=float, default=0.00, help="启用卷云/云丝层的概率；v19 默认关闭。")
    parser.add_argument("--brush_scale", type=float, default=0.94, help="云块整体尺度，建议 0.82 到 1.04；越小越偏小块拼接。")
    parser.add_argument("--suffix", type=str, default="_sccos")
    parser.add_argument("--same_name", action="store_true")
    parser.add_argument("--keep_subdirs", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--exts", type=str, default=".jpg,.jpeg,.png,.bmp,.tif,.tiff")
    return parser.parse_args()


def main():
    print(">>> Enter main()", flush=True)
    args = parse_args()
    print(">>> Args parsed", flush=True)

    if args.same_name and args.num_per_image != 1:
        raise ValueError("--same_name can only be used when --num_per_image=1.")
    args.min_main_cloud_patches = int(np.clip(args.min_main_cloud_patches, 1, 4))
    args.max_main_cloud_patches = int(np.clip(args.max_main_cloud_patches, args.min_main_cloud_patches, 4))

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    ann_dir = Path(args.ann_dir) if args.ann_dir else None
    out_ann_dir = Path(args.out_ann_dir) if args.out_ann_dir else out_dir / "Annotations"

    print(f">>> image_dir        = {image_dir}", flush=True)
    print(f">>> out_dir          = {out_dir}", flush=True)
    print(f">>> seed             = {args.seed}", flush=True)
    print(f">>> cloud_density    = {args.cloud_density}", flush=True)
    print(f">>> fragment_density = {args.fragment_density}", flush=True)
    print(f">>> max_cloud_area   = {args.max_cloud_area}", flush=True)
    print(f">>> outside_density  = {args.outside_density}", flush=True)
    print(f">>> outside_alpha    = {args.outside_alpha}", flush=True)
    print(f">>> secondary_density= {args.secondary_density}", flush=True)
    print(f">>> free_density     = {args.free_density}", flush=True)
    print(f">>> transition_strength = {args.transition_strength}", flush=True)
    print(f">>> min_main_cloud_patches = {args.min_main_cloud_patches}", flush=True)
    print(f">>> max_main_cloud_patches = {args.max_main_cloud_patches}", flush=True)
    print(f">>> main_patch_min_sep_scale = {args.main_patch_min_sep_scale}", flush=True)
    print(f">>> veil_density     = {args.veil_density}", flush=True)
    print(f">>> veil_alpha       = {args.veil_alpha}", flush=True)
    print(f">>> veil_prob        = {args.veil_prob}", flush=True)
    print(f">>> cirrus_density   = {args.cirrus_density}", flush=True)
    print(f">>> cirrus_alpha     = {args.cirrus_alpha}", flush=True)
    print(f">>> cirrus_prob      = {args.cirrus_prob}", flush=True)
    print(f">>> brush_scale      = {args.brush_scale}", flush=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")

    exts = {e.strip().lower() for e in args.exts.split(",")}
    image_paths = sorted([p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    print(f">>> Found images: {len(image_paths)}", flush=True)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in: {image_dir}")

    rng = np.random.default_rng(args.seed)
    total_outputs = len(image_paths) * args.num_per_image
    severity_plan = build_severity_plan(total_outputs, rng, light_ratio=args.light_ratio, medium_ratio=args.medium_ratio, heavy_ratio=args.heavy_ratio)
    severity_count = {"light": 0, "medium": 0, "heavy": 0}
    plan_idx = 0

    for img_path in tqdm(image_paths, desc="Adding SCCOS-style cloudlet clouds v25.8"):
        rel = img_path.relative_to(image_dir)
        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image_np = np.array(image)

        for i in range(args.num_per_image):
            severity = severity_plan[plan_idx]
            severity_count[severity] += 1
            plan_idx += 1

            local_seed = int(rng.integers(0, 2**31 - 1))
            local_rng = np.random.default_rng(local_seed)
            aug_np = add_synthetic_clouds(
                image_np,
                local_rng,
                severity=severity,
                cloud_density=args.cloud_density,
                max_cloud_area=args.max_cloud_area,
                fragment_density=args.fragment_density,
                outside_density=args.outside_density,
                outside_alpha=args.outside_alpha,
                secondary_density=args.secondary_density,
                free_density=args.free_density,
                transition_strength=args.transition_strength,
                veil_density=args.veil_density,
                veil_alpha=args.veil_alpha,
                veil_prob=args.veil_prob,
                cirrus_density=args.cirrus_density,
                cirrus_alpha=args.cirrus_alpha,
                cirrus_prob=args.cirrus_prob,
                brush_scale=args.brush_scale,
                min_main_cloud_patches=args.min_main_cloud_patches,
                max_main_cloud_patches=args.max_main_cloud_patches,
                main_patch_min_sep_scale=args.main_patch_min_sep_scale,
            )

            if args.same_name:
                new_rel = rel
            else:
                if args.keep_subdirs:
                    base_stem = rel.stem
                    parent = rel.parent
                else:
                    base_stem = "_".join(rel.with_suffix("").parts)
                    parent = Path("")
                if args.num_per_image == 1:
                    new_name = f"{base_stem}_{severity}{args.suffix}{rel.suffix}"
                else:
                    new_name = f"{base_stem}_{severity}{args.suffix}_{i + 1}{rel.suffix}"
                new_rel = parent / new_name

            out_img_path = out_dir / new_rel
            save_image(aug_np, out_img_path, jpeg_quality=args.jpeg_quality)

            if ann_dir is not None:
                src_ann = find_annotation(img_path, image_dir, ann_dir)
                if src_ann is not None:
                    ann_suffix = src_ann.suffix
                    out_ann_path = out_ann_dir / new_rel.with_suffix(ann_suffix)
                    copy_annotation(src_ann, out_ann_path, out_img_path)

    print(f"Done. Output saved to: {out_dir}")
    print(f"Actual severity count: light={severity_count['light']}, medium={severity_count['medium']}, heavy={severity_count['heavy']}")
    print(f"Actual severity ratio: light={severity_count['light'] / total_outputs:.4f}, medium={severity_count['medium'] / total_outputs:.4f}, heavy={severity_count['heavy'] / total_outputs:.4f}")


if __name__ == "__main__":
    main()
