# -*- coding: utf-8 -*-
"""
使用 imgaug 批量给光学遥感图像添加随机雾（light / medium / heavy）。

改进版：
1. 整体降低 light / medium / heavy 雾强，避免“雾太重、图像看不清”。
2. 黑边保护改为“只保护贴近边缘的细黑边”，避免把与边界相连的暗水面/暗地物整块保留下来。
3. 增加过雾判断：不仅限制过白，也限制亮度提升过大、对比度损失过大。
4. 快速减淡/修复内部黑色小点。
5. 默认 max_tries=3，提高运行速度。
6. 支持多进程 workers 加速。
7. 输出保持原文件名，并生成 fog_log.csv。

依赖：
    pip install "numpy<2" opencv-python imgaug

示例：
    python addfog_fast_clean.py ^
        --input D:\Downloadss\DOTA\DOTA602\DOTA_ship_m ^
        --output D:\Downloadss\DOTA\DOTA602\Foggy_DOTA2 ^
        --light-ratio 0.14 ^
        --medium-ratio 0.57 ^
        --heavy-ratio 0.29 ^
        --seed 42 ^
        --max-tries 2 ^
        --workers 4 ^
        --verbose
"""

import argparse
import csv
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import imgaug.augmenters as iaa


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ============================================================
# 图像读写
# ============================================================

def read_image_bgr(image_path):
    """使用 imdecode 读取图像，兼容 Windows 中文路径。"""
    image_path = Path(image_path)
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image_bgr


def write_image_bgr(output_path, image_bgr):
    """使用 imencode 写图像，兼容 Windows 中文路径。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ext = output_path.suffix.lower()
    if ext not in IMG_EXTS:
        ext = ".png"

    success, buffer = cv2.imencode(ext, image_bgr)
    if not success:
        raise RuntimeError(f"图像编码失败: {output_path}")

    buffer.tofile(str(output_path))


# ============================================================
# 黑边保护与黑点修复
# ============================================================

def get_border_black_mask(
    original_rgb,
    threshold=8,
    border_width=12,
    max_component_area_ratio=0.02,
):
    """
    检测需要保护的边界黑边，但避免把大片暗水面/暗地物误判成黑边。

    旧逻辑的问题：
    - 只要“近黑区域”和图像边界相连，就整块保护；
    - 对遥感图像里的暗水面、阴影、海湾很危险，容易出现左侧大片发黑。

    新逻辑：
    - 默认只保护距离图像外边界 border_width 像素以内的近黑像素；
    - 对非常小的边界连通域可整体保护；
    - 大块连通暗区域不会被整块保留下来。
    """
    if threshold <= 0 or border_width <= 0:
        return np.zeros(original_rgb.shape[:2], dtype=bool)

    black_mask = np.all(original_rgb <= threshold, axis=2).astype(np.uint8)
    h, w = black_mask.shape

    # 只保护贴近图像边缘的黑边，避免整片暗水面被保护成黑块
    edge_band = np.zeros_like(black_mask, dtype=bool)
    bw = max(1, int(border_width))
    edge_band[:bw, :] = True
    edge_band[h - bw:, :] = True
    edge_band[:, :bw] = True
    edge_band[:, w - bw:] = True

    border_mask = (black_mask.astype(bool) & edge_band)

    # 对很小的边界黑色连通域，允许整体保护，避免小黑角被雾化成灰边
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        black_mask,
        connectivity=8
    )

    max_area = int(h * w * max_component_area_ratio)

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area > max_area:
            continue

        component = labels == label_id
        touches_border = (
            np.any(component[0, :]) or
            np.any(component[h - 1, :]) or
            np.any(component[:, 0]) or
            np.any(component[:, w - 1])
        )

        if touches_border:
            border_mask[component] = True

    return border_mask

def preserve_black_borders(
    original_rgb,
    foggy_rgb,
    threshold=8,
    border_width=12,
    max_component_area_ratio=0.02,
):
    """
    只保护贴近图像外边缘的细黑边。
    不再把所有“与边界相连的近黑区域”整块保护，避免左侧/水面大片发黑。
    """
    if threshold <= 0 or border_width <= 0:
        return foggy_rgb

    border_mask = get_border_black_mask(
        original_rgb,
        threshold=threshold,
        border_width=border_width,
        max_component_area_ratio=max_component_area_ratio,
    )

    if np.any(border_mask):
        foggy_rgb = foggy_rgb.copy()
        foggy_rgb[border_mask] = original_rgb[border_mask]

    return foggy_rgb

def remove_black_speckles_fast(
    original_rgb,
    foggy_rgb,
    black_threshold=28,
    max_area=120,
    inpaint_radius=2,
    preserve_black_threshold=8,
    border_width=12,
    max_component_area_ratio=0.02,
):
    """
    快速减淡/修复内部黑色小点。

    与之前版本不同：
    - 只处理非常黑的点 gray < black_threshold；
    - 只处理小面积连通区域；
    - 不处理船体、建筑阴影、码头阴影等大面积真实暗区域；
    - 使用 inpaint 修补小黑点，速度比复杂中值混合更稳定；
    - 不做全图模糊，尽量保留云纹理和原始亮度。

    推荐参数：
    black_threshold=25~35
    max_area=80~180
    inpaint_radius=1~2
    """
    foggy_gray = cv2.cvtColor(foggy_rgb, cv2.COLOR_RGB2GRAY)

    # 候选黑点：只取非常黑的像素
    dark_mask = (foggy_gray < black_threshold).astype(np.uint8)

    # 排除图像边界黑边
    border_mask = get_border_black_mask(
        original_rgb,
        threshold=preserve_black_threshold,
        border_width=border_width,
        max_component_area_ratio=max_component_area_ratio,
    )
    dark_mask[border_mask] = 0

    # 连通域筛选，只修小黑点
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark_mask,
        connectivity=8
    )

    speckle_mask = np.zeros_like(dark_mask, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area <= max_area:
            speckle_mask[labels == label_id] = 255

    if np.count_nonzero(speckle_mask) == 0:
        return foggy_rgb

    # 轻微闭运算，连接零碎黑点，但不大面积扩张
    kernel = np.ones((3, 3), np.uint8)
    speckle_mask = cv2.morphologyEx(
        speckle_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # inpaint 只修小黑点，不影响整体图像
    fixed_rgb = cv2.inpaint(
        foggy_rgb,
        speckle_mask,
        inpaint_radius,
        cv2.INPAINT_TELEA
    )

    return fixed_rgb


# ============================================================
# 质量指标与过白判断
# ============================================================

def compute_quality_metrics(original_rgb, foggy_rgb):
    """计算质量指标，用于日志及过白控制。"""
    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    foggy_gray = cv2.cvtColor(foggy_rgb, cv2.COLOR_RGB2GRAY)

    original_gray_f = original_gray.astype(np.float32)
    foggy_gray_f = foggy_gray.astype(np.float32)

    diff = foggy_gray_f - original_gray_f
    brightness_increase = float(np.mean(diff))

    original_std = float(np.std(original_gray_f))
    foggy_std = float(np.std(foggy_gray_f))

    contrast_drop = 0.0 if original_std < 1e-6 else float(
        1.0 - foggy_std / original_std
    )

    brightened_ratio = float(np.mean(diff > 12.0))

    newly_white = (foggy_gray > 245) & (original_gray < 230)
    newly_white_ratio = float(np.mean(newly_white))

    height, width = foggy_gray.shape
    y1, y2 = int(height * 0.25), int(height * 0.75)
    x1, x2 = int(width * 0.25), int(width * 0.75)

    center_gray = foggy_gray[y1:y2, x1:x2]
    center_white_ratio = float(np.mean(center_gray > 245)) if center_gray.size > 0 else 0.0

    return {
        "brightness_increase": brightness_increase,
        "contrast_drop": contrast_drop,
        "brightened_ratio": brightened_ratio,
        "newly_white_ratio": newly_white_ratio,
        "center_white_ratio": center_white_ratio,
    }


def is_overwhite(metrics, fog_level):
    """
    根据雾等级设定安全阈值。

    不只判断“过白”，也判断“过雾”：
    - brightness_increase 太大：整图被抬得太亮；
    - contrast_drop 太大：细节被雾吃掉；
    - brightened_ratio 太大：大面积像盖了一层灰白膜。
    """
    limits = {
        "light": {
            "newly_white_ratio": 0.003,
            "center_white_ratio": 0.012,
            "brightness_increase": 18.0,
            "contrast_drop": 0.28,
            "brightened_ratio": 0.55,
        },
        "medium": {
            "newly_white_ratio": 0.008,
            "center_white_ratio": 0.025,
            "brightness_increase": 28.0,
            "contrast_drop": 0.40,
            "brightened_ratio": 0.70,
        },
        "heavy": {
            "newly_white_ratio": 0.018,
            "center_white_ratio": 0.045,
            "brightness_increase": 38.0,
            "contrast_drop": 0.52,
            "brightened_ratio": 0.82,
        },
    }

    cfg = limits[fog_level]

    return (
        metrics["newly_white_ratio"] > cfg["newly_white_ratio"]
        or metrics["center_white_ratio"] > cfg["center_white_ratio"]
        or metrics["brightness_increase"] > cfg["brightness_increase"]
        or metrics["contrast_drop"] > cfg["contrast_drop"]
        or metrics["brightened_ratio"] > cfg["brightened_ratio"]
    )

# ============================================================
# 雾增强器
# ============================================================

def make_fog_augmenter(fog_level, seed=None):
    """
    用 imgaug 的 CloudLayer 构造三个等级的雾。

    这版比原脚本更保守：
    - light / medium / heavy 都降低 alpha_multiplier 和 density_multiplier；
    - heavy 不再追求特别浓，避免出现“整张图只剩灰幕”的情况；
    - 轻微降低对比和饱和度，但尽量保留遥感纹理。
    """
    if fog_level == "light":
        seq = iaa.Sequential([
            iaa.CloudLayer(
                intensity_mean=(217, 234),
                intensity_freq_exponent=(-2.6, -2.2),
                intensity_coarse_scale=2,
                alpha_min=(0.47, 0.59),
                alpha_multiplier=(0.08, 0.150),
                alpha_size_px_max=(2, 5),
                alpha_freq_exponent=(-4.2, -3.0),
                sparsity=(0.92, 0.98),
                density_multiplier=(0.26, 0.41),
                seed=seed,
            ),
            iaa.LinearContrast((0.95, 1.00)),
            iaa.MultiplySaturation((0.96, 1.00)),
        ])

    elif fog_level == "medium":
        seq = iaa.Sequential([
            iaa.CloudLayer(
                intensity_mean=(221, 238),
                intensity_freq_exponent=(-2.3, -1.9),
                intensity_coarse_scale=2,
                alpha_min=(0.50, 0.68),
                alpha_multiplier=(0.09, 0.180),
                alpha_size_px_max=(2, 7),
                alpha_freq_exponent=(-4.0, -2.4),
                sparsity=(0.89, 0.97),
                density_multiplier=(0.29, 0.51),
                seed=seed,
            ),
            iaa.LinearContrast((0.90, 0.96)),
            iaa.MultiplySaturation((0.89, 0.97)),
        ])

    elif fog_level == "heavy":
        seq = iaa.Sequential([
            iaa.CloudLayer(
                intensity_mean=(223, 241),
                intensity_freq_exponent=(-2.1, -1.7),
                intensity_coarse_scale=2,
                alpha_min=(0.6, 0.72),
                alpha_multiplier=(0.14, 0.23),
                alpha_size_px_max=(3, 8),
                alpha_freq_exponent=(-3.8, -2.4),
                sparsity=(0.87, 0.96),
                density_multiplier=(0.35, 0.62),
                seed=seed,
            ),
            iaa.LinearContrast((0.86, 0.94)),
            iaa.MultiplySaturation((0.84, 0.93)),
        ])

    else:
        raise ValueError(f"未知雾等级: {fog_level}")

    return seq

# ============================================================
# 单图加雾
# ============================================================

def apply_fog_with_retry(
    original_rgb,
    fog_level,
    rng,
    preserve_black_threshold=8,
    border_width=12,
    max_border_component_area_ratio=0.02,
    max_tries=3,
    remove_speckles=True,
    black_threshold=28,
    max_speckle_area=120,
):
    """
    对单张图像加雾。

    为了提速：
    - 默认 max_tries=3；
    - 如果想更快，可以命令行设置 --max-tries 1。
    """
    best_rgb = None
    best_metrics = None
    best_score = None
    best_try = None

    for t in range(1, max_tries + 1):
        aug_seed = int(rng.integers(0, 2**31 - 1))
        aug = make_fog_augmenter(fog_level=fog_level, seed=aug_seed)

        foggy_rgb = aug(image=original_rgb)

        # 只保护边界黑边
        foggy_rgb = preserve_black_borders(
            original_rgb,
            foggy_rgb,
            threshold=preserve_black_threshold,
            border_width=border_width,
            max_component_area_ratio=max_border_component_area_ratio,
        )

        # 快速修复内部小黑点
        if remove_speckles:
            foggy_rgb = remove_black_speckles_fast(
                original_rgb,
                foggy_rgb,
                black_threshold=black_threshold,
                max_area=max_speckle_area,
                inpaint_radius=2,
                preserve_black_threshold=preserve_black_threshold,
                border_width=border_width,
                max_component_area_ratio=max_border_component_area_ratio,
            )

        metrics = compute_quality_metrics(original_rgb, foggy_rgb)

        score = (
            metrics["newly_white_ratio"] * 3.0
            + metrics["center_white_ratio"] * 2.0
            + max(0.0, metrics["contrast_drop"]) * 1.5
            + max(0.0, metrics["brightness_increase"]) / 80.0
            + metrics["brightened_ratio"] * 0.5
        )

        if best_score is None or score < best_score:
            best_rgb = foggy_rgb
            best_metrics = metrics
            best_score = score
            best_try = t

        if not is_overwhite(metrics, fog_level):
            return foggy_rgb, metrics, t, True

    return best_rgb, best_metrics, best_try, False


# ============================================================
# 雾等级规划
# ============================================================

def build_level_plan(num_images, light_ratio, medium_ratio, heavy_ratio, rng):
    """
    按数据集级别规划各等级张数，尽量接近目标比例，且总数严格等于 num_images。
    """
    ratios = np.array([light_ratio, medium_ratio, heavy_ratio], dtype=np.float64)
    raw_counts = ratios * num_images
    base_counts = np.floor(raw_counts).astype(int)
    remainder = num_images - int(base_counts.sum())

    fractional = raw_counts - base_counts
    order = np.argsort(-fractional)

    for i in range(remainder):
        base_counts[order[i % len(order)]] += 1

    levels = (
        ["light"] * base_counts[0]
        + ["medium"] * base_counts[1]
        + ["heavy"] * base_counts[2]
    )

    rng.shuffle(levels)

    return levels, {
        "light": int(base_counts[0]),
        "medium": int(base_counts[1]),
        "heavy": int(base_counts[2]),
    }


def validate_or_normalize_ratios(
    light_ratio,
    medium_ratio,
    heavy_ratio,
    normalize_ratios=False
):
    total = light_ratio + medium_ratio + heavy_ratio

    if total <= 0:
        raise ValueError("light_ratio + medium_ratio + heavy_ratio 必须大于 0")

    if normalize_ratios:
        return light_ratio / total, medium_ratio / total, heavy_ratio / total

    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"当前比例和为 {total:.6f}，不等于 1.0。"
            "因为每张图都只加一种雾，所以三类比例之和必须为 1。"
            "如果你想按权重自动归一化，请加 --normalize-ratios。"
        )

    return light_ratio, medium_ratio, heavy_ratio


# ============================================================
# 单图处理 worker
# ============================================================

def process_one_image_worker(task):
    """
    多进程 worker。

    注意：
    - 为了 Windows 多进程兼容，所有参数通过 task 传入；
    - 每张图用独立 seed，保证可复现。
    """
    (
        image_path,
        output_path,
        fog_level,
        seed,
        preserve_black_threshold,
        border_width,
        max_border_component_area_ratio,
        max_tries,
        remove_speckles,
        black_threshold,
        max_speckle_area,
        verbose,
    ) = task

    image_path = Path(image_path)
    output_path = Path(output_path)

    image_bgr = read_image_bgr(image_path)
    if image_bgr is None:
        return {
            "ok": False,
            "input_path": str(image_path),
            "output_path": str(output_path),
            "error": "读取失败",
        }

    rng = np.random.default_rng(seed)

    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    foggy_rgb, metrics, used_try, passed = apply_fog_with_retry(
        original_rgb=original_rgb,
        fog_level=fog_level,
        rng=rng,
        preserve_black_threshold=preserve_black_threshold,
        border_width=border_width,
        max_border_component_area_ratio=max_border_component_area_ratio,
        max_tries=max_tries,
        remove_speckles=remove_speckles,
        black_threshold=black_threshold,
        max_speckle_area=max_speckle_area,
    )

    foggy_bgr = cv2.cvtColor(foggy_rgb, cv2.COLOR_RGB2BGR)
    write_image_bgr(output_path, foggy_bgr)

    if verbose:
        print(
            f"[ok] level={fog_level:6s} | tries={used_try} | pass={int(bool(passed))} | "
            f"{image_path} -> {output_path} | "
            f"brightness+={metrics['brightness_increase']:.2f}, "
            f"contrast_drop={metrics['contrast_drop']:.3f}, "
            f"new_white={metrics['newly_white_ratio']:.4f}, "
            f"center_white={metrics['center_white_ratio']:.4f}"
        )

    return {
        "ok": True,
        "input_path": str(image_path),
        "output_path": str(output_path),
        "fog_level": fog_level,
        "retry_used": used_try,
        "quality_passed": int(bool(passed)),
        **metrics,
    }


# ============================================================
# 文件夹处理
# ============================================================

def process_folder(
    input_dir,
    output_dir,
    light_ratio,
    medium_ratio,
    heavy_ratio,
    seed,
    preserve_black_threshold,
    border_width,
    max_border_component_area_ratio,
    max_tries,
    remove_speckles,
    black_threshold,
    max_speckle_area,
    workers,
    verbose,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    main_rng = np.random.default_rng(seed)

    image_paths = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    image_paths.sort()

    num_images = len(image_paths)

    if num_images == 0:
        raise RuntimeError(f"在 {input_dir} 下没有找到图像文件")

    levels_plan, planned_counts = build_level_plan(
        num_images=num_images,
        light_ratio=light_ratio,
        medium_ratio=medium_ratio,
        heavy_ratio=heavy_ratio,
        rng=main_rng,
    )

    print(f"找到 {num_images} 张图像")
    print("每张图像生成: 1 张（且只加一种雾）")
    print("输出命名方式: 保持原图文件名")
    print(f"输出目录: {output_dir}")
    print(f"max_tries: {max_tries}")
    print(f"preserve_black_threshold: {preserve_black_threshold}")
    print(f"border_width: {border_width}")
    print(f"max_border_component_area_ratio: {max_border_component_area_ratio}")
    print(f"workers: {workers}")
    print(f"remove_speckles: {remove_speckles}")
    print("计划雾等级分布:")

    for k in ["light", "medium", "heavy"]:
        print(f"  {k:6s}: {planned_counts[k]} 张 ({planned_counts[k] / num_images:.2%})")

    tasks = []

    for idx, (image_path, fog_level) in enumerate(zip(image_paths, levels_plan)):
        relative_path = image_path.relative_to(input_dir)
        output_path = output_dir / relative_path

        if seed is None:
            task_seed = int(main_rng.integers(0, 2**31 - 1))
        else:
            task_seed = int(seed + idx * 10007)

        tasks.append((
            str(image_path),
            str(output_path),
            fog_level,
            task_seed,
            preserve_black_threshold,
            border_width,
            max_border_component_area_ratio,
            max_tries,
            remove_speckles,
            black_threshold,
            max_speckle_area,
            verbose,
        ))

    success_count = 0
    fail_count = 0
    fog_counter = Counter()
    log_rows = []

    if workers <= 1:
        for task in tasks:
            result = process_one_image_worker(task)

            if result["ok"]:
                success_count += 1
                fog_counter[result["fog_level"]] += 1
                log_rows.append({k: result[k] for k in [
                    "input_path",
                    "output_path",
                    "fog_level",
                    "retry_used",
                    "quality_passed",
                    "brightness_increase",
                    "contrast_drop",
                    "brightened_ratio",
                    "newly_white_ratio",
                    "center_white_ratio",
                ]})
            else:
                fail_count += 1
                print(f"[fail] {result['input_path']} | {result.get('error', '')}")

    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_one_image_worker, task) for task in tasks]

            for future in as_completed(futures):
                result = future.result()

                if result["ok"]:
                    success_count += 1
                    fog_counter[result["fog_level"]] += 1
                    log_rows.append({k: result[k] for k in [
                        "input_path",
                        "output_path",
                        "fog_level",
                        "retry_used",
                        "quality_passed",
                        "brightness_increase",
                        "contrast_drop",
                        "brightened_ratio",
                        "newly_white_ratio",
                        "center_white_ratio",
                    ]})
                else:
                    fail_count += 1
                    print(f"[fail] {result['input_path']} | {result.get('error', '')}")

    log_path = output_dir / "fog_log.csv"

    if log_rows:
        log_rows.sort(key=lambda x: x["input_path"])

        with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

    print("处理完成")
    print(f"成功生成: {success_count} 张")
    print(f"失败跳过: {fail_count} 张")
    print("实际雾等级统计:")

    for level in ["light", "medium", "heavy"]:
        print(f"  {level:6s}: {fog_counter[level]} 张")

    print(f"日志文件: {log_path}")
    print(f"结果保存在: {output_dir}")


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="使用 imgaug 批量给光学遥感图像添加随机轻雾 / 中雾 / 浓雾，并快速减淡黑色小点。"
    )

    parser.add_argument("--input", required=True, help="输入图像文件夹")
    parser.add_argument("--output", required=True, help="输出图像文件夹")

    parser.add_argument("--light-ratio", type=float, default=0.14, help="轻雾比例，默认 0.14")
    parser.add_argument("--medium-ratio", type=float, default=0.57, help="中雾比例，默认 0.57")
    parser.add_argument("--heavy-ratio", type=float, default=0.29, help="浓雾比例，默认 0.29")

    parser.add_argument(
        "--normalize-ratios",
        action="store_true",
        help="若三类比例和不为 1，则自动按权重归一化",
    )

    parser.add_argument("--seed", type=int, default=None, help="随机种子")

    parser.add_argument(
        "--preserve-black-threshold",
        type=int,
        default=8,
        help="只保护边界黑边的阈值，默认 8；设为 0 表示不保护黑边",
    )

    parser.add_argument(
        "--border-width",
        type=int,
        default=12,
        help="只保护距离图像边缘多少像素以内的黑边，默认 12；可设为 0 关闭黑边保护",
    )

    parser.add_argument(
        "--max-border-component-area-ratio",
        type=float,
        default=0.02,
        help="允许整体保护的小型边界黑色连通域最大面积比例，默认 0.02；防止暗水面被整块保护",
    )

    parser.add_argument(
        "--max-tries",
        type=int,
        default=3,
        help="单张图像因过白/过雾而重采样的最大次数，默认 3；想更快可设为 1",
    )

    parser.add_argument(
        "--disable-remove-speckles",
        action="store_true",
        help="关闭内部黑色小点修复",
    )

    parser.add_argument(
        "--black-threshold",
        type=int,
        default=28,
        help="黑点检测阈值，默认 28；黑点多可调到 35",
    )

    parser.add_argument(
        "--max-speckle-area",
        type=int,
        default=120,
        help="最大黑点连通域面积，默认 120；黑点多可调到 180",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行进程数，默认 1；Windows 建议 2~4",
    )

    parser.add_argument("--verbose", action="store_true", help="打印详细日志")

    args = parser.parse_args()

    for name in ["light_ratio", "medium_ratio", "heavy_ratio"]:
        if getattr(args, name) < 0:
            raise ValueError(f"{name} 不能为负数")

    light_ratio, medium_ratio, heavy_ratio = validate_or_normalize_ratios(
        args.light_ratio,
        args.medium_ratio,
        args.heavy_ratio,
        normalize_ratios=args.normalize_ratios,
    )

    process_folder(
        input_dir=args.input,
        output_dir=args.output,
        light_ratio=light_ratio,
        medium_ratio=medium_ratio,
        heavy_ratio=heavy_ratio,
        seed=args.seed,
        preserve_black_threshold=args.preserve_black_threshold,
        border_width=args.border_width,
        max_border_component_area_ratio=args.max_border_component_area_ratio,
        max_tries=args.max_tries,
        remove_speckles=not args.disable_remove_speckles,
        black_threshold=args.black_threshold,
        max_speckle_area=args.max_speckle_area,
        workers=args.workers,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
