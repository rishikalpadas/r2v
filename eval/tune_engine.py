"""Automated Parameter Tuning Engine for Raster-to-Vector (r2v).

Caches super-resolved GPU images to achieve ultra-fast evaluation (~1-2s per image),
evaluates candidate parameter configurations across diverse representative designs,
scores them against automated defect detectors, SSIM, Delta E, and node counts,
and outputs the globally optimal universal default settings.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import csv
import itertools
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from engine.loader import load_image
from engine.presets import FitParams, Preset, SegmentParams, resolve_preset
from engine.stages.enhance import enhance
from engine.stages.export import build_doc, to_svg
from engine.stages.fit_geometry import fit_geometry
from engine.stages.segment import segment
from engine.stages.trace import trace
from engine.types import RasterImage
from eval.defects import detect_all, geometry_from_state
from eval.metrics import count_nodes, mean_delta_e, render_svg, ssim
from eval.run_defects import NORM, _crop_box, _load_rgba, _scaled


class CachedSample:
    def __init__(self, name: str, path: Path, raw_image: RasterImage, enhanced_image: RasterImage):
        self.name = name
        self.path = path
        self.raw_image = raw_image
        self.enhanced_image = enhanced_image
        self.target = enhanced_image.pixels
        self.h, self.w = self.target.shape[:2]
        self.k = (
            self.w / raw_image.original_size[0],
            self.h / raw_image.original_size[1],
        )


def load_and_cache_samples(sample_names: list[str], images_dir: Path) -> list[CachedSample]:
    """Enhance each sample once on GPU and cache in RAM and disk."""
    cache_dir = Path("eval/cache_enhanced")
    cache_dir.mkdir(parents=True, exist_ok=True)
    preset = resolve_preset("auto")
    cached = []
    print(f"Loading/caching enhanced images for {len(sample_names)} representative samples...")
    for name in sample_names:
        p = images_dir / name
        if not p.is_file():
            print(f"Warning: {name} not found in {images_dir}")
            continue
        raw_img = load_image(p.read_bytes())
        disk_cache = cache_dir / f"{p.stem}_enh.npz"
        if disk_cache.is_file():
            data = np.load(disk_cache)
            pixels = data["pixels"]
            scale = float(data["scale"])
            enh_img = RasterImage(
                pixels=pixels,
                original_size=raw_img.original_size,
                scale=scale,
                color_space=raw_img.color_space,
                source_format=raw_img.source_format,
            )
            cached.append(CachedSample(name, p, raw_img, enh_img))
            print(f"  Loaded {name} from disk cache ({enh_img.scale}x, {enh_img.pixels.shape[1]}x{enh_img.pixels.shape[0]})")
        else:
            t0 = time.perf_counter()
            enh_img, _ = enhance(raw_img, preset.enhance, preset.segment.max_colors)
            dt = time.perf_counter() - t0
            np.savez_compressed(disk_cache, pixels=enh_img.pixels, scale=enh_img.scale)
            cached.append(CachedSample(name, p, raw_img, enh_img))
            print(f"  Enhanced & cached {name} on GPU ({enh_img.scale}x, {enh_img.pixels.shape[1]}x{enh_img.pixels.shape[0]}) in {dt:.1f}s")
    return cached


def evaluate_config(
    samples: list[CachedSample],
    fit_params: FitParams,
    segment_params: SegmentParams,
) -> dict[str, Any]:
    """Run segment -> trace -> fit -> export -> metrics for a given configuration across all cached samples."""
    preset = resolve_preset("auto")
    preset_export = preset.export

    total_combined = 0.0
    total_ssim = 0.0
    total_delta_e = 0.0
    total_nodes = 0
    total_crossings = 0
    total_wobble = 0.0
    total_spikes = 0.0
    total_specks = 0

    per_image_results = {}

    for sample in samples:
        # 1. Segment
        regions = segment(sample.enhanced_image, segment_params)
        # 2. Trace
        graph = trace(regions)
        # 3. Fit
        fitted = fit_geometry(graph, fit_params, sample.enhanced_image.scale)
        # 4. Export
        doc = build_doc(
            regions,
            graph,
            fitted,
            preset_export,
            scale=sample.enhanced_image.scale,
            original_size=sample.raw_image.original_size,
        )
        svg = to_svg(doc)
        nodes, _ = count_nodes(svg)

        # 5. Measure
        render = render_svg(svg, sample.w, sample.h)
        if render.shape[:2] != (sample.h, sample.w):
            render = np.array(
                Image.fromarray(render, "RGBA").resize(
                    (sample.w, sample.h), Image.Resampling.LANCZOS
                )
            )

        # Build mock state for geometry detectors
        from engine.pipeline import PipelineState
        state = PipelineState(
            image=sample.enhanced_image,
            preset=preset,
            regions=regions,
            graph=graph,
            fitted=fitted,
            doc=doc,
        )
        geom = geometry_from_state(state)
        results = {
            d: (r if d == "worst_patches" else _scaled(r, sample.k))
            for d, r in detect_all(geom, sample.target, render, sample.raw_image.pixels).items()
        }

        combined = sum(float(results[d].score) / NORM[d] for d in NORM if d in results)
        cur_ssim = ssim(sample.target, render)
        cur_dE = mean_delta_e(sample.target, render)

        total_combined += combined
        total_ssim += cur_ssim
        total_delta_e += cur_dE
        total_nodes += nodes
        total_crossings += int(results.get("crossings", type("Obj", (), {"score": 0})).score)
        total_wobble += float(results.get("wobble", type("Obj", (), {"score": 0.0})).score)
        total_spikes += float(results.get("junction_spikes", type("Obj", (), {"score": 0.0})).score)
        total_specks += int(results.get("specks", type("Obj", (), {"score": 0})).score)

        per_image_results[sample.name] = {
            "combined": combined,
            "ssim": cur_ssim,
            "delta_e": cur_dE,
            "nodes": nodes,
        }

    n = len(samples)
    avg_combined = total_combined / n
    avg_ssim = total_ssim / n
    avg_delta_e = total_delta_e / n
    avg_nodes = total_nodes / n

    # Unified Objective Loss: Lower is better.
    # We heavily penalize defects, reward SSIM, penalize Delta E, and gently penalize node bloat.
    objective_loss = (
        avg_combined
        + 60.0 * (1.0 - avg_ssim)
        + 3.0 * avg_delta_e
        + 0.004 * avg_nodes
        + 2.0 * total_crossings
    )

    return {
        "objective_loss": objective_loss,
        "avg_combined": avg_combined,
        "avg_ssim": avg_ssim,
        "avg_delta_e": avg_delta_e,
        "avg_nodes": avg_nodes,
        "total_crossings": total_crossings,
        "total_wobble": total_wobble,
        "total_spikes": total_spikes,
        "total_specks": total_specks,
        "per_image": per_image_results,
    }


def grid_search(cached_samples: list[CachedSample]) -> tuple[dict, dict]:
    """Perform systematic parameter sweep over key hyperparameters."""
    base_preset = resolve_preset("auto")
    base_fit = base_preset.fit
    base_seg = base_preset.segment

    print("\n--- Evaluating Baseline Settings ---")
    base_res = evaluate_config(cached_samples, base_fit, base_seg)
    print(f"Baseline Objective Loss: {base_res['objective_loss']:.3f} | Combined: {base_res['avg_combined']:.2f} | SSIM: {base_res['avg_ssim']:.4f} | Nodes: {base_res['avg_nodes']:.0f}")

    # Parameter grid for calibration
    tolerances = [0.65, 0.75, 0.88, 1.0]
    smooth_sigmas = [0.85, 1.0, 1.15]
    corner_angles = [32.0, 35.0, 38.0]
    region_merge_dEs = [7.5, 9.0, 10.5]
    speck_areas = [8.0, 12.0, 16.0]

    best_loss = base_res["objective_loss"]
    best_fit = base_fit
    best_seg = base_seg
    best_res = base_res

    # Strategy: Coordinate sweep (one or two parameters at a time to test sensitivity efficiently)
    print("\n--- Phase 1: Tolerance & Smoothing Calibration ---")
    for tol in tolerances:
        for sigma in smooth_sigmas:
            test_fit = replace(base_fit, tolerance_px=tol, smooth_sigma=sigma)
            res = evaluate_config(cached_samples, test_fit, base_seg)
            improved = res["objective_loss"] < best_loss
            star = " * [BEST]" if improved else ""
            print(f"  tol={tol:.2f}, sigma={sigma:.2f} -> Loss={res['objective_loss']:.3f} (Defects={res['avg_combined']:.2f}, SSIM={res['avg_ssim']:.4f}, Nodes={res['avg_nodes']:.0f}){star}")
            if improved:
                best_loss = res["objective_loss"]
                best_fit = test_fit
                best_res = res

    print(f"\nPhase 1 Optimal: tolerance={best_fit.tolerance_px}, smooth_sigma={best_fit.smooth_sigma} (Loss: {best_loss:.3f})")

    print("\n--- Phase 2: Corner Angle Calibration ---")
    for ca in corner_angles:
        test_fit = replace(best_fit, corner_angle_deg=ca)
        res = evaluate_config(cached_samples, test_fit, base_seg)
        improved = res["objective_loss"] < best_loss
        star = " * [BEST]" if improved else ""
        print(f"  corner_angle={ca:.1f} deg -> Loss={res['objective_loss']:.3f} (Defects={res['avg_combined']:.2f}, Spikes={res['total_spikes']:.1f}, Nodes={res['avg_nodes']:.0f}){star}")
        if improved:
            best_loss = res["objective_loss"]
            best_fit = test_fit
            best_res = res

    print(f"\nPhase 2 Optimal: corner_angle={best_fit.corner_angle_deg} deg (Loss: {best_loss:.3f})")

    print("\n--- Phase 3: Region Merge & Speck Filter Calibration ---")
    for rm in region_merge_dEs:
        for sp in speck_areas:
            test_seg = replace(base_seg, region_merge_delta_e=rm, speck_area_px=sp)
            res = evaluate_config(cached_samples, best_fit, test_seg)
            improved = res["objective_loss"] < best_loss
            star = " * [BEST]" if improved else ""
            print(f"  merge_dE={rm:.1f}, speck_area={sp:.1f} -> Loss={res['objective_loss']:.3f} (Defects={res['avg_combined']:.2f}, Specks={res['total_specks']}){star}")
            if improved:
                best_loss = res["objective_loss"]
                best_seg = test_seg
                best_res = res

    print(f"\n=== Auto-Tuning Optimization Complete ===")
    print(f"Baseline Loss: {base_res['objective_loss']:.3f} -> Optimal Loss: {best_loss:.3f} ({(best_loss - base_res['objective_loss']) / base_res['objective_loss'] * 100:+.1f}%)")
    print(f"Defects: {base_res['avg_combined']:.2f} -> {best_res['avg_combined']:.2f}")
    print(f"Nodes: {base_res['avg_nodes']:.0f} -> {best_res['avg_nodes']:.0f}")
    print(f"SSIM: {base_res['avg_ssim']:.4f} -> {best_res['avg_ssim']:.4f}")
    print(f"\nRecommended Optimal Settings:")
    print(f"  FitParams(tolerance_px={best_fit.tolerance_px}, smooth_sigma={best_fit.smooth_sigma}, corner_angle_deg={best_fit.corner_angle_deg})")
    print(f"  SegmentParams(region_merge_delta_e={best_seg.region_merge_delta_e}, speck_area_px={best_seg.speck_area_px})")

    return asdict(best_fit), asdict(best_seg)


def main():
    parser = argparse.ArgumentParser(description="Auto-tune r2v hyperparameters.")
    parser.add_argument("--images-dir", type=Path, default=Path("eval/dataset/logos"))
    parser.add_argument("--samples", nargs="*", default=[
        "e (13).jpeg",  # gradient banner
        "e (15).jpeg",  # cartoon outlines
        "e (12).jpeg",  # smileys & typography
        "e (18).jpeg",  # complex multi-character
        "e (14).jpeg",  # fine cursive script
        "e (21).jpeg",  # new sample 1
        "e (22).jpeg",  # new sample 2
    ])
    args = parser.parse_args()

    cached = load_and_cache_samples(args.samples, args.images_dir)
    grid_search(cached)


if __name__ == "__main__":
    main()
