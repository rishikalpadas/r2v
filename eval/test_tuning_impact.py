from pathlib import Path
from engine.presets import resolve_preset, FitParams, SegmentParams
from eval.tune_engine import load_and_cache_samples, evaluate_config

samples = load_and_cache_samples(
    ['e (13).jpeg', 'e (15).jpeg', 'e (12).jpeg', 'e (14).jpeg', 'e (18).jpeg'],
    Path('eval/dataset/logos')
)

base_fit = FitParams()
base_seg = SegmentParams()

# Balanced candidate:
# - keeps tolerance at 0.75 px to prevent node bloat on smooth curves
# - gentle smooth_sigma 0.95 to preserve crisp text without wobble
# - corner_angle 34.0 deg for clean junctions
# - region_merge_delta_e 10.0 and speck_area 14.0 to filter noise without deleting details
balanced_fit = FitParams(tolerance_px=0.75, smooth_sigma=0.95, corner_angle_deg=34.0)
balanced_seg = SegmentParams(region_merge_delta_e=10.0, speck_area_px=14.0)

res_base = evaluate_config(samples, base_fit, base_seg)
res_bal = evaluate_config(samples, balanced_fit, balanced_seg)

print('=== BALANCED CALIBRATION COMPARISON ===')
for name in ['e (13).jpeg', 'e (15).jpeg', 'e (12).jpeg', 'e (14).jpeg', 'e (18).jpeg']:
    b = res_base['per_image'][name]
    t = res_bal['per_image'][name]
    print(f"{name:15s} | Defect: {b['combined']:6.2f} -> {t['combined']:6.2f} | SSIM: {b['ssim']:6.4f} -> {t['ssim']:6.4f} | Nodes: {b['nodes']:5d} -> {t['nodes']:5d}")

print(f"\nAvg Defect: {res_base['avg_combined']:.2f} -> {res_bal['avg_combined']:.2f}")
print(f"Avg SSIM:   {res_base['avg_ssim']:.4f} -> {res_bal['avg_ssim']:.4f}")
print(f"Avg Nodes:  {res_base['avg_nodes']:.0f} -> {res_bal['avg_nodes']:.0f}")
