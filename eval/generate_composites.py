"""Generate full-canvas side-by-side, outline overlay, and diff heatmap composites for each logo."""
from __future__ import annotations

import re
from pathlib import Path
from PIL import Image, ImageChops
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000

from eval.metrics import render_svg

OUTLINE_STYLE = '<style>*{{fill:none!important;stroke:#ff007f!important;stroke-width:1.5px!important;vector-effect:non-scaling-stroke}}</style>'

def main():
    root = Path('eval/dataset/logos')
    out_dir = Path('eval/reports/defects_after_gradient')
    svg_dir = out_dir / 'svg'
    comp_dir = out_dir / 'full_composites'
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in sorted(root.glob('*.*')):
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", img_path.name)
        svg_file = svg_dir / f"{safe_name}.svg"
        if not svg_file.is_file():
            print(f"Skipping {img_path.name}: {svg_file} not found yet")
            continue
            
        print(f"Generating composites for {img_path.name}...")
        src_img = Image.open(img_path).convert('RGB')
        w, h = src_img.size
        svg_text = svg_file.read_text(encoding='utf-8')
        
        # 1. Render normal vector SVG
        vec_px = render_svg(svg_text, w, h)
        vec_img = Image.fromarray(vec_px, 'RGBA')
        
        # Composite vector over white background for clean RGB view
        vec_rgb = Image.new('RGB', (w, h), (255, 255, 255))
        vec_rgb.paste(vec_img, (0, 0), vec_img)
        
        # 2. Side-by-Side (Raster on Left, Vector on Right)
        sbs = Image.new('RGB', (w * 2 + 10, h), (30, 30, 30))
        sbs.paste(src_img, (0, 0))
        sbs.paste(vec_rgb, (w + 10, 0))
        sbs.save(comp_dir / f"{img_path.stem}_side_by_side.png")
        
        # 3. Magenta Outline Overlay (Vector Outlines on top of Raster)
        # Inject outline style into SVG
        svg_outlines = re.sub(r"<svg([^>]*)>", rf"<svg\1>{OUTLINE_STYLE}", svg_text, count=1)
        outline_px = render_svg(svg_outlines, w, h)
        outline_img = Image.fromarray(outline_px, 'RGBA')
        
        overlay = src_img.copy().convert('RGBA')
        overlay.alpha_composite(outline_img)
        overlay.convert('RGB').save(comp_dir / f"{img_path.stem}_overlay.png")
        
        # 4. Color & Edge Difference Heatmap (Delta E)
        src_arr = np.array(src_img, dtype=np.float64) / 255.0
        vec_arr = np.array(vec_rgb, dtype=np.float64) / 255.0
        src_lab = rgb2lab(src_arr)
        vec_lab = rgb2lab(vec_arr)
        de = deltaE_ciede2000(src_lab, vec_lab)
        
        # Map de (0 to 15) to colormap: 0=black, 2=blue, 5=green, 10=yellow, 15+=red
        de_norm = np.clip(de / 12.0, 0.0, 1.0)
        # Simple perceptual turbo heatmap
        r = np.clip(1.5 - np.abs(de_norm * 3.0 - 2.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(de_norm * 3.0 - 1.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(de_norm * 3.0 - 0.0), 0.0, 1.0)
        heatmap = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
        # Where error is very low (< 1.5 dE), show dim grayscale source
        mask_low = de < 1.5
        gray_src = np.array(src_img.convert('L'))[..., None] // 3
        heatmap[mask_low] = gray_src[mask_low]
        Image.fromarray(heatmap).save(comp_dir / f"{img_path.stem}_diff_heatmap.png")
        
    print(f"All composites saved in {comp_dir}!")

if __name__ == '__main__':
    main()
