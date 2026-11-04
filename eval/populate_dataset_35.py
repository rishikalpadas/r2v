"""Populate eval/dataset/logos to 35 images combining the existing 17 + 18 new real samples."""
import hashlib
import json
import shutil
from pathlib import Path

logos_dir = Path('eval/dataset/logos')
new_samples_dir = Path('../files/new samples')
b1_dir = Path('../test samples/batch 1')

def get_hashes(folder):
    res = {}
    for p in folder.glob('*.*'):
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            res[h] = p
    return res

existing = get_hashes(logos_dir)
new_s = get_hashes(new_samples_dir)
b1 = get_hashes(b1_dir)

# 17 new from new samples
candidates = [p for h, p in new_s.items() if h not in existing]
candidates.sort(key=lambda p: p.name)

# 1 from batch 1 to make it exactly 18 new = 35 total
for h, p in b1.items():
    if h not in existing and h not in new_s:
        candidates.append(p)
        break

manifest = {}
for p in sorted(logos_dir.glob('*.*')):
    if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
        manifest[p.name] = {'source': 'original_17', 'path': str(p)}

next_num = 21
for src in candidates:
    target_name = f'e ({next_num}).jpeg'
    target_path = logos_dir / target_name
    shutil.copy2(src, target_path)
    manifest[target_name] = {'source': str(src), 'size': src.stat().st_size}
    print(f'Copied {src.name[:40]:40s} -> {target_name}')
    next_num += 1

Path('eval/dataset/manifest_35.json').write_text(json.dumps(manifest, indent=2))
total_count = len([p for p in logos_dir.glob('*.*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')])
print(f'Total images in {logos_dir}: {total_count}')
