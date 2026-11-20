"""Build a clean, documented subset of the Roboflow 'detection f1 cars' v12 dataset.

Steps: convert polygon labels to boxes, drop images with empty labels,
remove validation and test images that have a near-duplicate in an earlier
split, using a 64-bit difference hash, then sample 1000 / 250 / 250 images with a fixed seed.
Put this file next to the folder 'detection f1 cars.v12i.yolov8' and run:
    python build_dataset.py
"""
import csv, random, shutil
from pathlib import Path
from PIL import Image

SEED = 42
N = {'train': 1000, 'valid': 250, 'test': 250}
THR = 8
BASE = Path(__file__).parent
ROOT = BASE / "detection f1 cars.v12i.yolov8"
OUT = BASE / "f1_subset"
names = ['Ferrari', 'Mclaren', 'Mercedes', 'Redbull']


def dhash(path, size=8):
    img = Image.open(path).convert('L').resize((size + 1, size))
    px = list(img.tobytes())
    bits = 0
    for r in range(size):
        for c in range(size):
            i = r * (size + 1) + c
            bits = (bits << 1) | (px[i] > px[i + 1])
    return bits


def to_boxes(label_path):
    rows = []
    for line in label_path.read_text(encoding='utf-8').splitlines():
        p = line.split()
        if not p:
            continue
        cls, v = int(p[0]), [float(x) for x in p[1:]]
        if len(v) == 4:
            xc, yc, w, h = v
        else:
            xs, ys = v[0::2], v[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        rows.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return rows


def load(split):
    items = {}
    for img in (ROOT / split / 'images').glob('*.jpg'):
        lab = ROOT / split / 'labels' / (img.stem + '.txt')
        if lab.exists():
            rows = to_boxes(lab)
            if rows:
                items[img] = rows
    return items


data = {s: load(s) for s in ['train', 'valid', 'test']}
hashes = {s: {p: dhash(p) for p in data[s]} for s in data}


def has_near(h, others):
    return any(bin(h ^ o).count('1') <= THR for o in others)


train_h = list(hashes['train'].values())
valid_h = list(hashes['valid'].values())


clean = {
    'train': list(data['train']),
    'valid': [p for p in data['valid'] if not has_near(hashes['valid'][p], train_h)],
    'test': [p for p in data['test']
             if not has_near(hashes['test'][p], train_h)
             and not has_near(hashes['test'][p], valid_h)],
}

rng = random.Random(SEED)
manifest = []
for split in ['train', 'valid', 'test']:
    pool = sorted(clean[split])
    chosen = rng.sample(pool, min(N[split], len(pool)))
    (OUT / split / 'images').mkdir(parents=True, exist_ok=True)
    (OUT / split / 'labels').mkdir(parents=True, exist_ok=True)
    counts = [0] * 4
    for img in chosen:
        rows = data[split][img]
        shutil.copy(img, OUT / split / 'images' / img.name)
        (OUT / split / 'labels' / (img.stem + '.txt')).write_text("\n".join(rows) + "\n")
        for r in rows:
            counts[int(r.split()[0])] += 1
        manifest.append((split, img.name))
    print(split, ': non-empty', len(data[split]), '-> after dedup', len(pool),
          '-> chosen', len(chosen), ', objects:', dict(zip(names, counts)))

with open(OUT / 'split_manifest.csv', 'w', newline='', encoding='utf-8') as f:
    csv.writer(f).writerows([('split', 'filename')] + manifest)
(OUT / 'data.yaml').write_text(
    "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\n"
    "nc: 4\nnames: ['Ferrari', 'Mclaren', 'Mercedes', 'Redbull']\n", encoding='utf-8')
print('seed', SEED, 'threshold', THR, 'done ->', OUT)
