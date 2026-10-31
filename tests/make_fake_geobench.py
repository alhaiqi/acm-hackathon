"""Write two tiny GEO-Bench-format tasks (real geobench HDF5 + task_specs + partition) for offline tests.

The labels are learnable from the pixels, so a working pipeline must beat chance clearly.
"""
import sys
from pathlib import Path

import numpy as np

import geobench as gb
from geobench.dataset import Partition, write_sample_hdf5


def make(root: Path, name: str, seg: bool, n=(120, 40, 40), size=32, n_classes=3):
    bench = "segmentation_v1.0" if seg else "classification_v1.0"
    d = root / bench / name
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    infos = gb.sentinel2_13_bands[:4]
    if seg:
        label_type = gb.SegmentationClasses("label", spatial_resolution=10, n_classes=n_classes,
                                            class_names=[f"c{i}" for i in range(n_classes)])
    else:
        label_type = gb.Classification(n_classes, [f"c{i}" for i in range(n_classes)])
    part = Partition()
    k = 0
    for split, m in zip(("train", "valid", "test"), n):
        for _ in range(m):
            base = rng.normal(1000, 100, (4, size, size))
            if seg:
                lab = np.zeros((size, size), np.int16)
                for c in range(1, n_classes):
                    y0, x0 = rng.integers(0, size - 10, 2)
                    lab[y0:y0 + 10, x0:x0 + 10] = c
                base[0] += 600 * (lab == 1); base[1] += 600 * (lab == 2)
                label = gb.Band(lab, label_type, 10)
            else:
                label = int(rng.integers(0, n_classes))
                base[label % 4] += 300
            bands = [gb.Band(base[i].astype(np.int16), infos[i], 10) for i in range(4)]
            s = gb.Sample(bands, label, f"s{k:05d}")
            write_sample_hdf5(s, d)
            part.add(split, s.sample_name)
            k += 1
    part.save(str(d), "default")
    gb.TaskSpecifications(dataset_name=name, benchmark_name=bench, bands_info=infos, spatial_resolution=10,
                          patch_size=(size, size), n_time_steps=1, label_type=label_type).save(str(d), overwrite=True)
    return d


if __name__ == "__main__":
    root = Path(sys.argv[1])
    print(make(root, "m-eurosat", seg=False))
    print(make(root, "m-pv4ger-seg", seg=True))
