# data

**Everything here is git-ignored. Nothing in this directory is ever committed.**

This is the drop point for dataset bodies. Only this README is tracked; the rule in the
repository `.gitignore` is `/data/**` with an explicit exemption for `data/README.md`, so a
stray `git add -A` cannot commit a 14 GB archive.

## Layout the code expects

```
data/
├── synthrad/                      SynthRAD2023, Task 1 (MR → CT) only
│   └── Task1/
│       ├── brain/                 1BA001/{mr,ct,mask}.nii.gz …  60 A + 60 B + 60 C
│       ├── pelvis/                1PA001/…, 1PC001/…           120 A + 60 C (no centre B)
│       └── overview/              1_brain_train.xlsx, 1_pelvis_train.xlsx
└── t1t2/                          the T1/T2 MRI pairs used by studies/real/t1t2
    ├── T1/mri_*.png
    └── T2/mri_*.png
```

The patient folder name encodes `<task><anatomy><centre><index>`: `1` + `B`/`P` + centre letter
+ index. **Position 2 is the anatomy, position 3 is the centre** (`1BA001` = task 1, brain,
centre A). The loader (`harness/synthrad.py`) derives both from that name and will refuse a
folder it cannot parse rather than guessing.

## SynthRAD2023

- Zenodo DOI [`10.5281/zenodo.7260705`](https://doi.org/10.5281/zenodo.7260705), open, **CC BY-NC 4.0**,
  training split only.
- Task 1 archive: `Task1.zip`, 14 471 900 926 bytes, md5 `360bc61e2320d5c5a7145454a81c566a`.
- Uncompressed: brain 3.54 GB, pelvis 10.76 GB.
- Extract only what is needed:

  ```bash
  unzip Task1.zip -d data/synthrad            # both anatomies
  unzip Task1.zip 'Task1/brain/*' -d data/synthrad   # brain only (3.5 GB)
  ```

Verify the checksum before trusting any number produced from the data.

## T1/T2

Real T1/T2 MRI pairs, same-subject slices. **Registered pairs** — the T1 and T2 of one subject are
near-identical images, which is why the hidden-partner test saturates for pixel methods and is
never used as a headline on this data (see `reports/NMI_PAIRING_REPORT.md`).
