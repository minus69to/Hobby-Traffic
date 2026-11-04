"""Copy the best checkpoint of every trained seed into models/final/ (the only policy files kept in git).

Run ``python scripts/finalize_models.py [--src models/single] [--prefix single]``.
"""

from __future__ import annotations

import argparse
import shutil

from drs.config import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="models/single")
    parser.add_argument("--prefix", default="single")
    args = parser.parse_args()
    src = REPO_ROOT / args.src
    for best in sorted(src.glob("seed*/best")):
        seed = best.parent.name.replace("seed", "")
        dst = REPO_ROOT / "models" / "final" / f"{args.prefix}_seed{seed}"
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("policy.zip", "vecnormalize.pkl"):
            shutil.copy2(best / name, dst / name)
        meta = best.parent / "train_meta.json"
        if meta.exists():
            shutil.copy2(meta, dst / "train_meta.json")
        progress = best.parent / "progress.csv"
        if progress.exists():
            shutil.copy2(progress, dst / "progress.csv")
        size_kb = (dst / "policy.zip").stat().st_size // 1024
        print(f"{best.relative_to(REPO_ROOT)} -> {dst.relative_to(REPO_ROOT)} ({size_kb} kB)")


if __name__ == "__main__":
    main()
