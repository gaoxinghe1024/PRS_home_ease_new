"""Download the MCF and Le2i mirrors from Kaggle."""

import sys
import os
import shutil
import time
import kagglehub

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "cache", "kaggle")
SLUGS = {
    "mc": "soumicksarker/multiple-cameras-fall-dataset",
    "le2i": "tuyenldvn/falldataset-imvia",
}


def download(key):
    slug = SLUGS[key]
    print(f"[{key}] downloading {slug} ...", flush=True)
    t0 = time.time()
    p = kagglehub.dataset_download(slug)
    print(f"[{key}] kagglehub cache: {p}  elapsed {time.time() - t0:.0f}s", flush=True)
    dst = os.path.join(DEST, slug.split("/")[-1])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.isdir(dst):
        try:
            shutil.copytree(p, dst, copy_function=os.link)
            print(f"[{key}] hard-linked to {dst}", flush=True)
        except OSError:
            shutil.copytree(p, dst)
            print(f"[{key}] copied to {dst}", flush=True)
    else:
        print(f"[{key}] target already exists: {dst}", flush=True)
    return dst


if __name__ == "__main__":
    keys = sys.argv[1:] or ["mc", "le2i"]
    for k in keys:
        download(k)
    print("All done")
