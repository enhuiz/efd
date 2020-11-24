import argparse
import shutil
import numpy as np
import cv2
import tqdm
import torch
from pathlib import Path
from torchvision.utils import save_image
from einops import rearrange
from itertools import count
from more_itertools import chunked

from easyfd import s3fd


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = rearrange(frame, "h w c -> c h w")
        frame = frame.astype(np.float32) / 255.0
        yield frame


def main():
    parser = argparse.ArgumentParser(description="Only support single face.")
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--scale-factor", type=float, default=1)
    parser.add_argument("--reset", action="store_true", help="remove processed images")
    args = parser.parse_args()

    all_paths = args.videos
    pbar = tqdm.tqdm(all_paths)

    msg = "Are you sure to delete all processed folders and restart all again? (y): "
    if args.reset and input(msg) == "y":
        for p in all_paths:
            outdir = p.with_suffix("")
            if outdir.exists():
                shutil.rmtree(outdir)
                print(f"removing {outdir} ...")
            del outdir

    model = s3fd(pretrained=True)
    model = model.cuda()

    for p in pbar:
        outdir = p.with_suffix("")
        if outdir.exists():
            # skip existed
            continue
        pbar.set_description(str(p))
        counter = count()
        outdir.mkdir(parents=True, exist_ok=True)
        for batch in chunked(read_frames(p), args.batch_size):
            batch = torch.from_numpy(np.array(batch)).cuda()
            _, patch_iters = model.detect(batch, args.scale_factor)
            for i, patch_iter in zip(counter, patch_iters):
                outpath = outdir / f"{i:06d}.png"
                try:
                    patch = next(patch_iter)
                    save_image(patch, outpath)
                except:
                    print(
                        f"Warning: no face detected for {outpath}, "
                        "output a placeholder instead."
                    )
                    with open(outpath, "w") as f:
                        pass
