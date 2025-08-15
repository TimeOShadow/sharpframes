import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import imageio
import imageio_ffmpeg as _  # ensure ffmpeg plugin is available
import numpy as np
import cv2
from tqdm import tqdm
import tifffile


@dataclass
class FrameScore:
    index: int
    score: float


def compute_laplacian_variance(gray_image: np.ndarray) -> float:
    # gray_image: uint8 or uint16; convert to float32 for stability
    img = gray_image.astype(np.float32)
    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    return float(lap.var())


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    # frame expected in RGB (imageio returns RGB)
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        # Unexpected channels, squeeze or take first 3
        if frame.shape[2] > 3:
            gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return gray


def robust_auto_threshold(scores: List[float], k: float = 2.0) -> float:
    arr = np.asarray(scores, dtype=np.float64)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median)) + 1e-9
    return float(median + k * 1.4826 * mad)


def select_by_ratio(indices: List[int], keep_ratio: float, min_interval: int) -> List[int]:
    if not indices:
        return []
    if keep_ratio <= 0:
        return []
    if keep_ratio >= 1.0:
        # still enforce min_interval if provided
        stride = max(1, min_interval)
        return indices[::stride]
    stride = max(1, int(round(1.0 / keep_ratio)))
    selected = []
    last_idx = -10**12
    for idx in indices[::stride]:
        if not selected or (idx - last_idx) >= max(1, min_interval):
            selected.append(idx)
            last_idx = idx
    if not selected and indices:
        selected = [indices[0]]
    return selected


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_dng_linear(path: str, rgb8_or_16: np.ndarray) -> None:
    # We store as linear DNG using tifffile with required baseline tags.
    # Convert to uint16 linear if input is uint8
    if rgb8_or_16.dtype == np.uint8:
        data16 = (rgb8_or_16.astype(np.uint16) * 257)
    else:
        data16 = rgb8_or_16.astype(np.uint16)

    # tifffile.imwrite can write DNG with proper metadata tags; we set minimal tags.
    # Note: This produces linear RGB DNG (not mosaic Bayer).
    tifffile.imwrite(
        path,
        data16,
        photometric='rgb',
        dtype=np.uint16,
        metadata={'DNGVersion': (1, 4, 0, 0)},
        compression=None
    )


def pass_one_score(video_path: str, every: int) -> Tuple[List[FrameScore], int]:
    reader = imageio.get_reader(video_path)
    scores: List[FrameScore] = []
    n_total = 0
    try:
        for i, frame in enumerate(tqdm(reader, desc='Pass1: scoring', unit='f')):
            n_total += 1
            if every > 1 and (i % every != 0):
                continue
            gray = to_grayscale(frame)
            score = compute_laplacian_variance(gray)
            scores.append(FrameScore(index=i, score=score))
    finally:
        reader.close()
    return scores, n_total


def pass_two_export(video_path: str, indices: List[int], out_dir: str, save_format: str) -> None:
    ensure_dir(out_dir)
    # Efficient second pass using set for quick membership
    target_set = set(indices)
    reader = imageio.get_reader(video_path)
    try:
        for i, frame in enumerate(tqdm(reader, desc='Pass2: exporting', unit='f')):
            if i not in target_set:
                continue
            # frame is RGB uint8 typically; convert to uint16 linear
            out_name = f"frame_{i:08d}.{ 'dng' if save_format == 'dng' else 'dng' }"
            out_path = os.path.join(out_dir, out_name)
            write_dng_linear(out_path, frame)
    finally:
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='SharpFrames: threshold + ratio sampling to DNG')
    parser.add_argument('--input', required=True, help='Path to input video file')
    parser.add_argument('--output', required=True, help='Directory to store extracted frames')
    parser.add_argument('--score_threshold', default='auto', help='auto or numeric threshold for LapVar')
    parser.add_argument('--threshold_k', type=float, default=2.0, help='k for auto threshold median + k*MAD')
    parser.add_argument('--keep_ratio', type=float, default=1.0, help='ratio on candidates after threshold')
    parser.add_argument('--min_interval', type=int, default=1, help='minimum frame interval between kept')
    parser.add_argument('--sample_every', type=int, default=1, help='coarse scoring stride to speed up')
    parser.add_argument('--save_format', default='dng', choices=['dng'], help='output raw format')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f'Input file not found: {args.input}', file=sys.stderr)
        sys.exit(1)

    # Pass 1: score
    scores, n_total = pass_one_score(args.input, every=max(1, int(args.sample_every)))
    if not scores:
        print('No frames scored. Check input file.', file=sys.stderr)
        sys.exit(2)

    score_values = [s.score for s in scores]
    if isinstance(args.score_threshold, str) and args.score_threshold.lower() == 'auto':
        thr = robust_auto_threshold(score_values, k=args.threshold_k)
    else:
        thr = float(args.score_threshold)

    candidates = [s.index for s in scores if s.score >= thr]
    candidates.sort()

    # Ratio sampling among candidates
    selected = select_by_ratio(candidates, keep_ratio=float(args.keep_ratio), min_interval=int(args.min_interval))

    print(f'Total frames (iterated): {n_total}')
    print(f'Scored frames: {len(scores)}; candidates >= threshold: {len(candidates)}; selected after ratio: {len(selected)}')

    # Pass 2: export
    pass_two_export(args.input, selected, args.output, args.save_format)


if __name__ == '__main__':
    main()


