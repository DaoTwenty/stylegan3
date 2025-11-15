# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import copy
import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm 
from pathlib import Path
import PIL.Image

import legacy

#----------------------------------------------------------------------------

def gen_interp_batches(G, out_base_dir, seeds, n_sequences=1, w_frames=60*4, kind='cubic', num_keyframes=None,  wraps=2, psi=1, device=torch.device('cuda'), shuffle_seed=None):
    """
    Generate multiple independent interpolation sequences and save each frame as PNG.

    Args:
        G: Generator with mapping() and synthesis().
        out_base_dir: Base folder for saving sequences. Each sequence gets out_base_dir/sequence_i/
        seeds: List of integer seeds to sample latent vectors.
        n_sequences: Number of independent sequences to generate.
        w_frames: Number of frames per interpolation.
        kind: Interpolation type ('linear', 'cubic', etc.).
        psi: Truncation psi for StyleGAN.
        device: Torch device.
        shuffle_seed: Optional RNG seed to shuffle seeds for each sequence.
    """
    os.makedirs(out_base_dir, exist_ok=True)
    if num_keyframes is None:
        if len(seeds) % (n_sequences) != 0:
                raise ValueError('Number of input seeds must be divisible by number of sequences n_sequences')
        num_keyframes = len(seeds) // (n_sequences)

    # Prepare seeds for multiple sequences
    all_seeds = np.zeros(num_keyframes*n_sequences, dtype=np.int64)
    for idx in range(num_keyframes*n_sequences):
        all_seeds[idx] = seeds[idx % len(seeds)]

    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)

    zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
    ws = G.mapping(z=zs, c=None, truncation_psi=psi)
    _ = G.synthesis(ws[:1]) # warm up
    ws = ws.reshape(n_sequences, num_keyframes, *ws.shape[1:])

    seq_folders = []
    seq_interpolations = []
    for si in range(n_sequences):
        seq_folder = Path(out_base_dir) / f"interpolation_{si+1}"
        seq_folder.mkdir(parents=True, exist_ok=True)
        seq_folders.append(seq_folder)

        x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
        y = np.tile(ws[si].cpu().numpy(), [wraps * 2 + 1, 1, 1])
        interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
        seq_interpolations.append(interp)

    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        for si in range(n_sequences):
            interp = seq_interpolations[si]
            w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)
            img = G.synthesis(ws=w.unsqueeze(0), noise_mode='const')

            img = (img.permute(0, 1, 2, 3) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            frame_path = seq_folders[si] / f"frame_{frame_idx:04d}.png"
            PIL.Image.fromarray(img[0].cpu().numpy()[0]).save(frame_path)

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--batch-size', type=int, help='Number of interpolation sequences to generate', default=1)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--output', help='Output folder directory', type=str, required=True, metavar='FILE')
def generate_images(
    network_pkl: str,
    seeds: List[int],
    batch_size: int,
    shuffle_seed: Optional[int],
    num_keyframes: Optional[int],
    w_frames: int,
    truncation_psi: float,
    output: str
):
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    gen_interp_batches(G=G, out_base_dir=output, seeds=seeds, n_sequences=batch_size, num_keyframes=num_keyframes, w_frames=w_frames, shuffle_seed=shuffle_seed, psi=truncation_psi)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
