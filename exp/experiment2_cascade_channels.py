import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from channel import Channel


ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / '黑白格.png'
OUTPUT_DIR = ROOT / 'experiment2_outputs'


def load_binary_image(path):
    image = Image.open(path).convert('1')
    array = (np.array(image, dtype=np.uint8) > 0).astype(np.uint8)
    return array


def save_binary_image(bits, path):
    image = Image.fromarray((bits.astype(np.uint8) * 255), mode='L')
    image.save(path)


def build_bsc(keep_prob):
    return Channel(
        [keep_prob, 1.0 - keep_prob, 1.0 - keep_prob, keep_prob],
        x=2,
        input_symbol=['0', '1'],
        output_symbol=['0', '1'],
        channel_type='MemoryLess',
    )


def vectorized_channel_pass(bits, channel, rng):
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    keep_prob = float(channel.P_channel[0, 0])
    flips = rng.random(bits.shape[0]) > keep_prob
    return np.bitwise_xor(bits, flips.astype(np.uint8))


def equivalent_channel(channel, levels):
    matrix = np.linalg.matrix_power(channel.P_channel, levels)
    return Channel(
        matrix.reshape(-1),
        x=2,
        input_symbol=['0', '1'],
        output_symbol=['0', '1'],
        channel_type='MemoryLess',
    )


def theoretical_mutual_information_curve(keep_probs, cascade_levels):
    p_x = np.array([0.5, 0.5], dtype=np.float32)
    curves = {level: [] for level in cascade_levels}
    for keep_prob in keep_probs:
        base_channel = build_bsc(float(keep_prob))
        for level in cascade_levels:
            eq_channel = equivalent_channel(base_channel, level)
            curves[level].append(float(eq_channel.mean_mutual_information(p_x)))
    return curves


def transmit_image_through_cascade(bits, base_channel, levels, rng):
    out = np.asarray(bits, dtype=np.uint8).reshape(-1)
    for _ in range(levels):
        out = vectorized_channel_pass(out, base_channel, rng)
    return out


def main():
    parser = argparse.ArgumentParser(description='Experiment 2 rewritten with pyShannon interfaces.')
    parser.add_argument('--keep-prob', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=20260405)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    image = load_binary_image(IMAGE_PATH)
    flat_bits = image.reshape(-1)

    cascade_levels = [1, 2, 3]
    keep_probs = np.arange(0.05, 1.0, 0.05)
    curves = theoretical_mutual_information_curve(keep_probs, cascade_levels)

    plt.figure(figsize=(8, 5))
    for level in cascade_levels:
        plt.plot(keep_probs, curves[level], marker='o', label=f'I(X;Y{level})')
    plt.xlabel('keep probability')
    plt.ylabel('mutual information')
    plt.title('Cascade-channel mutual information')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mutual_information_curve.png', dpi=200)
    plt.close()

    base_channel = build_bsc(args.keep_prob)
    print(f'keep_prob = {args.keep_prob:.2f}')
    for level in cascade_levels:
        eq_channel = equivalent_channel(base_channel, level)
        mi = eq_channel.mean_mutual_information(np.array([0.5, 0.5], dtype=np.float32))
        print(f'I(X;Y{level}) = {float(mi):.6f}')

    save_binary_image(image, OUTPUT_DIR / 'input_binary.png')
    for level in cascade_levels:
        out_bits = transmit_image_through_cascade(flat_bits, base_channel, level, rng)
        save_binary_image(out_bits.reshape(image.shape), OUTPUT_DIR / f'cascade_level_{level}.png')


if __name__ == '__main__':
    main()
