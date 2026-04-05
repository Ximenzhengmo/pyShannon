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
from codec import HuffmanCodec, tolerant_huffman_decode


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'experiment4_outputs'
GRAY_IMAGE_PATH = ROOT / 'gray_image.jpg'
FALLBACK_IMAGE_PATH = ROOT / 'image.png'


def resolve_image_path():
    if GRAY_IMAGE_PATH.exists():
        return GRAY_IMAGE_PATH
    if FALLBACK_IMAGE_PATH.exists():
        return FALLBACK_IMAGE_PATH
    raise FileNotFoundError("Neither `gray_image.jpg` nor `image.png` was found in ./exp.")


def load_grayscale_image(path):
    return np.array(Image.open(path).convert('L'), dtype=np.uint8)


def save_grayscale_image(array, path):
    Image.fromarray(array.astype(np.uint8), mode='L').save(path)


def build_bsc(keep_prob=0.99):
    return Channel(
        [keep_prob, 1.0 - keep_prob, 1.0 - keep_prob, keep_prob],
        x=2,
        input_symbol=['0', '1'],
        output_symbol=['0', '1'],
        channel_type='MemoryLess',
    )


def transmit_bitstring(bitstring, channel, rng):
    bits = np.fromiter((1 if b == '1' else 0 for b in bitstring), dtype=np.uint8)
    keep_prob = float(channel.P_channel[0, 0])
    flips = rng.random(bits.shape[0]) > keep_prob
    out = np.bitwise_xor(bits, flips.astype(np.uint8))
    return ''.join(out.astype(str).tolist())


def fixed_length_encode(pixels):
    return ''.join(format(int(v), '08b') for v in pixels)


def fixed_length_decode(bitstring, expected_len):
    usable = (len(bitstring) // 8) * 8
    values = [int(bitstring[i:i + 8], 2) for i in range(0, usable, 8)]
    arr = np.array(values[:expected_len], dtype=np.uint8)
    if arr.size < expected_len:
        arr = np.pad(arr, (0, expected_len - arr.size), mode='constant')
    return arr


def repetition_encode(bitstring, repeats=3):
    return ''.join(bit * repeats for bit in bitstring)


def repetition_decode(bitstring, repeats=3):
    decoded = []
    usable = (len(bitstring) // repeats) * repeats
    for i in range(0, usable, repeats):
        segment = bitstring[i:i + repeats]
        decoded.append('1' if segment.count('1') > segment.count('0') else '0')
    return ''.join(decoded)


def psnr(original, reconstructed):
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def main():
    parser = argparse.ArgumentParser(description='Experiment 4 rewritten with pyShannon interfaces.')
    parser.add_argument('--keep-prob', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=20260405)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    image_path = resolve_image_path()
    image = load_grayscale_image(image_path)
    flat_pixels = image.reshape(-1)
    expected_len = flat_pixels.size

    channel = build_bsc(args.keep_prob)

    # System 1: fixed-length source representation.
    fixed_bits = fixed_length_encode(flat_pixels)
    fixed_rx_bits = transmit_bitstring(fixed_bits, channel, rng)
    fixed_rx_pixels = fixed_length_decode(fixed_rx_bits, expected_len).reshape(image.shape)
    save_grayscale_image(fixed_rx_pixels, OUTPUT_DIR / 'system1_fixed_length.png')

    # System 2: Huffman source coding.
    codec = HuffmanCodec.from_data(flat_pixels.tolist())
    huffman_bits = codec.encode(flat_pixels.tolist())
    huffman_rx_bits = transmit_bitstring(huffman_bits, channel, rng)
    huffman_rx_pixels = np.array(
        tolerant_huffman_decode(
            huffman_rx_bits,
            codec,
            expected_length=expected_len,
            return_list=True,
        ),
        dtype=np.uint8,
    ).reshape(image.shape)
    save_grayscale_image(huffman_rx_pixels, OUTPUT_DIR / 'system2_huffman.png')

    # System 3: repetition channel coding on fixed-length bits.
    repeated_bits = repetition_encode(fixed_bits, repeats=3)
    repeated_rx_bits = transmit_bitstring(repeated_bits, channel, rng)
    decoded_bits = repetition_decode(repeated_rx_bits, repeats=3)
    repeated_rx_pixels = fixed_length_decode(decoded_bits, expected_len).reshape(image.shape)
    save_grayscale_image(repeated_rx_pixels, OUTPUT_DIR / 'system3_repetition.png')

    save_grayscale_image(image, OUTPUT_DIR / 'input_gray.png')

    systems = {
        'fixed_length': {
            'channel_input_bits': len(fixed_bits),
            'psnr': float(psnr(image, fixed_rx_pixels)),
        },
        'huffman_source_coding': {
            'channel_input_bits': len(huffman_bits),
            'psnr': float(psnr(image, huffman_rx_pixels)),
        },
        'repetition_channel_coding': {
            'channel_input_bits': len(repeated_bits),
            'psnr': float(psnr(image, repeated_rx_pixels)),
        },
    }

    for name, info in systems.items():
        print(name)
        print(f"  channel_input_bits = {info['channel_input_bits']}")
        print(f"  psnr = {info['psnr']:.4f} dB")

    labels = list(systems.keys())
    lengths = [systems[name]['channel_input_bits'] for name in labels]
    psnr_values = [systems[name]['psnr'] for name in labels]

    plt.figure(figsize=(9, 4))
    plt.bar(labels, lengths, color=['#4361ee', '#f9844a', '#2a9d8f'])
    plt.ylabel('channel input length (bits)')
    plt.title('Communication efficiency comparison')
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'channel_input_length.png', dpi=200)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.bar(labels, psnr_values, color=['#4361ee', '#f9844a', '#2a9d8f'])
    plt.ylabel('PSNR (dB)')
    plt.title('Communication reliability comparison')
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'psnr_comparison.png', dpi=200)
    plt.close()


if __name__ == '__main__':
    main()
