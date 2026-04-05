import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from codec import HuffmanCodec
from probability import entropy
from source import Source


ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / 'image.png'
OUTPUT_DIR = ROOT / 'experiment3_outputs'


def load_binary_image_bits(path):
    image = Image.open(path).convert('1')
    bits = (np.array(image, dtype=np.uint8) > 0).astype(np.uint8).reshape(-1)
    return bits


def group_bits(bits, n):
    usable = (len(bits) // n) * n
    bits = bits[:usable].reshape(-1, n)
    return [''.join(map(str, row.tolist())) for row in bits]


def analyze_blocks(blocks, n):
    codec = HuffmanCodec.from_data(blocks)
    probabilities = codec.probabilities
    avg_block_length = codec.average_code_length()
    avg_symbol_length = avg_block_length / n
    efficiency = float(entropy(list(probabilities.values())) / avg_block_length)
    preview = sorted(codec.codes.items(), key=lambda item: item[0])[: min(16, len(codec.codes))]
    return {
        'codec': codec,
        'probabilities': probabilities,
        'avg_block_length': float(avg_block_length),
        'avg_symbol_length': float(avg_symbol_length),
        'efficiency': efficiency,
        'preview': preview,
    }


def simulate_memoryless_blocks(bit_stream, n, rng):
    p1 = float(np.mean(bit_stream))
    source = Source([1.0 - p1, p1], x=2, symbol=['0', '1'], source_type='MemoryLess')
    block_prob = source.prob_topk(n).reshape(-1)
    block_count = len(bit_stream) // n
    sampled = rng.choice(len(block_prob), size=block_count, p=block_prob)
    return [format(index, f'0{n}b') for index in sampled]


def save_preview_csv(path, preview, probabilities):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['symbol', 'probability', 'code'])
        for symbol, code in preview:
            writer.writerow([symbol, probabilities[symbol], code])


def main():
    parser = argparse.ArgumentParser(description='Experiment 3 rewritten with pyShannon interfaces.')
    parser.add_argument('--extensions', type=int, nargs='*', default=[2, 4, 8])
    parser.add_argument('--seed', type=int, default=20260405)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    bits = load_binary_image_bits(IMAGE_PATH)
    real_rows = []
    simulated_rows = []

    p1 = float(np.mean(bits))
    print(f'raw Bernoulli distribution: P(0)={1.0 - p1:.6f}, P(1)={p1:.6f}')

    for n in args.extensions:
        real_blocks = group_bits(bits, n)
        sim_blocks = simulate_memoryless_blocks(bits, n, rng)

        real_result = analyze_blocks(real_blocks, n)
        sim_result = analyze_blocks(sim_blocks, n)

        real_rows.append((n, real_result['avg_symbol_length'], real_result['efficiency']))
        simulated_rows.append((n, sim_result['avg_symbol_length'], sim_result['efficiency']))

        print(f'-- real image blocks, N={n}')
        print(f'average code length per source symbol = {real_result["avg_symbol_length"]:.6f}')
        print(f'coding efficiency = {real_result["efficiency"]:.6f}')
        print(real_result['preview'])

        print(f'-- simulated memoryless blocks, N={n}')
        print(f'average code length per source symbol = {sim_result["avg_symbol_length"]:.6f}')
        print(f'coding efficiency = {sim_result["efficiency"]:.6f}')
        print(sim_result['preview'])

        save_preview_csv(
            OUTPUT_DIR / f'real_preview_N{n}.csv',
            real_result['preview'],
            real_result['probabilities'],
        )
        save_preview_csv(
            OUTPUT_DIR / f'sim_preview_N{n}.csv',
            sim_result['preview'],
            sim_result['probabilities'],
        )

    with (OUTPUT_DIR / 'summary_real.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['N', 'avg_code_length_per_symbol', 'efficiency'])
        writer.writerows(real_rows)

    with (OUTPUT_DIR / 'summary_simulated.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['N', 'avg_code_length_per_symbol', 'efficiency'])
        writer.writerows(simulated_rows)

    x = [row[0] for row in real_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, [row[1] for row in real_rows], marker='o', label='real image')
    plt.plot(x, [row[1] for row in simulated_rows], marker='o', label='memoryless simulation')
    plt.xlabel('extension N')
    plt.ylabel('avg code length per source symbol')
    plt.title('Huffman average code length')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'avg_code_length.png', dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, [row[2] for row in real_rows], marker='o', label='real image')
    plt.plot(x, [row[2] for row in simulated_rows], marker='o', label='memoryless simulation')
    plt.xlabel('extension N')
    plt.ylabel('coding efficiency')
    plt.title('Huffman coding efficiency')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'coding_efficiency.png', dpi=200)
    plt.close()


if __name__ == '__main__':
    main()
