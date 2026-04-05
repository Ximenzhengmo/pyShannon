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

from probability import conditional_distribution, conditional_entropy, entropy
from source import Source


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'experiment1_outputs'


def build_memoryless_source():
    return Source([0.4231, 0.5769], x=2, symbol=['0', '1'], source_type='MemoryLess')


def build_markov_source():
    src = Source(
        [0.25, 0.75, 0.60, 0.40, 0.90, 0.10, 0.20, 0.80],
        x=2,
        m=2,
        symbol=['0', '1'],
        source_type='Markov',
    )
    src.setup(2, [0.2308, 0.1923, 0.1923, 0.3846])
    return src


def theoretical_joint_entropies(source, max_k):
    values = []
    for k in range(1, max_k + 1):
        p = source.prob_topk(k).reshape(-1)
        values.append(float(entropy(p)))
    return np.array(values, dtype=np.float64)


def conditional_entropies_from_joint_entropies(joint_entropies):
    cond = [joint_entropies[0]]
    for idx in range(1, len(joint_entropies)):
        cond.append(joint_entropies[idx] - joint_entropies[idx - 1])
    return np.array(cond, dtype=np.float64)


def simulate_conditional_entropy(source, k, sample_size, rng):
    p_joint = source.prob_topk(k).reshape(-1)
    counts = rng.multinomial(sample_size, p_joint)
    p_joint_hat = counts.astype(np.float64) / sample_size

    if k == 1:
        return float(entropy(p_joint_hat))

    joint_2d = p_joint_hat.reshape(-1, source.x)
    prefix = joint_2d.sum(axis=1, keepdims=True)
    p_last_given_prefix = conditional_distribution(joint_2d, prefix)
    return float(conditional_entropy(p_last_given_prefix.reshape(-1), joint_2d.reshape(-1)))


def run_source_experiment(name, source, sample_sizes, max_k, rng, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    joint_entropies = theoretical_joint_entropies(source, max_k)
    conditional_theory = conditional_entropies_from_joint_entropies(joint_entropies)

    simulated = {}
    for sample_size in sample_sizes:
        simulated[sample_size] = np.array(
            [simulate_conditional_entropy(source, k, sample_size, rng) for k in range(1, max_k + 1)],
            dtype=np.float64,
        )

    csv_path = output_dir / f'{name}_conditional_entropy.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['N', 'theory'] + [f'sim_{n}' for n in sample_sizes]
        writer.writerow(header)
        for idx in range(max_k):
            row = [idx + 1, conditional_theory[idx]] + [simulated[n][idx] for n in sample_sizes]
            writer.writerow(row)

    plt.figure(figsize=(8, 5))
    x = np.arange(1, max_k + 1)
    plt.plot(x, conditional_theory, marker='o', linewidth=2, label='theory')
    for sample_size in sample_sizes:
        plt.plot(x, simulated[sample_size], marker='o', linestyle='--', label=f'n={sample_size}')
    plt.xlabel('N')
    plt.ylabel('Conditional entropy')
    plt.title(f'{name} conditional entropy')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_conditional_entropy.png', dpi=200)
    plt.close()

    print(f'[{name}] theory conditional entropies')
    print(np.round(conditional_theory, 6))
    for sample_size in sample_sizes:
        print(f'[{name}] simulated conditional entropies (n={sample_size})')
        print(np.round(simulated[sample_size], 6))


def main():
    parser = argparse.ArgumentParser(description='Experiment 1 rewritten with pyShannon interfaces.')
    parser.add_argument('--max-k', type=int, default=10)
    parser.add_argument('--sample-sizes', type=int, nargs='*', default=[1000, 10000, 100000])
    parser.add_argument('--seed', type=int, default=20260405)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_source_experiment(
        name='memoryless_source',
        source=build_memoryless_source(),
        sample_sizes=args.sample_sizes,
        max_k=args.max_k,
        rng=rng,
        output_dir=OUTPUT_DIR,
    )
    run_source_experiment(
        name='markov_source',
        source=build_markov_source(),
        sample_sizes=args.sample_sizes,
        max_k=args.max_k,
        rng=rng,
        output_dir=OUTPUT_DIR,
    )


if __name__ == '__main__':
    main()

