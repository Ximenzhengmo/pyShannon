import numpy as np

from probability import (
    self_information,
    mutual_information,
    Entropy,
    conditional_entropy,
    mean_contitional_mutial_information,
    mean_mutual_information,
)
from chain import MarkovChain, MemoryLessChain
from source import Source
from channel import Channel
from codec import huffman_encoding, HuffmanCodec


np.set_printoptions(precision=4, suppress=True)


def print_section(title):
    print("\n" + "=" * 16 + f" {title} " + "=" * 16)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1) Basic feature examples
    # ------------------------------------------------------------------

    print_section("1.1 信息量与熵")

    pX = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    pY = np.array([0.6, 0.4], dtype=np.float32)
    pXY = np.array(
        [
            [0.42, 0.08],
            [0.12, 0.18],
            [0.06, 0.14],
        ],
        dtype=np.float32,
    )

    # p(X|Y)
    pX_given_Y = pXY / pY.reshape(1, -1)

    print("self_information([0.5, 0.25, 0.125]) =", self_information([0.5, 0.25, 0.125]))
    print("mutual_information(pX, pX|Y=y0) =", mutual_information(pX, pX_given_Y[:, 0]))

    hx = Entropy(pX)
    hy = Entropy(pY)
    hxy = Entropy(pXY.reshape(-1))
    h_x_given_y = conditional_entropy(pX_given_Y.reshape(-1), pXY.reshape(-1))
    ixy = mean_mutual_information(pX, pY, pXY.reshape(-1))
    i_x_single_y = mean_contitional_mutial_information(pX, pX_given_Y[:, 0])

    print("H(X) =", hx)
    print("H(Y) =", hy)
    print("H(X,Y) =", hxy)
    print("H(X|Y) =", h_x_given_y)
    print("I(X;Y) =", ixy)
    print("I(X;y0) =", i_x_single_y)

    print_section("1.2 MemoryLessChain")

    mlc = MemoryLessChain([0.25, 0.75], x=2)
    print("prob_topk(3):")
    print(mlc.prob_topk(3))
    print("prob_k([1, 3, 5]):")
    print(mlc.prob_k([1, 3, 5]))

    print_section("1.3 MarkovChain")

    mc = MarkovChain(
        [
            0.25, 0.75,
            0.60, 0.40,
            0.90, 0.10,
            0.20, 0.80,
        ],
        x=2,
        m=2,
    )
    mc.setup(2, [0.2308, 0.1923, 0.1923, 0.3846])
    print("prob_topk(4):")
    print(mc.prob_topk(4))
    print("prob_k([1, 2, 8]):")
    print(mc.prob_k([1, 2, 8]))

    print_section("1.4 两类信源")

    # Memoryless source
    src_mem = Source([0.75, 0.25], symbol=["0", "1"], source_type="MemoryLess")
    seq_iter = src_mem.random_sequence_gen(
        k=9,
        n=3,
        sep="-",
        prior="0-1-@-0-1-@-0",
        placeholder="@",
    )
    print("MemoryLessSource samples:")
    for s in seq_iter:
        print(s)

    # Markov source
    src_mk = Source(
        [
            0.25, 0.75,
            0.60, 0.40,
            0.90, 0.10,
            0.20, 0.80,
        ],
        x=2,
        m=2,
        symbol=["0", "1"],
        source_type="Markov",
    )
    src_mk.setup(2, [0.2308, 0.1923, 0.1923, 0.3846])
    sym_iter = src_mk.symbol_gen(k=12, prior="01@01@0", placeholder="@")
    print("MarkovSource symbols:")
    print(" ".join(list(sym_iter)))

    print_section("1.5 两类信道")

    # Memoryless channel
    dmc = Channel(
        [
            0.90, 0.10,
            0.10, 0.90,
        ],
        x=2,
        input_symbol=["0", "1"],
        output_symbol=["0", "1"],
        channel_type="MemoryLess",
    )

    pX_ch = [0.5, 0.5]
    x_bits = "01011001"
    y_bits = dmc.channel_pass(x_bits, rng=np.random.default_rng(7))

    print("DMC p(Y):", dmc.output_prob(pX_ch))
    print("DMC p(X,Y):")
    print(dmc.joint_prob(pX_ch))
    print("DMC I(X;Y):", dmc.mean_mutual_information(pX_ch))
    print(f"DMC pass: {x_bits} -> {y_bits}")
    print("DMC P(y|x):", dmc.channel_prob(x_bits, y_bits))

    # Markov channel
    mkc = Channel(
        [
            0.95, 0.05,
            0.20, 0.80,
            0.80, 0.20,
            0.10, 0.90,
        ],
        x=2,
        m=2,
        input_symbol=["0", "1"],
        output_symbol=["0", "1"],
        channel_type="Markov",
    )

    x_bits_m = "0101101"
    y_bits_m = mkc.channel_pass(x_bits_m, prior="0", rng=np.random.default_rng(7))
    print(f"MarkovChannel pass: {x_bits_m} -> {y_bits_m}")
    print("MarkovChannel P(y|x):", mkc.channel_prob(x_bits_m, y_bits_m, prior="0"))

    print_section("1.6 哈夫曼编码")

    codes_from_prob = huffman_encoding([0.20, 0.19, 0.18, 0.17, 0.15, 0.10, 0.01])
    print("codes from probabilities:", codes_from_prob)

    sample_symbols = ["A", "A", "B", "C", "A", "D", "B", "A", "C", "A", "B", "A"]
    codec_basic = HuffmanCodec.from_data(sample_symbols)
    bits_basic = codec_basic.encode(sample_symbols)
    restore_basic = codec_basic.decode(bits_basic, return_list=True)
    print("codes from real data:", codec_basic.codes)
    print("basic decode equal:", restore_basic == sample_symbols)

    # ------------------------------------------------------------------
    # 2) Joint examples
    # ------------------------------------------------------------------

    print_section("2.1 信源 -> 哈夫曼编解码")

    src_joint = Source(
        [
            0.70, 0.20, 0.07, 0.03,
            0.25, 0.50, 0.20, 0.05,
            0.20, 0.25, 0.45, 0.10,
            0.30, 0.20, 0.10, 0.40,
        ],
        x=4,
        m=1,
        symbol=["A", "B", "C", "D"],
        source_type="Markov",
    )
    src_joint.setup(1, [0.55, 0.25, 0.15, 0.05])
    stream = list(src_joint.symbol_gen(k=60, prior="A@", placeholder="@"))

    codec_joint = HuffmanCodec.from_data(stream)
    bits_joint = codec_joint.encode(stream)
    restore_joint = codec_joint.decode(bits_joint, return_list=True)

    print("stream head:", "".join(stream[:30]), "...")
    print("decode equal:", restore_joint == stream)
    fixed_bits = int(np.ceil(np.log2(len(codec_joint.codes)))) * len(stream)
    print(f"fixed bits={fixed_bits}, huffman bits={len(bits_joint)}")

    print_section("2.2 信源 -> 哈夫曼编码 -> 信道 -> 哈夫曼解码")

    # deterministic error-free bit channel for demonstration
    bit_channel = Channel(
        [
            1.0, 0.0,
            0.0, 1.0,
        ],
        x=2,
        input_symbol=["0", "1"],
        output_symbol=["0", "1"],
        channel_type="MemoryLess",
    )

    tx_bits = bits_joint
    rx_bits = bit_channel.channel_pass(tx_bits, rng=np.random.default_rng(2026))
    rx_stream = codec_joint.decode(rx_bits, return_list=True)

    print("tx bits == rx bits:", tx_bits == rx_bits)
    print("decoded stream equal:", rx_stream == stream)
    print("P(rx|tx):", bit_channel.channel_prob(tx_bits, rx_bits))
