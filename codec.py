from utils import *
from chain import MarkovChain, MemoryLessChain
import heapq
from collections import defaultdict


# 定义哈夫曼树节点类
class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

# 构建哈夫曼树
def build_huffman_tree(probabilities):
    # 创建初始节点列表
    nodes = []
    for i, prob in enumerate(probabilities):
        if prob > 0:
            node = HuffmanNode(i, prob)
            heapq.heappush(nodes, node)

    # 构建哈夫曼树
    while len(nodes) > 1:
        left = heapq.heappop(nodes)
        right = heapq.heappop(nodes)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(nodes, merged)

    return nodes[0]

# 生成哈夫曼编码
def generate_huffman_codes(root):
    codes = {}

    def traverse(node, code):
        if node.char is not None:
            codes[node.char] = code
            return
        traverse(node.left, code + '0')
        traverse(node.right, code + '1')

    traverse(root, '')
    return codes

# 主函数
def huffman_encoding(probabilities):
    # 构建哈夫曼树
    root = build_huffman_tree(probabilities)
    # 生成哈夫曼编码
    codes = generate_huffman_codes(root)
    return codes

# 示例使用
if __name__ == "__main__":
    print( Entropy( [0.20,0.19,0.18,0.17,0.15,0.10,0.01] ) )