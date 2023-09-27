from typing import *


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def TreeGenerator(nodes: List[int]) -> Optional[TreeNode]:
    if not nodes:
        return None

    nodes = [TreeNode(val) if val is not None else None for val in nodes]
    root = nodes[0]

    for i in range(len(nodes)):
        if nodes[i] is not None:
            left_idx = 2 * i + 1
            right_idx = 2 * i + 2
            if left_idx < len(nodes):
                nodes[i].left = nodes[left_idx]
            if right_idx < len(nodes):
                nodes[i].right = nodes[right_idx]

    return root
