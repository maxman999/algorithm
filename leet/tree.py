import collections
from typing import *
import null as null
from modules.Tree import TreeNode

from modules.Tree import TreeGenerator


def maxDepth(root: Optional[TreeNode]) -> int:
    if root is None:
        return 0

    queue = collections.deque([root])
    depth = 0

    while queue:
        depth += 1
        # 큐 연산 추출 노드의 자식 노드 삽입
        for _ in range(len(queue)):
            cur_root = queue.popleft()
            if cur_root.left:
                queue.append(cur_root.left)
            if cur_root.right:
                queue.append(cur_root.right)

    return depth


def myDiameterOfBinaryTree(root: Optional[TreeNode]) -> int:
    return 0


class Solution:
    longest = 0

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        def dfs(node: TreeNode) -> int:
            if not node:
                return -1
            # 왼쪽, 오른쪽의 각 리프 노드까지 탐색
            left = dfs(node.left)
            right = dfs(node.right)

            # 가장 긴 경로
            self.longest = max(self.longest, left + right + 2)
            # 상태값
            return max(left, right) + 1

        dfs(root)
        return self.longest


if __name__ == '__main__':
    root = TreeGenerator([1, 2, 3, 4, 5])
    print(Solution().diameterOfBinaryTree(root))

    # root = TreeGenerator([3, 9, 20, null, null, 15, 7])
    # print(maxDepth(root))
