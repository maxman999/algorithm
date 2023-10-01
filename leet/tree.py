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


def myLongestUnivaluePath(root: Optional[TreeNode]) -> int:
    longest = [0]

    def dfs(node: TreeNode):
        if not node:
            return 0

        temp = node.val
        left = dfs(node.left)
        right = dfs(node.right)

        if node.left and node.left.val == node.val:
            left += 1
        else:
            left = 0

        if node.right and node.right.val == node.val:
            right += 1
        else:
            right = 0

        longest[0] = max(longest[0], left + right)

        return max(left, right)

    dfs(root)

    return longest[0]


def myInvertTree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    Q = collections.deque()
    Q.append(root)

    while Q:
        curr = Q.popleft()
        if not curr:
            continue
        temp = curr.left
        curr.left = curr.right
        curr.right = temp

        Q.append(curr.left)
        Q.append(curr.right)

    return root


def mergeTrees(t1: Optional[TreeNode], t2: Optional[TreeNode]) -> Optional[TreeNode]:
    if t1 and t2:
        node = TreeNode(t1.val + t2.val)
        node.left = mergeTrees(t1.left, t2.left)
        node.right = mergeTrees(t1.right, t2.right)
        return node
    else:
        return t1 or t2


if __name__ == '__main__':
    root1 = TreeGenerator([1, 3, 2, 5])
    root2 = TreeGenerator([2, 1, 3, None, 4, None, 7])
    mergeTrees(root1, root2)

    # root = TreeGenerator([4, 2, 7, 1, 3, 6, null])
    # myInvertTree(root)

    # root = TreeGenerator([1, null, 1, 1, 1, 1, 1, 1])
    # print(myLongestUnivaluePath(root))

    # root = TreeGenerator([1, 2, 3, 4, 5])
    # print(Solution().diameterOfBinaryTree(root))

    # root = TreeGenerator([3, 9, 20, null, null, 15, 7])
    # print(maxDepth(root))
