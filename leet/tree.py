import collections
import sys
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


class Codec:

    def serialize(self, root: TreeNode) -> str:
        queue = collections.deque([root])
        result = ['#']
        # 트리 BFS 직렬화

        while queue:
            node = queue.popleft()
            if node:
                queue.append(node.left)
                queue.append(node.right)

                result.append(str(node.val))
            else:
                result.append('#')
        return ' '.join(result)

    def deserialize(self, data: str) -> TreeNode:
        # 예외 처리
        if data == '# #':
            return None

        nodes = data.split()

        root = TreeNode(int(nodes[1]))
        queue = collections.deque([root])
        index = 2

        # 빠른 런너처럼 자식 노드 결과를 먼저 확인 후 큐 삽입
        while queue:
            node = queue.popleft()
            if nodes[index] != '#':
                node.left = TreeNode(int(nodes[index]))
                queue.append(node.left)
            index += 1

            if nodes[index] != '#':
                node.right = TreeNode(int(nodes[index]))
                queue.append(node.right)
            index += 1

        return root


def myIsBalanced(root: Optional[TreeNode]) -> bool:
    result = [True]

    def bfs(node: TreeNode):
        if not node or node.val is null:
            return 0

        left_height = bfs(node.left)
        right_height = bfs(node.right)

        if abs(left_height - right_height) > 1:
            result[0] = False

        return max(left_height, right_height) + 1

    bfs(root)

    return result[0]


def findMinHeightTrees(n: int, edges: List[List[int]]) -> List[int]:
    if n <= 1:
        return [0]

    # 양방향 그래프 구성
    graph = collections.defaultdict(list)
    for i, j in edges:
        graph[i].append(j)
        graph[j].append(i)

    # 첫번째 리프 노드 추가
    leaves = []
    for i in range(n + 1):
        if len(graph[i]) == 1:
            leaves.append(i)

    # 루트 노드만 남을 때까지 반복 제거
    while n > 2:
        n -= len(leaves)
        new_leaves = []
        for leaf in leaves:
            neighbor = graph[leaf].pop()
            graph[neighbor].remove(leaf)

            if len(graph[neighbor]) == 1:
                new_leaves.append(neighbor)
        leaves = new_leaves

    return leaves


def sortedArrayToBST(nums: List[int]) -> Optional[TreeNode]:
    if not nums:
        return None

    mid = len(nums) // 2

    # 분할 정복으로 이진 검색 결과 트리 구성
    node = TreeNode(nums[mid])
    node.left = sortedArrayToBST(nums[:mid])
    node.right = sortedArrayToBST(nums[mid + 1:])

    return node


def mySortedArrayToBST(nums: List[int]) -> Optional[TreeNode]:
    if len(nums) < 1:
        return

    mid = len(nums) // 2

    root = TreeNode(nums[mid])

    root.left = mySortedArrayToBST(nums[:mid])
    root.right = mySortedArrayToBST(nums[(mid + 1):])

    return root


class Solution:
    val: int = 0

    def bstToGst(self, root: TreeNode) -> TreeNode:
        # 중위 순회 노드 값 누적
        if root:
            self.bstToGst(root.right)
            if root.val is not null:
                self.val += root.val
            root.val = self.val
            self.bstToGst(root.left)
        return root


def myBstToGst(root: TreeNode) -> TreeNode:
    curr_sum = 0

    def dfs(node: TreeNode):
        if not node:
            return

        nonlocal curr_sum

        dfs(node.right)
        if node.val is not null:
            curr_sum += node.val
        node.val = curr_sum
        dfs(node.left)

        return node

    return dfs(root)


def myRangeSumBST(root: Optional[TreeNode], low: int, high: int) -> int:
    result = 0

    def dfs(node: TreeNode):
        if not node or node.val is null:
            return

        nonlocal result
        if node.val > high:
            dfs(node.left)
        elif node.val < low:
            dfs(node.right)
        else:
            dfs(node.right)
            dfs(node.left)

        if low <= node.val <= high:
            result += node.val

        return node

    dfs(root)

    return result


if __name__ == '__main__':
    # root = [10, 5, 15, 3, 7, null, 18]
    # low = 7
    # high = 15
    # print(myRangeSumBST(TreeGenerator(root), low, high))

    # result = myBstToGst(TreeGenerator([4, 1, 6, 0, 2, 5, 7, null, null, null, 3, null, null, null, 8]))
    # print(result)
    # Solution().bstToGst(TreeGenerator([4, 1, 6, 0, 2, 5, 7, null, null, null, 3, null, null, null, 8]))

    # sortedArrayToBST([1, 2, 3, 4, 5, 6, 7, 8, 9])
    result = mySortedArrayToBST([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(result)

    # findMinHeightTrees(6, [[3, 0], [3, 1], [3, 2], [3, 4], [5, 4]])

    # print(myIsBalanced(TreeGenerator([1, 2, 2, 3, 3, null, null, 4, 4])))

    # print(Codec().serialize(TreeGenerator([1, 2, 3, '#', '#', 4, 5])))
    # print(Codec().deserialize('# 1 2 3 # # 4 5 # # # # # # # #'))

    # root1 = TreeGenerator([1, 3, 2, 5])
    # root2 = TreeGenerator([2, 1, 3, None, 4, None, 7])
    # mergeTrees(root1, root2)

    # root = TreeGenerator([4, 2, 7, 1, 3, 6, null])
    # myInvertTree(root)

    # root = TreeGenerator([1, null, 1, 1, 1, 1, 1, 1])
    # print(myLongestUnivaluePath(root))

    # root = TreeGenerator([1, 2, 3, 4, 5])
    # print(Solution().diameterOfBinaryTree(root))

    # root = TreeGenerator([3, 9, 20, null, null, 15, 7])
    # print(maxDepth(root))
