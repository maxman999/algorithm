import collections
import random
import time
from typing import *
from modules.ListNode import Node


def recursive_dfs(v, discovered=[]):
    discovered.append(v)
    for w in graph[v]:
        if w not in discovered:
            discovered = recursive_dfs(w, discovered)
    return discovered


def myRecursive_dfs(v, result=[]):
    if v not in result:
        result.append(v)
        for w in graph[v]:
            result = myRecursive_dfs(w)
    return result


def iterative_dfs(start_v):
    discovered = []
    stack = [start_v]
    while stack:
        v = stack.pop()
        if v not in discovered:
            discovered.append(v)
            for w in graph[v]:
                stack.append(w)
    return discovered


def myIterative_dfs(start):
    stack = [start]
    result = []

    while stack:
        curr = stack.pop()
        if curr not in result:
            result.append(curr)
            for i in graph[curr]:
                stack.append(i)
    return result


def iterative_bfs(start_v):
    discovered = [start_v]
    queue = [start_v]
    while queue:
        v = queue.pop(0)
        for w in graph[v]:
            if w not in discovered:
                discovered.append(w)
                queue.append(w)
    return discovered


def myIterative_bfs(start):
    queue = [start]
    result = []

    while queue:
        curr = queue.pop(0)
        if curr not in result:
            result.append(curr)
            for adj_v in graph[curr]:
                queue.append(adj_v)

    return result


def timer(func, param):
    start = time.time()
    result = func(param)
    end = time.time()

    print(result)
    print(f"{end - start:.5f} sec")


def cloneGraph(node: Optional['Node']) -> Optional['Node']:
    result_dict = {node.val: Node(node.val, [])}
    stack = [node]

    while stack:
        curr = stack.pop()
        curr_clone = result_dict[curr.val]

        for ngbr in curr.neighbors:
            if ngbr.val not in result_dict:
                result_dict[ngbr.val] = Node(ngbr.val, [])
                stack.append(ngbr)

            curr_clone.neighbors.append(result_dict[ngbr.val])

    return result_dict[node.val]


def numIsLand(grid: List[List[str]]) -> int:
    # 예외 처리
    if not grid:
        return 0

    def dfs(i: int, j: int):
        # 더 이상 땅이 아닌 경우 종료
        if i < 0 or i >= len(grid) or \
                j < 0 or j >= len(grid[0]) or \
                grid[i][j] != '1':
            return

        grid[i][j] = '0'

        # 동서남북 탐색
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)

    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                dfs(i, j)
                # 모든 육지 탐색 후 카운트 1 증가
                count += 1
    return count


def letterCombinations(digits: str, dict) -> List[str]:
    def dfs(index, path):
        # 끝까지 탐색하면 백트래킹
        if len(path) == len(digits):
            result.append(path)
            return

        for c in dic[digits[index]]:
            dfs(index + 1, path + c)

    # 예외처리
    if not digits:
        return []

    dic = dict
    result = []

    dfs(0, "")

    return result


def myNumIslands(grid: List[List[str]]) -> int:
    count = 0

    def backtrack(i, j):
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] == '0':
            return

        if grid[i][j] == '1':
            grid[i][j] = '0'

        backtrack(i + 1, j)
        backtrack(i - 1, j)
        backtrack(i, j + 1)
        backtrack(i, j - 1)

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == "1":
                backtrack(i, j)
                count += 1

    return count


def myLetterCombinations1(digits: str, dict) -> List[str]:
    result = []

    def backTrac(idx, path):
        if len(path) == len(digits):
            result.append(path)
            return

        for char in dict[digits[idx]]:
            backTrac(idx + 1, path + char)

    backTrac(0, "")

    return result


def combine(n: int, k: int) -> List[List[int]]:
    results = []

    def dfs(elements: List[int], start: int, k: int):
        if k == 0:
            results.append(elements[:])
            return

        # 자신 이전의 모든 값을 고정하여 재귀 호출
        for i in range(start, n + 1):
            elements.append(i)
            dfs(elements, i + 1, k - 1)
            elements.pop()

    dfs([], 1, k)
    return results


def permute(nums: List[int]) -> List[List[int]]:
    results = []
    prev_element = []

    def dfs(elements):
        # 리프 노드일 때 결과 추가
        if len(elements) == 0:
            results.append(prev_element[:])
        # 순열 생성 재귀 호출
        for e in elements:
            next_elements = elements[:]
            next_elements.remove(e)

            prev_element.append(e)
            dfs(next_elements)
            prev_element.pop()

    dfs(nums)
    return results


def myPermute(nums: List[int]) -> List[List[int]]:
    result = []
    prev = []

    def dfs(nums: List[int]):
        if len(nums) <= 0:
            result.append(prev[:])
            return

        for e in nums:
            next_els = nums[:]
            next_els.remove(e)

            prev.append(e)
            dfs(next_els)
            prev.pop()

    dfs(nums)

    return result


def combinationSum(candidates: List[int], target: int) -> List[int]:
    result = []

    def dfs(csum, index, path):
        # 종료 조건
        if csum < 0:
            return
        if csum == 0:
            result.append(path)
            return
        # 자신부터 하위 원소 까지의 나열 재귀 호출
        for i in range(index, len(candidates)):
            dfs(csum - candidates[i], i, path + [candidates[i]])

    dfs(target, 0, [])
    return result


def myCombinationSum(candidates: List[int], target: int) -> List[int]:
    result = []
    prev = []

    def backtrack(sum, start):
        if sum > target:
            return
        if sum == target:
            result.append(prev[:])

        for i in range(start, len(candidates)):
            prev.append(candidates[i])
            sum += candidates[i]
            backtrack(sum, i)
            sum -= candidates[i]
            prev.pop()
        return

    backtrack(0, 0)
    return result


def subsets(nums: List[int]) -> List[List[int]]:
    result = []

    def backtrack(idx: int, path: List[int]):
        result.append(path)
        for i in range(idx, len(nums)):
            backtrack(i + 1, path + [nums[i]])

    backtrack(0, [])

    return result


def mySubsets(nums: List[int]) -> List[List[int]]:
    result = []

    def backtrack(idx: int, path: List[int]):
        result.append(path[:])

        for i in range(idx, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result


def findItinerary_recursive(tickets: List[List[str]]) -> List[str]:
    graph = collections.defaultdict(list)
    # 그래프 순서대로 구성
    for a, b in sorted(tickets, reverse=True):
        graph[a].append(b)

    route = []

    def dfs(a):
        # 첫 번째 값을 읽어 어휘 순 방문
        while graph[a]:
            # dfs(graph[a].pop(0))
            dfs(graph[a].pop())
        route.append(a)

    dfs("JFK")
    # 다시 뒤집어 어휘 순 결과로
    return route[::-1]


def findItinerary_iterative(tickets: List[List[str]]) -> List[str]:
    graph = collections.defaultdict(list)
    # 그래프 순서대로 구성
    for a, b in sorted(tickets):
        graph[a].append(b)

    route, stack = [], ['JFK']

    while stack:
        # 반복으로 스택을 구성하되 막히는 부분에서 풀어내는 처리
        while graph[stack[-1]]:
            stack.append(graph[stack[-1]].pop(0))
        route.append(stack.pop())

    return route[::-1]


def myFindItinerary(tickets: List[List[str]]) -> List[str]:
    adj = collections.defaultdict(list)
    for a, b in sorted(tickets):
        adj[a].append(b)

    result = []

    def backtrack(curr):
        result.append(curr)
        while adj[curr]:
            backtrack(adj[curr].pop(0))

        return

    backtrack("JFK")

    return result


def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    graph = collections.defaultdict(list)
    for a, b in prerequisites:
        graph[a].append(b)

    traced = set()

    def dfs(i):
        # 순환 구조이면 False
        if i in traced:
            return False
        traced.add(i)
        for y in graph[i]:
            if not dfs(y):
                return False
        # 탐색 종료 후 순환 노드 삭제
        traced.remove(i)
        return True

    # 순환 구조 판별
    for x in list(graph):
        if not dfs(x):
            return False

    return True


if __name__ == '__main__':
    # prerequisites = [[0, 10], [3, 18], [5, 5], [6, 11], [11, 14], [13, 1], [15, 1], [17, 4]]
    # prerequisites1 = [[0, 1], [1, 0]]
    # prerequisites2 = [[1, 0]]
    # print(canFinish(8, prerequisites))
    # print(canFinish(2, prerequisites1))
    # print(canFinish(1, prerequisites2))

    # ["JFK","MUC","LHR","SFO","SJC"]
    tickets1 = [["JFK", "SFO"], ["JFK", "ATL"], ["SFO", "ATL"], ["ATL", "JFK"], ["ATL", "SFO"]]
    tickets2 = [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
    tickets3 = [["JFK", "NRT"], ["JFK", "KUL"], ["KUL", "JFK"]]
    tickets4 = [["JFK", "NRT"], ["JFK", "KUL"], ["NRT", "JFK"]]

    print(findItinerary_recursive(tickets3))

    # print(mySubsets(nums))
    # print(subsets(nums))

    # print(combinationSum(candidates, target))

    # grid = [
    #     ['1', '1', '1', '1', '0'],
    #     ['1', '1', '0', '1', '0'],
    #     ['1', '1', '0', '0', '0'],
    #     ['0', '0', '0', '0', '0']
    # ]
    #
    # dic = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl",
    #        "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}

    graph = {
        1: [2, 3, 4, 6],
        2: [5, 1],
        3: [5, 4],
        4: [5, 7],
        5: [7, 8, 9],
        6: [3],
        7: [3],
        8: [1, 2, 3],
        9: [6, 7, 8]
    }

    # adjList = [[2, 4], [1, 3], [2, 4], [1, 3]]
    # head = Node(1, [Node(2), Node(4)])
    # head.neighbors[0].neighbors = [head, Node(3, head.neighbors)]
    # head.neighbors[1].neighbors = [head, head.neighbors[0].neighbors[1]]
    # print("result : ", cloneGraph(head))
