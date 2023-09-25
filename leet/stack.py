import collections
from typing import *
import heapq
import functools
import itertools
import re
import sys
import math
import bisect
from modules.ListNode import NodeGenerator
from modules.ListNode import ListNode


def validateParenthesis(s: str) -> bool:
    stack = []
    parenthesis_table = {
        '(': ')',
        '{': '}',
        '[': ']',
    }

    for char in s:
        if char in parenthesis_table:
            stack.append(char)
        elif not stack or char != parenthesis_table[stack.pop()]:
            return False
    return len(stack) == 0


def weatherForecast(T: List[int]) -> List:
    answer = [0] * len(T)
    stack = []
    for i, cur in enumerate(T):
        # 현재 온도가 스택 값보다 높다면 정답처리
        while stack and cur > T[stack[-1]]:
            last = stack.pop()
            answer[last] = i - last
        stack.append(i)
    return answer


def mergeKLists(lists: List[ListNode]) -> ListNode:
    root = result = ListNode(None)
    heap = []

    # 각 연결리스트의 루트를 힙에 저장
    for i in range(len(lists)):
        if lists[i]:
            heapq.heappush(heap, (lists[i].val, i, lists[i]))

    # 힙 추출 이후 다음 노드는 다시 저장
    while heap:
        node = heapq.heappop(heap)
        idx = node[1]
        result.next = node[2]

        result = result.next
        if result.next:
            heapq.heappush(heap, (result.next.val, idx, result.next))

    return root.next


def myMergeKLists(lists: List[ListNode]) -> ListNode:
    heap = []
    root = result = ListNode(None)

    for i in range(len(lists)):
        node = lists[i]
        heapq.heappush(heap, (node.val, i, node))

    while heap:
        popped = heapq.heappop(heap)
        prior_node = popped[2]
        idx = popped[1]

        result.next = prior_node
        result = result.next

        if prior_node.next:
            heapq.heappush(heap, (prior_node.next.val, idx, prior_node.next))

    return root.next


if __name__ == '__main__':
    list1 = NodeGenerator(1, 4, 5).head
    list2 = NodeGenerator(1, 3, 4).head
    list3 = NodeGenerator(2, 6).head
    target_lists = [list1, list2, list3]

    merged_list = myMergeKLists(target_lists)

    while merged_list:
        print(merged_list.val)
        merged_list = merged_list.next
