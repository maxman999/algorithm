import collections
from typing import *
import heapq
import functools
import itertools
import re
import sys
import math
import bisect
import numbers

from modules.ListNode import ListNode
from modules.ListNode import NodeGenerator


def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    if (not l1) or (l2 and l1.val > l2.val):
        l1, l2 = l2, l1
    if l1:
        l1.next = mergeTwoLists(l1.next, l2)
    return l1


def reverseList(head: ListNode) -> ListNode:
    prev = None

    while head:
        next_node = head.next
        head.next = prev
        prev = head
        head = next_node
    return prev


def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    root = head = ListNode(0)
    carry = 0
    while l1 or l2 or carry:
        sum = 0
        # 두 입력값의 합 계산
        if l1:
            sum += l1.val
            l1 = l1.next
        if l2:
            sum += l2.val
            l2 = l2.next

        # 몫(자리올림수)과 나머지(값) 계산
        carry, val = divmod(sum + carry, 10)
        head.next = ListNode(val)
        head = head.next
    return root.next


def swapPairs(head: ListNode) -> ListNode:
    root = prev = ListNode(None)
    prev.next = head
    while head and head.next:
        # b가 a(head)를 가리키도록 할당
        b = head.next
        head.next = b.next
        b.next = head

        # prev가 b를 가리키도록 할당
        prev.next = b

        # 다음번 비교를 위해 이동
        head = head.next
        prev = prev.next.next
    return root.next


def swapPairsRecursive(head: ListNode) -> ListNode:
    if head and head.next:
        p = head.next
        # 스왑된 값 리턴 받
        head.next = swapPairsRecursive(p.next)
        p.next = head
        return p
    return head


def oddEvenList(head: ListNode) -> ListNode:
    # 예외처리
    if head is None:
        return None

    odd = head
    even = head.next
    even_head = head.next

    # 반복하면서 홀짝 노드처리
    while even and even.next:
        odd.next, even.next = odd.next.next, even.next.next
        odd, even = odd.next, even.next

    # 홀수 노드의 마지막을 짝수 헤드로 연결
    odd.next = even_head

    return head


def reverseBetween(head: ListNode, m: numbers, n: numbers) -> ListNode:
    # 예외 처리
    if not head or m == n:
        return head
    root = start = ListNode(None)
    root.next = head
    # start, end 지정
    for _ in range(m - 1):
        start = start.next
    end = start.next

    # 반복하면서 노드 차례대로 뒤집기
    for _ in range(n - m):
        tmp = start.next
        start.next = end.next
        end.next = end.next.next
        start.next.next = tmp
    return root.next


def factorial(num: numbers) -> numbers:
    rtv = 1
    if num >= 1:
        rtv = num * factorial(num - 1)
    return rtv


if __name__ == '__main__':
    list1 = NodeGenerator(1, 2, 3, 4, 5).head
    result = reverseBetween(list1, 2, 4)

    while result:
        print(result.val)
        result = result.next
