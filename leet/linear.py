import collections
from typing import *
import heapq
import functools
import itertools
import re
import sys
import math
import bisect
from modules import ListNode


if __name__ == '__main__':
    def trap(height: List[int]) -> int:
        if not height:
            return 0

        volume = 0
        left, right = 0, len(height) - 1
        left_max, right_max = height[left], height[right]

        while left < right:
            left_max, right_max = max(height[left], left_max), max(height[right], right_max)
            # 더 높은 쪽을 향해 투 포인터 이동
            if left_max <= right_max:
                volume += left_max - height[left]
                left += 1
            else:
                volume += right_max - height[right]
                right -= 1
        return volume


    def trap_stack(height: List[int]) -> int:
        stack = []
        volume = 0

        for i in range(len(height)):
            # 변곡점을 만나는 경우
            while stack and height[i] > height[stack[-1]]:
                # 스택에서 꺼낸다
                top = stack.pop()

                if not len(stack):
                    break
                # 이전과의 차이만큼 물 높이 처리
                distance = i - stack[-1] - 1
                waters = min(height[i], height[stack[-1]]) - height[top]

                volume += distance * waters

            stack.append(i)
        return volume


    height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    print(trap_stack(height))


    def productExceptSelf(nums: List[int]) -> List[int]:
        out = []
        p = 1
        # 왼쪽 곱셈
        for i in range(0, len(nums)):
            out.append(p)
            p = p * nums[i]
        p = 1
        # 왼쪽 곱셈 결과에 오른쪽 값을 차례대로 곱셈
        for i in range(len(nums) - 1, 0 - 1, -1):
            out[i] = out[i] * p
            p = p * nums[i]
        return out


    q3_input = [2, 3, 4, 5]
    print(productExceptSelf(q3_input))


    def maxProfit(prices: List[int]) -> int:
        profit = 0
        min_price = sys.maxsize

        # 최솟값과 최댓값을 계속 갱신
        for price in prices:
            min_price = min(min_price, price)
            profit = max(profit, price - min_price)

        return profit


    q4_input = [7, 1, 5, 3, 6, 4]
    print(maxProfit(q4_input))


    def isPalindrome(head: ListNode) -> bool:
        q: Deque = collections.deque()
        if not head:
            return True

        node = head
        while node is not None:
            q.append(node.val)
            node = node.next

        # 리스트 변환
        while len(q) > 1:
            if q.popleft() != q.pop():
                return False

        return True

    def isPalindromeUsingRunner(head: ListNode) -> bool :
        rev = None
        slow = fast = head
        #런너를 이용해 역순 연결리스트 구성
        while fast and fast.next:
            fast = fast.next.next
            rev, rev.next, slow = slow, rev, slow.next
        if fast:
            slow = slow.next
        #팰린드롬 여부 확인
        while rev and rev.val == slow.val:
            slow, rev = slow.next, rev.next
        return not rev
