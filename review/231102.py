import collections
import heapq
import sys
from typing import *

from modules.ListNode import ListNode, NodeGenerator


def myTrap(height: List[int]) -> int:
    left = 0
    right = len(height) - 1
    max_height = 0
    volume = 0

    while left < right:
        if height[left] <= height[right]:
            max_height = max(max_height, height[left])
            volume += max_height - height[left]
            left += 1
        else:
            max_height = max(max_height, height[right])
            volume += max_height - height[right]
            right -= 1
    return volume


def myThreeSum(nums: List[int]) -> List[List[int]]:
    result = []
    nums.sort()
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left = i + 1
        right = len(nums) - 1

        while left < right:
            sum = nums[i] + nums[left] + nums[right]
            if sum > 0:
                right -= 1
            elif sum < 0:
                left += 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1

                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    return result


def myArrayPairSum(nums: List[int]) -> int:
    nums.sort()
    sum = 0
    for i in range(0, len(nums), 2):
        sum += nums[i]
    return sum


def myProductExceptSelf(nums: List[int]) -> List[int]:
    result = []
    p = 1
    for i in range(len(nums)):
        result.append(p)
        p = p * nums[i]

    p = 1
    for i in range(len(nums) - 1, 0 - 1, -1):
        result[i] = result[i] * p
        p = p * nums[i]
    return result


def myMaxProfit(prices: List[int]) -> int:
    max_profit = 0
    min_price = sys.maxsize

    for i in range(len(prices)):
        min_price = min(min_price, prices[i])
        max_profit = max(max_profit, prices[i] - min_price)

    return max_profit


def myIsValid(s: str) -> bool:
    parentheses_table = {
        "(": ")",
        "{": "}",
        "[": "]",
    }

    stack = []

    for char in s:
        if char in parentheses_table:
            stack.append(char)
        else:
            if char != parentheses_table[stack.pop()]:
                return False
    return len(stack) == 0


def myDailyTemperatures(temperatures: List[int]) -> List[int]:
    result = [0] * len(temperatures)
    stack = []
    for i, cur in enumerate(temperatures):
        while stack and cur > temperatures[stack[-1]]:
            prev = stack.pop()
            result[prev] = i - prev
        stack.append(i)

    return result


def myMergeKLists(lists: List[Optional[ListNode]]):
    heap = []
    for i in range(len(lists)):
        if lists[i]:
            heapq.heappush(heap, (lists[i].val, i, lists[i]))

    head = result = ListNode(None)
    while heap:
        curr = heapq.heappop(heap)
        result.next = curr[2]
        result = result.next

        if result.next:
            heapq.heappush(heap, (curr[2].next.val, curr[1], curr[2].next))

    return head.next


def myNumJewelsInStones(jewels: str, stones: str) -> int:
    result = 0
    dict = {}
    for j in jewels:
        dict[j] = 0

    for s in stones:
        if s in dict:
            result += 1
    return result


def myLengthOfLongestSubstring(s: str) -> int:
    used = {}
    max_length = start = 0
    for i, char in enumerate(s):
        if char in dict and start <= used[char]:
            start = used[char] + 1
        else:
            max_length = max(max_length, i - start + 1)
        used[char] = i
    return max_length


if __name__ == "__main__":
    s = "pwwkew"
    print(myLengthOfLongestSubstring(s))

    # jewels = "aA"
    # stones = "aAAbbbb"
    # print(myNumJewelsInStones(jewels, stones))

    # list1 = NodeGenerator(1, 2, 2).head
    # list2 = NodeGenerator(1, 1, 2).head
    # target_lists = [list1, list2]
    # merged_list = myMergeKLists(target_lists)
    #
    # while merged_list:
    #     print(merged_list.val)
    #     merged_list = merged_list.next

    # temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
    # print(myDailyTemperatures(temperatures))

    # s = "("
    # print(myIsValid(s))

    # prices = [7, 1, 5, 3, 6, 4]
    # print(myMaxProfit(prices))

    # nums = [1, 2, 3, 4]
    # print(myProductExceptSelf(nums))

    # nums = [1, 4, 3, 2]
    # print(myArrayPairSum(nums))

    # nums = [-1,0,1,2,-1,-4]
    # print(myThreeSum(nums))

    # height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    # print(myTrap(height))
