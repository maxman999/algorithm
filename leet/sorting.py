import collections
from typing import *


def myMergeSort(nums: list):
    if len(nums) < 2:
        return nums

    mid = len(nums) // 2
    temp = []

    left = myMergeSort(nums[:mid])
    right = myMergeSort(nums[mid:])

    while len(temp) < len(nums):
        if not left:
            temp += right
        elif not right:
            temp += left
        elif left[0] < right[0]:
            temp.append(left.pop(0))
        else:
            temp.append(right.pop(0))

    return temp


if __name__ == '__main__':
    nums = [7, 1, 4, 9, 2, 5, 6, 3, 8]
    print(myMergeSort(nums))
