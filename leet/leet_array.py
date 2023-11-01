from typing import *


def myTwoSum(nums: List[int], target: int) -> List[int]:
    result = []
    for i in range(len(nums)):
        sub = target - nums[i]
        if sub in nums:
            sub_index = nums.index(sub)
            if sub_index != i:
                result.append(i)
                result.append(sub_index)
                break
    return result


def twoSum(nums: List[int], target: int) -> List[int]:
    # for i, n in enumerate(nums):
    #     complement = target - n
    #
    #     if complement in nums[i + 1:]:
    #         return [nums.index(n), nums[i + 1:].index(complement) + (i + 1)]

    nums_map = {}
    # 키와 값을 바꿔서 딕셔너리로 저장
    for i, num in enumerate(nums):
        nums_map[num] = i
    # 타겟에서 첫 번째 수를 뺀 결과를 키로 조회
    for i, num in enumerate(nums):
        if target - num in nums_map and i != nums_map[target - num]:
            return [i, nums_map[target - num]]


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


def threeSum(nums: List[int]) -> List[List[int]]:
    result = []
    nums.sort()

    for i in range(len(nums) - 2):
        # 중복된 값 건너 뛰기
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # 간격을 좁혀가며 sum 계산
        left, right = i + 1, len(nums) - 1
        while left < right:
            sum = nums[i] + nums[left] + nums[right]
            if sum < 0:
                left += 1
            elif sum > 0:
                right -= 1
            else:
                # sum = 0인 경우이므로 정답 및 스킵처리
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    return result


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
            if sum < 0:
                left += 1
            elif sum > 0:
                right -= 1
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
    output = []
    p = 1
    for i in range(len(nums)):
        output.append(p)
        p *= nums[i]

    p = 1
    for j in range(len(nums) - 1, 0 - 1, -1):
        output[i] = output[i] * p
        p = p * nums[i]
    return output


if __name__ == "__main__":
    # nums = [3, 2, 3]
    # target = 6
    # print(twoSum(nums, target))

    height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    print(trap(height))

    # nums = [-1, 0, 1, 2, -1, -4]
    # print(myThreeSum(nums))

    # nums = [1, 4, 3, 2]
    # print(myArrayPairSum(nums))
    #
    # nums = [1, 2, 3, 4]
    # print(myProductExceptSelf(nums))
