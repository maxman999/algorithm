import collections
import numbers
import heapq
from typing import *
from linkedList import ListNode


class MyHashMap:
    def __init__(self, size: numbers):
        self.bucket = [[] for _ in range(size)]
        self.bucket_size = size

    def put(self, key: numbers, value: numbers):
        hash_code = self.hashing(key, self.bucket_size)
        for i in range(len(self.bucket[hash_code])):
            if self.bucket[hash_code][i][0] == key:
                self.bucket[hash_code][i] = (key, value)
                return
        self.bucket[hash_code].append((key, value))

    def get(self, key):
        hash_code = self.hashing(key, self.bucket_size)
        for i in range(len(self.bucket[hash_code])):
            if self.bucket[hash_code][i][0] == key:
                return self.bucket[hash_code][i][1]
        return None

    def remove(self, key):
        hash_code = self.hashing(key, self.bucket_size)
        for i in range(len(self.bucket[hash_code])):
            if self.bucket[hash_code][i][0] == key:
                del self.bucket[hash_code][i]
                return

    def hashing(self, val, size):
        return (val * 31) % size


class MyHashMap2:
    # 초기화
    def __init__(self):
        self.size = 1000
        self.table = collections.defaultdict(ListNode)

    # 삽입
    def put(self, key: int, value: int) -> None:
        index = key % self.size
        # 인덱스에 노드가 없다면 삽입 후 종료
        if self.table[index].val is None:
            self.table[index] = ListNode(key, value)
            return
            # 인덱스에 노드가 있는 경우 연결리스트 처리
        p = self.table[index]
        while p:
            if p.key == key:
                p.val = value
                return
            if p.next is None:
                break
            p = p.next
        p.next = ListNode(key, value)

    # 조회
    def get(self, key: int) -> int:
        index = key % self.size
        if self.table[index].val is None:
            return -1

        # 노드가 존재할 때 일치하는 키 탐색
        p = self.table[index]
        while p:
            if p.key == key:
                return p.val
            p = p.next
        return -1

    # 삭제
    def remove(self, key: int) -> None:
        index = key % self.size
        if self.table[index].val is None:
            return

        # 인덱스의 첫번째 노드일 때 삭제처리
        p = self.table[index]
        if p.key == key:
            self.table[index] = ListNode() if p.next is None else p.next
            return

        # 연결 리스트 노드 삭제
        prev = p
        while p:
            if p.key == key:
                prev.next = p.next
                return
            prev, p = p, p.next


def numJewelsInStones(jewels: str, stones: str) -> int:
    cnt = 0
    jewel_dict = {}
    for j in jewels:
        jewel_dict[j] = j

    for s in stones:
        if jewel_dict.get(s):
            cnt += 1
    return cnt


def lengthOfLongestSubstring(s: str) -> int:
    used = {}
    max_length = start = 0
    for index, char in enumerate(s):
        # 이미 등장했던 문자라면 'start' 위치 갱신
        if char in used and start <= used[char]:
            start = used[char] + 1
        else:  # 최대 부분 문자열 갱신
            max_length = max(max_length, index - start + 1)

        # 현재 문자의 위치 삽입
        used[char] = index

    return max_length


def myLengthOfLongestSubstring(s: str) -> int:
    used = {}
    start = 0
    max_length = 0

    for index, char in enumerate(s):
        if char in used and start <= used[char]:
            start = used[char] + 1
        else:
            max_length = max(max_length, index - start + 1)

        used[char] = index

    return max_length


def topKFrequent(nums: List[int], k: int) -> List[int]:
    counter = {}
    result_heap = []
    result = []

    for num in nums:
        if counter.get(num):
            counter[num] += 1
        else:
            counter[num] = 1

    for i in counter:
        heapq.heappush(result_heap, ((counter[i] * -1), i))

    for _ in range(k):
        result.append(heapq.heappop(result_heap)[1])

    return result


if __name__ == '__main__':
    nums = [4, 1, -1, 2, -1, 2, 3]
    k = 2

    print(topKFrequent(nums, k))
