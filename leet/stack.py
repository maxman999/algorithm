import collections
from typing import *
import heapq
from modules.ListNode import ListNode, NodeGenerator


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


def myValidateParenthesis(s: str) -> bool:
    parenthesis_map = {
        "{": "}",
        "(": ")",
        "[": "]",
    }

    stack = []
    for char in s:
        if char in parenthesis_map:
            stack.append(char)
        else:
            if not stack or char != parenthesis_map.get(stack.pop()):
                return False

    return len(stack) == 0


def removeDuplicateLetters(s: str) -> str:
    counter, seen, stack = collections.Counter(s), set(), []

    temp = set(s)

    for char in s:
        counter[char] -= 1
        if char in seen:
            continue
        # 뒤에 붙일 문자가 남아 있다면 스택에서 제거
        while stack and char < stack[-1] and counter[stack[-1]] > 0:
            seen.remove(stack.pop())
        stack.append(char)
        seen.add(char)
    return ''.join(stack)


def dailyTemperatures(temperatures: List[int]) -> List[int]:
    result = [0] * len(temperatures)
    stack = []

    for i, cur in enumerate(temperatures):
        while stack and cur > temperatures[stack[-1]]:
            last = stack.pop()
            result[last] = i - last
        stack.append(i)

    return result


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
    result = head = ListNode(None)
    heap = []
    for i in range(len(lists)):
        heapq.heappush(heap, (lists[i].val, i, lists[i]))

    while heap:
        curr = heapq.heappop(heap)
        head.next = curr[2]
        head = head.next
        if curr[2].next:
            heapq.heappush(heap, (curr[2].next.val, curr[1], curr[2].next))
    return result.next


if __name__ == '__main__':
    list1 = NodeGenerator(1, 4, 5).head
    list2 = NodeGenerator(1, 3, 4).head
    list3 = NodeGenerator(2, 6).head
    target_lists = [list1, list2, list3]
    merged_list = myMergeKLists(target_lists)

    while merged_list:
        print(merged_list.val)
        merged_list = merged_list.next

    # s = "]"
    # print(myValidateParenthesis(s))
    #
    # s = "cbacdcbc"
    # print(removeDuplicateLetters(s))
    # temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
    # print(dailyTemperatures(temperatures))
