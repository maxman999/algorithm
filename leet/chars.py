import collections
import re
from typing import *


def myIsPalindrome(s: str) -> bool:
    chars = collections.deque(re.sub("[^a-z0-9]", '', s.lower()))
    while len(chars) >= 2:
        if chars.popleft() != chars.pop():
            return False

    return True


def myReverseString(s: List[str]) -> None:
    left = 0
    right = len(s) - 1

    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1


def myReorderLogFiles(logs: List[str]) -> List[str]:
    strings = []
    numbers = []

    for log in logs:
        if log.split()[1].isdigit():
            numbers.append(log)
        else:
            strings.append(log)

    return sorted(strings, key=lambda i: (i.split()[1:], i.split()[0])) + numbers


def myMostCommonWord(paragraph: str, banned: List[str]) -> str:
    dict = collections.defaultdict(int)
    result = ""
    preprocessed = re.sub(r'[^\w]', ' ', paragraph.lower()).split()

    for strs in preprocessed:
        if strs not in banned:
            dict[strs] += 1

    max_value = max(dict.values())
    for key, value in dict.items():
        if value == max_value:
            result = key
            break

    return result


def myGroupAnagrams(strs: List[str]) -> List[List[str]]:
    dict = collections.defaultdict(list)

    for s in strs:
        c_str = ''.join(sorted(s))
        dict[c_str].append(s)

    return list(dict.values())


def myLongestPalindrome(s: str) -> str:
    def helper(left: int, right: int, s: str):
        candidate = s[left]
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]

    if len(s) < 2 or s == s[::-1]:
        return s

    result = ''
    for i in range(len(s) - 1):
        result = max(result, helper(i, i + 1, s), helper(i, i + 2, s), key=len)

    return result


if __name__ == '__main__':
    strings = "aacabdkacaa"
    print(myLongestPalindrome(strings))

    # strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    # print(myGroupAnagrams(strs))

    # paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
    # banned = ["hit"]
    # print(myMostCommonWord(paragraph, banned))

    # logs = ["dig1 8 1 5 1", "let1 art can", "dig2 3 6", "let2 own kit dig", "let3 art zero"]
    # logs2 = ["a1 9 2 3 1", "g1 act car", "zo4 4 7", "ab1 off key dog", "a8 act zoo", "a2 act car"]
    # print(myReorderLogFiles(logs2))

    # s = ["h", "e", "l", "l", "o","t"]
    # myReverseString(s)
    # print(s)

    # s = "A man, a plan, a canal: Panama"
    # print(myIsPalindrome(s))

    # logs = ["dig1 8 1 5 1", "let1 art can", "dig2 3 6", "let2 own kit dig", "let3 art zero"]
    # print(reorderLogFiles(logs))

    # strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    # print(groupAnagrams(strs))
