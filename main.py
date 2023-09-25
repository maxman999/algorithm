import collections
from typing import *
import heapq
import functools
import itertools
import re
import sys
import math
import bisect

if __name__ == '__main__':
    # Q1
    def reorderLogFiles(logs: List[str]) -> List[str]:
        letters, digits = [], []
        for log in logs:
            if log.split()[1].isdigit():
                digits.append(log)
            else:
                letters.append(log)
        letters.sort(key=lambda x: (x.split()[1:], x.split()[0]))
        return letters + digits


    target_logs = ["dig1 8 1 5 1", "let1 art can", "dig2 3 6", "let2 own kit dig", "let3 art zero"]
    print(reorderLogFiles(target_logs))


    # Q2
    def mostCommonWord(paragraph: str, banned: List[str]) -> str:
        words = [word for word in re.sub(r'\W', ' ', paragraph).lower().split()
                 if word not in banned]
        counts = collections.Counter(words)
        return counts.most_common(1)[0][0]


    paragraph = "Bob hit a ball, the hit BALL flew far after it was hit"
    banned = ["hit"]
    print(mostCommonWord(paragraph, banned))


    # Q3
    def longestPalindrome(s: str) -> str:
        # 팰린드롬 판별 및 투 포인터 확장
        def expand(left: int, right: int) -> str:
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left + 1: right]

        # 해당사항 없을 때 빠르게 리턴
        if len(s) < 2 or s == s[::-1]:
            return s
        result = ''
        # 슬라이딩 윈도우 우측으로 이동
        for i in range(len(s) - 1):
            result = max(result,
                         expand(i, i + 1),
                         expand(i, i + 2),
                         key=len)
        return result


    print(longestPalindrome("asdweewp23ejkasddsa"))

    
