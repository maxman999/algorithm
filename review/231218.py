from typing import List


# 17. Letter Combinations of a Phone Number
def letterCombinations(digits: str, dic: {}) -> List[str]:
    result = []

    def backTrac(idx, path):
        if len(path) == len(digits):
            result.append(path)
            return

        for char in dic[digits[idx]]:
            backTrac(idx + 1, path + char)

    backTrac(0, "")

    return result


if __name__ == "__main__":
    dic = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
    print(letterCombinations("23", dic))
