import json

# Paste the 30 slugs that were manually reviewed here:
REVIEWED_SLUGS = [
    "palindrome-number", "two-sum", "roman-to-integer",
    {
  "palindrome-number": [
    {
      "header": "def isPalindrome(x):",
      "chunks": [
        {
          "step_id": "Part 1",
          "prompt": "Determine if x is negative, and return False immediately since negative numbers cannot be palindromes.",
          "expected_type": "code",
          "reference": "if x < 0: return False"
        },
        {
          "step_id": "Part 2",
          "prompt": "Construct the reverse of the number x as a new value you can compare against.",
          "expected_type": "code",
          "reference": "reversed_num = 0\noriginal = abs(x)\nwhile original > 0:\n    reversed_num = reversed_num * 10 + original % 10\n    original //= 10"
        },
        {
          "step_id": "Part 3",
          "prompt": "Using the original and its reversed value, return whether x reads the same forwards and backwards.",
          "expected_type": "code",
          "reference": "return reversed_num == abs(x)"
        }
      ]
    },
    {
      "header": "def isPalindrome(x):",
      "chunks": [
        {
          "step_id": "Part 1",
          "prompt": "Determine if x is negative, and return False immediately if it is.",
          "expected_type": "code",
          "reference": "if x < 0: return False"
        },
        {
          "step_id": "Part 2",
          "prompt": "Construct the reverse of the number x as a new value you can compare against.",
          "expected_type": "code",
          "reference": "reversed_num = 0\noriginal = abs(x)\nwhile original > 0:\n    reversed_num = reversed_num * 10 + original % 10\n    original //= 10"
        },
        {
          "step_id": "Part 3",
          "prompt": "Using the absolute values of x and its reversed value, return whether they are equal.",
          "expected_type": "code",
          "reference": "return x == reversed_num"
        }
      ]
    },
    {
      "header": "def isPalindrome(x):",
      "chunks": [
        {
          "step_id": "Part 1",
          "prompt": "Determine if x is negative, and return False immediately if it is.",
          "expected_type": "code",
          "reference": "if x < 0: return False"
        },
        {
          "step_id": "Part 2",
          "prompt": "Construct the reverse of the number x as a new value you can compare against.",
          "expected_type": "code",
          "reference": "reversed_num = 0\noriginal = abs(x)\nwhile original > 0:\n    reversed_num = reversed_num * 10 + original % 10\n    original //= 10"
        },
        {
          "step_id": "Part 3",
          "prompt": "Using the original and its reversed value, return whether x reads the same forwards and backwards.",
          "expected_type": "code",
          "reference": "return reversed_num == abs(x)"
        }
      ]
    },
    {
      "header": "def isPalindrome(x):",
      "chunks": [
        {
          "step_id": "Part 1",
          "prompt": "Determine if x is negative, and return False immediately if it is.",
          "expected_type": "code",
          "reference": "if x < 0: return False"
        },
        {
          "step_id": "Part 2",
          "prompt": "Construct the reverse of the number x as a new value you can compare against.",
          "expected_type": "code",
          "reference": "reversed_num = 0\noriginal = abs(x)\nwhile original > 0:\n    reversed_num = reversed_num * 10 + original % 10\n    original //= 10"
        },
        {
          "step_id": "Part 3",
          "prompt": "Using the original and its reversed value, return whether x reads the same forwards and backwards.",
          "expected_type": "code",
          "reference": "return x == reversed_num"
        }
      ]
    }
  ],
  "roman-to-integer": [
    {
      "header": "def romanToInt(s):",
      "chunks": [
        {
          "step_id": "Part 1",
          "prompt": "Create a dictionary to map each Roman numeral symbol to its integer value.",
          "expected_type": "code",
          "reference": "symbol_values = {\n    'I': 1,\n    'V': 5,\n    'X': 10,\n    'L': 50,\n    'C': 100,\n    'D': 500,\n    'M': 1000\n}"
        },
        {
          "step_id": "Part 2",
          "prompt": "Iterate through the string s, comparing each symbol with the next one to determine if subtraction should be applied.",
          "expected_type": "code",
          "reference": "result = 0\ni = 0\nwhile i < len(s) - 1:\n    if symbol_values[s[i]] < symbol_values[s[i + 1]]:\n        result += symbol_values[s[i + 1]] - symbol_values[s[i]]\n        i += 2\n    else:\n        result += symbol_values[s[i]]\n        i += 1\nif i == len(s) - 1:\n    result += symbol_values[s[-1]]"
        },
        {
          "step_id": "Part 3",
          "prompt": "Return the final integer value calculated from the Roman numeral.",
          "expected_type": "code",
          "reference": "return result"
        }
      ]
    },
    {
      "header": "def romanToInt(s):",
      "chunks": [
        {
          "step_id": "Part 1",
          "prompt": "Create a dictionary to map each Roman numeral symbol to its integer value.",
          "expected_type": "code",
          "reference": "symbol_values = {\n    'I': 1,\n    'V': 5,\n    'X': 10,\n    'L': 50,\n    'C': 100,\n    'D': 500,\n    'M': 1000\n}"
        },
        {
          "step_id": "Part 2",
          "prompt": "Iterate through the string s, comparing each character with the next one to determine if a subtraction case applies.",
          "expected_type": "code",
          "reference": "result = 0\ni = 0\nwhile i < len(s) - 1:\n    if symbol_values[s[i]] < symbol_values[s[i + 1]]:\n        result += symbol_values[s[i + 1]] - symbol_values[s[i]]\n        i += 2\n    else:\n        result += symbol_values[s[i]]\n        i += 1\nif i == len(s) - 1:\n    result += symbol_values[s[-1]]"
        },
        {
          "step_id": "Part 3",
          "prompt": "Return the accumulated result which is the integer value of the Roman numeral.",
          "expected_type": "code",
          "reference": "return result"
        }
      ]
    },
    {
      "header": "def romanToInt(s):",
      "chunks": [
        {
          "step_id": "Part 1",
          "prompt": "Create a dictionary to map each Roman numeral symbol to its integer value.",
          "expected_type": "code",
          "reference": "symbol_values = {\n    'I': 1,\n    'V': 5,\n    'X': 10,\n    'L': 50,\n    'C': 100,\n    'D': 500,\n    'M': 1000\n}"
        },
        {
          "step_id": "Part 2",
          "prompt": "Initialize the total integer value to 0 and iterate through each character in the string.",
          "expected_type": "code",
          "reference": "total = 0\ni = 0\nwhile i < len(s):\n    if i + 1 < len(s) and s[i] == 'I' and (s[i+1] == 'V' or s[i+1] == 'X'):\n        total += symbol_values[s[i+1]] - 1\n        i += 2\n    elif i + 1 < len(s) and s[i] == 'X' and (s[i+1] == 'L' or s[i+1] == 'C'):\n        total += symbol_values[s[i+1]] - 10\n        i += 2\n    elif i + 1 < len(s) and s[i] == 'C' and (s[i+1] == 'D' or s[i+1] == 'M'):\n        total += symbol_values[s[i+1]] - 100\n        i += 2\n    else:\n        total += symbol_values[s[i]]\n        i += 1"
        },
        {
          "step_id": "Part 3",
          "prompt": "Return the accumulated integer value.",
          "expected_type": "code",
          "reference": "return total"
        }
      ]
    },
    {
      "header": "def romanToInt(s):",
      "chunks": [
        {
          "step_id": "Part 1",
          "prompt": "Create a dictionary to map each Roman numeral symbol to its integer value.",
          "expected_type": "code",
          "reference": "symbol_values = {\n    'I': 1,\n    'V': 5,\n    'X': 10,\n    'L': 50,\n    'C': 100,\n    'D': 500,\n    'M': 1000\n}"
        },
        {
          "step_id": "Part 2",
          "prompt": "Initialize the total integer value to 0 and iterate through the string, adding or subtracting values based on the Roman numeral rules.",
          "expected_type": "code",
          "reference": "total = 0\ni = 0\nwhile i < len(s):\n    if i + 1 < len(s) and symbol_values[s[i]] < symbol_values[s[i + 1]]:\n        total += symbol_values[s[i + 1]] - symbol_values[s[i]]\n        i += 2\n    else:\n        total += symbol_values[s[i]]\n        i += 1\nreturn total"
        }
      ]
    }
  ],
  "two-sum": [
    {
      "header": "def twoSum(nums, target):",
      "chunks": [
        {
          "step_id": "Part 1",
          "prompt": "Create a dictionary to store the indices of each number in nums as you iterate through it.",
          "expected_type": "code",
          "reference": "num_to_index = {}"
        },
        {
          "step_id": "Part 2",
          "prompt": "Iterate through nums, for each number check if the complement (target - current number) exists in num_to_index. If so, return the indices of the two numbers.",
          "expected_type": "code",
          "reference": "for i, num in enumerate(nums):\n    complement = target - num\n    if complement in num_to_index:\n        return [num_to_index[complement], i]\n    num_to_index[num] = i"
        }
      ]
    },
    {
      "header": "def twoSum(nums, target):",
      "chunks": [
        {
          "step_id": "Part 1",
          "prompt": "Create a dictionary to map each number in nums to its index.",
          "expected_type": "code",
          "reference": "num_to_index = {num: i for i, num in enumerate(nums)}"
        },
        {
          "step_id": "Part 2",
          "prompt": "Using the lookup, find and return the indices of the two numbers that sum to target.",
          "expected_type": "code",
          "reference": "for num in nums:\n    complement = target - num\n    if complement in num_to_index and num_to_index[complement] != num_to_index[num]:\n        return [num_to_index[num], num_to_index[complement]]"
        }
      ]
    },
    {
      "header": "def twoSum(nums, target):",
      "chunks": [
        {
          "step_id": "Part 1",
          "prompt": "Create a dictionary to store the indices of each number as you iterate through nums.",
          "expected_type": "code",
          "reference": "num_to_index = {}"
        },
        {
          "step_id": "Part 2",
          "prompt": "Iterate through the list nums, for each number check if the complement (target - current number) exists in the dictionary. If it does, return the indices stored in the dictionary and the current index.",
          "expected_type": "code",
          "reference": "for i, num in enumerate(nums):\n    complement = target - num\n    if complement in num_to_index:\n        return [num_to_index[complement], i]\n    num_to_index[num] = i"
        }
      ]
    }
  ]
}
]

pool = json.load(open("main/chunk_pool.json"))
total_components = 0
missing = []

for slug in REVIEWED_SLUGS:
    entries = pool.get(slug, [])
    if not entries:
        missing.append(slug)
        continue
    # Uses the first pooled decomposition for that slug —
    # change index if a different one was reviewed
    total_components += len(entries[0]["chunks"])

print(f"Total components across {len(REVIEWED_SLUGS) - len(missing)} slugs: {total_components}")
if missing:
    print(f"⚠️  Not found in pool, count manually: {missing}")