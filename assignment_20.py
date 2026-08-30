"""Practice Set - 20 Problems

Upload this file on the Instructor page (teacher.html -> "Prepare an
assignment"). Each problem below is one function: its docstring is the
statement students see, its body is the private ground-truth solution.
"""

# --- problem: is-leap-year ---
def is_leap_year(year):
    """Given an integer year, return True if it is a leap year and False
    otherwise.

    A year is a leap year when it is divisible by 4, except that years
    divisible by 100 are not leap years, unless they are also divisible by 400.

    Example: is_leap_year(2024) -> True, is_leap_year(1900) -> False,
    is_leap_year(2000) -> True
    """
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    return year % 4 == 0


# --- problem: fizzbuzz-value ---
def fizzbuzz_value(n):
    """Given a positive integer n, return "FizzBuzz" if n is divisible by both
    3 and 5, "Fizz" if it is divisible only by 3, "Buzz" if it is divisible
    only by 5, and the string form of n otherwise.

    Example: fizzbuzz_value(15) -> "FizzBuzz", fizzbuzz_value(9) -> "Fizz",
    fizzbuzz_value(7) -> "7"
    """
    if n % 3 == 0 and n % 5 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)


# --- problem: is-prime ---
def is_prime(n):
    """Given an integer n, return True if n is a prime number and False
    otherwise. A prime number is greater than 1 and divisible only by 1 and
    itself.

    Example: is_prime(13) -> True, is_prime(1) -> False, is_prime(15) -> False
    """
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    divisor = 3
    while divisor * divisor <= n:
        if n % divisor == 0:
            return False
        divisor += 2
    return True


# --- problem: triangle-type ---
def triangle_type(a, b, c):
    """Given three positive side lengths a, b and c, return "equilateral" if
    all three sides are equal, "isosceles" if exactly two sides are equal,
    "scalene" if all sides differ, and "invalid" if the sides cannot form a
    triangle (the sum of any two sides must be strictly greater than the
    third).

    Example: triangle_type(3, 3, 5) -> "isosceles",
    triangle_type(1, 2, 3) -> "invalid"
    """
    if a + b <= c or a + c <= b or b + c <= a:
        return "invalid"
    if a == b and b == c:
        return "equilateral"
    if a == b or b == c or a == c:
        return "isosceles"
    return "scalene"


# --- problem: digit-sum ---
def digit_sum(n):
    """Given an integer n, return the sum of its decimal digits. A negative
    sign is ignored.

    Example: digit_sum(1234) -> 10, digit_sum(-905) -> 14
    """
    n = abs(n)
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total


# --- problem: reverse-integer ---
def reverse_integer(n):
    """Given a signed integer n, return n with its decimal digits reversed,
    keeping the sign. If the reversed value falls outside the 32-bit signed
    range [-2147483648, 2147483647], return 0 instead.

    Example: reverse_integer(-123) -> -321, reverse_integer(120) -> 21,
    reverse_integer(1534236469) -> 0
    """
    sign = -1 if n < 0 else 1
    n = abs(n)
    result = 0
    while n > 0:
        result = result * 10 + n % 10
        n //= 10
    result *= sign
    if result < -2147483648 or result > 2147483647:
        return 0
    return result


# --- problem: count-vowels ---
def count_vowels(text):
    """Given a string, return the number of vowels (a, e, i, o, u) it
    contains, treating uppercase and lowercase letters as the same.

    Example: count_vowels("Hello World") -> 3, count_vowels("XYZ") -> 0
    """
    count = 0
    for ch in text.lower():
        if ch in "aeiou":
            count += 1
    return count


# --- problem: is-palindrome-string ---
def is_palindrome_string(text):
    """Given a string, return True if it reads the same forwards and backwards
    once case is ignored and every character that is not a letter or digit is
    removed.

    Example: is_palindrome_string("A man, a plan, a canal: Panama") -> True,
    is_palindrome_string("race a car") -> False
    """
    cleaned = ""
    for ch in text.lower():
        if ch.isalnum():
            cleaned += ch
    left = 0
    right = len(cleaned) - 1
    while left < right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1
    return True


# --- problem: second-largest ---
def second_largest(nums):
    """Given a list of at least two distinct integers, return the second
    largest value in the list.

    Example: second_largest([4, 1, 9, 7]) -> 7, second_largest([2, 1]) -> 1
    """
    best = nums[0]
    second = nums[1]
    if second > best:
        best, second = second, best
    for n in nums[2:]:
        if n > best:
            best, second = n, best
        elif n > second:
            second = n
    return second


# --- problem: count-words ---
def count_words(text):
    """Given a string, return the number of words in it. A word is a maximal
    run of characters separated by spaces, tabs or newlines. Leading and
    trailing whitespace must not be counted as words.

    Example: count_words("  hello   world  ") -> 2, count_words("") -> 0
    """
    count = 0
    in_word = False
    for ch in text:
        if ch == " " or ch == "\t" or ch == "\n":
            in_word = False
        elif not in_word:
            in_word = True
            count += 1
    return count


# --- problem: is-armstrong ---
def is_armstrong(n):
    """Given a non-negative integer n, return True if it equals the sum of
    each of its digits raised to the power of the number of digits, and False
    otherwise.

    Example: 153 = 1**3 + 5**3 + 3**3, so is_armstrong(153) -> True;
    is_armstrong(154) -> False
    """
    digits = str(n)
    power = len(digits)
    total = 0
    for ch in digits:
        total += int(ch) ** power
    return total == n


# --- problem: collatz-steps ---
def collatz_steps(n):
    """Given a positive integer n, return how many steps it takes to reach 1
    by repeatedly halving n when it is even and replacing it with 3 * n + 1
    when it is odd.

    Example: collatz_steps(6) -> 8, collatz_steps(1) -> 0
    """
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        steps += 1
    return steps


# --- problem: sum-even-numbers ---
def sum_even_numbers(nums):
    """Given a list of integers, return the sum of the values that are even.
    Return 0 if the list contains no even values.

    Example: sum_even_numbers([1, 2, 3, 4, 5]) -> 6,
    sum_even_numbers([1, 3, 5]) -> 0
    """
    total = 0
    for n in nums:
        if n % 2 == 0:
            total += n
    return total


# --- problem: celsius-to-fahrenheit ---
def celsius_to_fahrenheit(celsius):
    """Given a temperature in degrees Celsius, return the equivalent
    temperature in degrees Fahrenheit, computed as celsius * 9 / 5 + 32.

    Example: celsius_to_fahrenheit(100) -> 212.0,
    celsius_to_fahrenheit(0) -> 32.0
    """
    fahrenheit = celsius * 9 / 5 + 32
    return fahrenheit


# --- problem: score-to-grade ---
def score_to_grade(score):
    """Given an integer exam score from 0 to 100, return its letter grade:
    "A" for 90 and above, "B" for 80 through 89, "C" for 70 through 79,
    "D" for 60 through 69, and "F" for anything below 60.

    Example: score_to_grade(84) -> "B", score_to_grade(59) -> "F"
    """
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


# --- problem: hamming-distance ---
def hamming_distance(x, y):
    """Given two non-negative integers x and y, return the number of bit
    positions at which their binary representations differ.

    Example: hamming_distance(1, 4) -> 2, hamming_distance(3, 3) -> 0
    """
    count = 0
    while x > 0 or y > 0:
        if x % 2 != y % 2:
            count += 1
        x //= 2
        y //= 2
    return count


# --- problem: is-power-of-two ---
def is_power_of_two(n):
    """Given an integer n, return True if n is a positive power of two
    (1, 2, 4, 8, 16, ...) and False otherwise.

    Example: is_power_of_two(16) -> True, is_power_of_two(6) -> False,
    is_power_of_two(0) -> False
    """
    if n < 1:
        return False
    while n % 2 == 0:
        n //= 2
    return n == 1


# --- problem: run-length-encode ---
def run_length_encode(text):
    """Given a non-empty string, return its run-length encoding: every run of
    the same character is replaced by that character followed by the length of
    the run.

    Example: run_length_encode("aaabbc") -> "a3b2c1",
    run_length_encode("x") -> "x1"
    """
    result = ""
    count = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            count += 1
        else:
            result += text[i - 1] + str(count)
            count = 1
    result += text[-1] + str(count)
    return result


# --- problem: caesar-cipher ---
def caesar_cipher(text, shift):
    """Given a string of lowercase letters and spaces and an integer shift,
    return the string with every letter advanced by shift positions through
    the alphabet, wrapping from "z" back to "a". Spaces are left unchanged.

    Example: caesar_cipher("abc xyz", 2) -> "cde zab"
    """
    result = ""
    for ch in text:
        if ch == " ":
            result += ch
        else:
            offset = (ord(ch) - ord("a") + shift) % 26
            result += chr(offset + ord("a"))
    return result


# --- problem: median-of-three ---
def median_of_three(a, b, c):
    """Given three numbers a, b and c, return their median - the value that is
    neither the smallest nor the largest. When some values are equal, the
    median is still the middle value once the three are sorted.

    Example: median_of_three(7, 2, 5) -> 5, median_of_three(4, 4, 1) -> 4
    """
    if (a <= b and b <= c) or (c <= b and b <= a):
        return b
    if (b <= a and a <= c) or (c <= a and a <= b):
        return a
    return c
