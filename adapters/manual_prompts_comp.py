humaneval_manual_prompt_dict = {
        50:['def decode_shift(s: str):',
            '''def encode_shift(s: str):
    "returns encoded string by shifting every character by 5 in the alphabet."
    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])
takes as input string encoded with encode_shift function. Returns decoded string.''',
            '',
        ],

        67:['def fruit_distribution(s,n):',
            '''    In this task, you will be given a string that represents a number of apples and oranges 
    that are distributed in a basket of fruit this basket contains 
    apples, oranges, and mango fruits. Given the string that represents the total number of 
    the oranges and apples and an integer that represent the total number of the fruits 
    in the basket return the number of the mango fruits in the basket.''',
        '''    for examble:
    fruit_distribution(5 apples and 6 oranges, 19) ->19 - 5 - 6 = 8
    fruit_distribution(0 apples and 1 oranges,3) -> 3 - 0 - 1 = 2
    fruit_distribution(2 apples and 3 oranges, 100) -> 100 - 2 - 3 = 95
    fruit_distribution(100 apples and 1 oranges,120) -> 120 - 100 - 1 = 19'''
        ],
        90:[
            'def next_smallest(lst):',
            """    You are given a list of integers.
    Write a function next_smallest() that returns the 2nd smallest element of the list.
    Return None if there is no such element.""",
    """    next_smallest([1, 2, 3, 4, 5]) == 2
    next_smallest([5, 1, 4, 3, 2]) == 2
    next_smallest([]) == None
    next_smallest([1, 1]) == None"""
        ],

        127:[
            'def intersection(interval1, interval2):',
            """    You are given two intervals,
    where each interval is a pair of integers. For example, interval = (start, end) = (1, 2).
    The given intervals are closed which means that the interval (start, end)
    includes both start and end.
    For each given interval, it is assumed that its start is less or equal its end.
    Your task is to determine whether the length of intersection of these two 
    intervals is a prime number.
    Example, the intersection of the intervals (1, 3), (2, 4) is (2, 3)
    which its length is 1, which not a prime number.
    If the length of the intersection is a prime number, return "YES",
    otherwise, return "NO".
    If the two intervals don't intersect, return "NO".""",
    """    [input/output] samples:
    intersection((1, 2), (2, 3)) ==> "NO"
    intersection((-1, 1), (0, 4)) ==> "NO"
    intersection((-3, -1), (-5, 5)) ==> "YES" """
        ],

        132:[
            'def is_nested(string):',
            """    Create a function that takes a string as input which contains only square brackets.
    The function should return True if and only if there is a valid subsequence of brackets 
    where at least one bracket in the subsequence is nested.""",
    """    is_nested('[[]]') ➞ True
    is_nested('[]]]]]]][[[[[]') ➞ False
    is_nested('[][]') ➞ False
    is_nested('[]') ➞ False
    is_nested('[[][]]') ➞ True
    is_nested('[[]][[') ➞ True"""
        ],

        137:[
            'def compare_one(a, b):',
            """    Create a function that takes integers, floats, or strings representing
    real numbers, and returns the larger variable in its given variable type.
    Return None if the values are equal.
    Note: If a real number is represented as a string, the floating point might be . or ,""",
    """    compare_one(1, 2.5) ➞ 2.5
    compare_one(1, 2,3) ➞ 2,3
    compare_one(5,1, 6) ➞ 6
    compare_one(1, 1) ➞ None"""
        ],

        140:[
            'def fix_spaces(text):',
            """Given a string text, replace all spaces in it with underscores, 
    and if a string has more than 2 consecutive spaces, 
    then replace all consecutive spaces with - """,
    """    fix_spaces("Example") == "Example"
    fix_spaces("Example 1") == "Example_1"
    fix_spaces(" Example 2") == "_Example_2"
    fix_spaces(" Example   3") == "_Example-3" """
        ],
        146:[
            "def simplify(x, n):",
            """Your task is to implement a function that will simplify the expression
    x * n. The function returns True if x * n evaluates to a whole number and False
    otherwise. Both x and n, are string representation of a fraction, and have the following format,
    <numerator>/<denominator> where both numerator and denominator are positive whole numbers.

    You can assume that x, and n are valid fractions, and do not have zero as denominator.""",
    """    simplify(1/5, 5/1) = True
    simplify(1/6, 2/1) = False
    simplify(7/10, 10/2) = False"""
        ],
        151:[
            "def double_the_difference(lst):",
            """    Given a list of numbers, return the sum of squares of the numbers
    in the list that are odd. Ignore numbers that are negative or not integers.""",
    """    double_the_difference([1, 3, 2, 0]) == 1 + 9 + 0 + 0 = 10
    double_the_difference([-1, -2, 0]) == 0
    double_the_difference([9, -2]) == 81
    double_the_difference([0]) == 0  
   
    If the input list is empty, return 0."""
        ],

        154:[
            "def cycpattern_check(a , b):",
            """You are given 2 words. You need to return True if the second word or any of its rotations is a substring in the first word""",
            """    cycpattern_check(abcd,abd) => False
    cycpattern_check(hello,ell) => True
    cycpattern_check(whassup,psus) => False
    cycpattern_check(abab,baa) => True
    cycpattern_check(efef,eeff) => False
    cycpattern_check(himenss,simen) => True"""
        ],
        158:[
            "def find_max(words):",
            """Write a function that accepts a list of strings.
    The list contains different words. Return the word with maximum number
    of unique characters. If multiple strings have maximum number of unique
    characters, return the one which comes first in lexicographical order.""",
    """
    find_max([name, of, string]) == string
    find_max([name, enam, game]) == enam
    find_max([aaaaaaa, bb ,cc]) == aaaaaaa"""
        ]
    }