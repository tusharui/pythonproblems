#easy level

# Q1. Count vowels in a string
def count_vowels(s):
    vowels = "aeiouAEIOU"
    return sum(1 for ch in s if ch in vowels)
print(count_vowels("hello world"))   

# Q2. Reverse a string
def reverse_string(s):
    return s[::-1]
print(reverse_string("python"))   

# Q3. Check palindrome
def is_palindrome(s):
    return s == s[::-1]
print(is_palindrome("madam"))   
print(is_palindrome("hello"))   

# Q4. Count words in a string
def word_count(s):
    return len(s.split())
print(word_count("I love Python programming"))  # 4

# Q5. Convert to uppercase
def to_upper(s):
    return s.upper()
print(to_upper("python"))   

# Q6. Find first occurrence of a substring
def first_occurrence(s, sub):
    return s.find(sub)
print(first_occurrence("programming", "gram"))  

# Q7. Replace spaces with dashes
def replace_spaces(s):
    return s.replace(" ", "-")
print(replace_spaces("hello world python"))  

# Q8. Count digits in a string
def count_digits(s):
    return sum(1 for ch in s if ch.isdigit())
print(count_digits("abc123xyz45"))  

# Q9. Remove vowels from a string
def remove_vowels(s):
    vowels = "aeiouAEIOU"
    return "".join(ch for ch in s if ch not in vowels)
print(remove_vowels("beautiful"))  

# Q10. Find frequency of each character
def char_frequency(s):
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    return freq
print(char_frequency("banana")) 

#easytomed
# Q1. Remove all special characters from a string
def remove_special(s):
    return "".join(ch for ch in s if ch.isalnum() or ch.isspace())
print(remove_special("Hello@World!123"))  

# Q2. Count uppercase and lowercase letters
def count_case(s):
    upper = sum(1 for ch in s if ch.isupper())
    lower = sum(1 for ch in s if ch.islower())
    return upper, lower
print(count_case("Hello World"))  

# Q3. Find the longest word in a sentence
def longest_word(s):
    words = s.split()
    return max(words, key=len)
print(longest_word("Python is powerful and amazing")) 

# Q4. Swap case (uppercase ↔ lowercase)
def swap_case(s):
    return s.swapcase()
print(swap_case("HeLLo WoRLd"))  

# Q5. Remove duplicate characters from a string
def remove_duplicates(s):
    result = ""
    for ch in s:
        if ch not in result:
            result += ch
    return result
print(remove_duplicates("programming"))  

# Q6. Count number of words starting with a capital letter
def count_capital_words(s):
    return sum(1 for word in s.split() if word[0].isupper())
print(count_capital_words("I Love Python and Java")) 

# Q7. Check if a string contains only digits
def is_only_digits(s):
    return s.isdigit()
print(is_only_digits("12345"))  
print(is_only_digits("12a45"))  

# Q8. Find all unique characters in a string
def unique_chars(s):
    return "".join(sorted(set(s)))
print(unique_chars("banana"))  

# Q9. Reverse the order of words in a sentence
def reverse_words(s):
    return " ".join(s.split()[::-1])
print(reverse_words("Python is fun"))  

# Q10. Find the most frequent character in a string
def most_frequent(s):
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    return max(freq, key=freq.get)
print(most_frequent("mississippi"))  

#med
# Q1. Find the first non-repeating character in a string
def first_non_repeating(s):
    for ch in s:
        if s.count(ch) == 1:
            return ch
    return None
print(first_non_repeating("aabbcde"))  

# Q2. Remove all consecutive duplicate characters
def remove_consecutive_dupes(s):
    result = s[0]
    for ch in s[1:]:
        if ch != result[-1]:
            result += ch
    return result
print(remove_consecutive_dupes("aaabbccdaa"))  

# Q3. Find the longest palindrome substring
def longest_palindrome(s):
    n = len(s)
    longest = ""
    for i in range(n):
        for j in range(i+1, n+1):
            sub = s[i:j]
            if sub == sub[::-1] and len(sub) > len(longest):
                longest = sub
    return longest
print(longest_palindrome("babad"))  

# Q4. Check if two strings are rotations of each other
def is_rotation(s1, s2):
    return len(s1) == len(s2) and s2 in (s1 + s1)
print(is_rotation("abcd", "cdab"))  

# Q5. Count how many times each word appears in a sentence
def word_frequency(s):
    words = s.lower().split()
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return freq
print(word_frequency("Python is fun and Python is powerful"))

# Q6. Find all substrings of a string
def all_substrings(s):
    subs = []
    for i in range(len(s)):
        for j in range(i+1, len(s)+1):
            subs.append(s[i:j])
    return subs
print(all_substrings("abc"))

# Q7. Encode a string using run-length encoding
def run_length_encode(s):
    result = ""
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            count += 1
        else:
            result += s[i-1] + str(count)
            count = 1
    result += s[-1] + str(count)
    return result
print(run_length_encode("aaabbc"))  

# Q8. Find the longest word that appears more than once
def longest_repeated_word(s):
    words = s.split()
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    repeated = [w for w in words if freq[w] > 1]
    return max(repeated, key=len) if repeated else None
print(longest_repeated_word("this is a test this is python test"))  

# Q9. Check if a string can be rearranged to form a palindrome
def can_form_palindrome(s):
    from collections import Counter
    freq = Counter(s.replace(" ", "").lower())
    odd_count = sum(1 for v in freq.values() if v % 2 != 0)
    return odd_count <= 1
print(can_form_palindrome("civic"))     
print(can_form_palindrome("ivicc"))     
print(can_form_palindrome("hello"))     

# Q10. Find the longest common prefix among words
def longest_common_prefix(words):
    if not words:
        return ""
    prefix = words[0]
    for w in words[1:]:
        while not w.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix
print(longest_common_prefix(["flower","flow","flight"]))  

#medtohard
# Q1. Find the longest substring without repeating characters
def longest_unique_substring(s):
    seen = {}
    start = max_len = 0
    for i, ch in enumerate(s):
        if ch in seen and seen[ch] >= start:
            start = seen[ch] + 1
        seen[ch] = i
        max_len = max(max_len, i - start + 1)
    return max_len
print(longest_unique_substring("abcabcbb")) 

# Q2. Group words that are anagrams
def group_anagrams(words):
    from collections import defaultdict
    groups = defaultdict(list)
    for w in words:
        groups["".join(sorted(w))].append(w)
    return list(groups.values())
print(group_anagrams(["bat","tab","cat","act","tac"]))  

# Q3. Find the longest common substring of two strings
def longest_common_substring(s1, s2):
    n, m = len(s1), len(s2)
    dp = [[0]*(m+1) for _ in range(n+1)]
    longest, end_idx = 0, 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > longest:
                    longest = dp[i][j]
                    end_idx = i
    return s1[end_idx-longest:end_idx]
print(longest_common_substring("abcdef","zcdemf"))  

# Q4. Check if a string is a valid shuffle of two strings
def is_shuffle(s1, s2, s3):
    if len(s1)+len(s2) != len(s3):
        return False
    from collections import Counter
    return Counter(s1+s2) == Counter(s3)
print(is_shuffle("abc","def","dabecf"))  

# Q5. Find all permutations of a string
def permutations(s):
    if len(s) == 1:
        return [s]
    result = []
    for i, ch in enumerate(s):
        for perm in permutations(s[:i] + s[i+1:]):
            result.append(ch + perm)
    return result
print(permutations("abc"))  

# Q6. Minimum window substring containing all characters of another string
def min_window(s, t):
    from collections import Counter
    need, missing = Counter(t), len(t)
    i = start = end = 0
    for j, ch in enumerate(s, 1):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1
        if missing == 0:
            while i < j and need[s[i]] < 0:
                need[s[i]] += 1
                i += 1
            if end == 0 or j - i <= end - start:
                start, end = i, j
    return s[start:end]
print(min_window("ADOBECODEBANC", "ABC"))  

# Q7. Find the longest repeated substring
def longest_repeated_substring(s):
    n = len(s)
    suffixes = sorted(s[i:] for i in range(n))
    longest = ""
    for i in range(n-1):
        j = 0
        while j < min(len(suffixes[i]), len(suffixes[i+1])) and suffixes[i][j] == suffixes[i+1][j]:
            j += 1
        if j > len(longest):
            longest = suffixes[i][:j]
    return longest
print(longest_repeated_substring("banana"))  

# Q8. Convert a Roman numeral to integer
def roman_to_int(s):
    values = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    total = 0
    for i in range(len(s)):
        if i+1 < len(s) and values[s[i]] < values[s[i+1]]:
            total -= values[s[i]]
        else:
            total += values[s[i]]
    return total
print(roman_to_int("MCMXCIV"))  

# Q9. Find the edit distance (Levenshtein) between two strings
def edit_distance(s1, s2):
    n, m = len(s1), len(s2)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        for j in range(m+1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]
print(edit_distance("kitten","sitting"))  

# Q10. Convert a string into its zigzag pattern (given number of rows)
def zigzag_convert(s, numRows):
    if numRows == 1 or numRows >= len(s):
        return s
    rows = [""]*numRows
    index, step = 0, 1
    for ch in s:
        rows[index] += ch
        if index == 0:
            step = 1
        elif index == numRows-1:
            step = -1
        index += step
    return "".join(rows)
print(zigzag_convert("PAYPALISHIRING", 3))  


#hard 
# Q1. Find the longest substring without repeating characters
def longest_unique_substring(s):
    start, max_len, seen = 0, 0, {}
    for i, ch in enumerate(s):
        if ch in seen and seen[ch] >= start:
            start = seen[ch] + 1
        seen[ch] = i
        max_len = max(max_len, i - start + 1)
    return max_len
print(longest_unique_substring("abcabcbb"))  

# Q2. Find all permutations of a string
def string_permutations(s, step=0):
    if step == len(s):
        print("".join(s))
    for i in range(step, len(s)):
        s_copy = [c for c in s]
        s_copy[step], s_copy[i] = s_copy[i], s_copy[step]
        string_permutations(s_copy, step + 1)
print("Permutations of 'abc':")
string_permutations(list("abc"))

# Q3. Implement basic pattern matching (supports '.' and '*')
def is_match(s, p):
    if not p:
        return not s
    first_match = bool(s) and p[0] in {s[0], '.'}
    if len(p) >= 2 and p[1] == '*':
        return (is_match(s, p[2:]) or
                (first_match and is_match(s[1:], p)))
    else:
        return first_match and is_match(s[1:], p[1:])
print(is_match("aab", "c*a*b"))  

# Q4. Find the longest repeating substring
def longest_repeating_substring(s):
    n = len(s)
    longest = ""
    for i in range(n):
        for j in range(i+1, n):
            k = 0
            while j+k < n and s[i+k] == s[j+k]:
                k += 1
                if k > len(longest):
                    longest = s[i:i+k]
    return longest
print(longest_repeating_substring("banana")) 

# Q5. Group anagrams from a list of words
def group_anagrams(words):
    from collections import defaultdict
    groups = defaultdict(list)
    for w in words:
        groups["".join(sorted(w))].append(w)
    return list(groups.values())
print(group_anagrams(["eat","tea","tan","ate","nat","bat"]))

# Q6. Find the minimum window substring containing all characters of another string
def min_window(s, t):
    from collections import Counter
    need, missing = Counter(t), len(t)
    left = start = end = 0
    for right, ch in enumerate(s, 1):
        missing -= need[ch] > 0
        need[ch] -= 1
        if not missing:
            while left < right and need[s[left]] < 0:
                need[s[left]] += 1
                left += 1
            if not end or right - left <= end - start:
                start, end = left, right
    return s[start:end]
print(min_window("ADOBECODEBANC", "ABC"))  

# Q7. Check if two strings are isomorphic
def is_isomorphic(s, t):
    return len(set(zip(s, t))) == len(set(s)) == len(set(t))
print(is_isomorphic("egg", "add"))  
print(is_isomorphic("foo", "bar"))  

# Q8. Generate all valid parentheses combinations of length n
def generate_parentheses(n, open_count=0, close_count=0, s=""):
    if open_count == close_count == n:
        print(s)
        return
    if open_count < n:
        generate_parentheses(n, open_count+1, close_count, s+"(")
    if close_count < open_count:
        generate_parentheses(n, open_count, close_count+1, s+")")
print("Valid parentheses for n=3:")
generate_parentheses(3)

# Q9. Implement Rabin-Karp substring search
def rabin_karp(text, pattern, q=101):
    d = 256
    n, m = len(text), len(pattern)
    p = t = 0
    h = pow(d, m-1) % q
    for i in range(m):
        p = (d*p + ord(pattern[i])) % q
        t = (d*t + ord(text[i])) % q
    for i in range(n-m+1):
        if p == t and text[i:i+m] == pattern:
            return i
        if i < n-m:
            t = (d*(t-ord(text[i])*h) + ord(text[i+m])) % q
    return -1
print(rabin_karp("GEEKS FOR GEEKS", "FOR"))  

# Q10. Convert a string to its zigzag pattern (row-wise)
def zigzag_convert(s, numRows):
    if numRows == 1 or numRows >= len(s):
        return s
    rows = ["" for _ in range(numRows)]
    index, step = 0, 1
    for ch in s:
        rows[index] += ch
        if index == 0:
            step = 1
        elif index == numRows - 1:
            step = -1
        index += step
    return "".join(rows)
print(zigzag_convert("PAYPALISHIRING", 3))  


#famousrepeatedques 
# 1. Reverse a String
def reverse_str(s): return s[::-1]

# 2. Palindrome Check
def is_palindrome(s): return s == s[::-1]

# 3. First Non-Repeating Character
def first_unique(s):
    for ch in s:
        if s.count(ch) == 1: return ch
    return None

# 4. Anagram Check
def is_anagram(a,b): return sorted(a) == sorted(b)

# 5. Longest Common Prefix
def lcp(strs):
    if not strs: return ""
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
    return prefix

# 6. Count Substring Occurrences
def count_sub(s, sub): return s.count(sub)

# 7. Remove Duplicates
def remove_dupes(s):
    seen, res = set(), []
    for ch in s:
        if ch not in seen:
            seen.add(ch); res.append(ch)
    return "".join(res)

# 8. String Compression
def compress(s):
    res = ""; cnt = 1
    for i in range(1, len(s)+1):
        if i < len(s) and s[i] == s[i-1]: cnt += 1
        else: res += s[i-1] + str(cnt); cnt = 1
    return res

# 9. Longest Palindromic Substring
def longest_palindrome(s):
    res = ""
    for i in range(len(s)):
        for j in range(i,len(s)):
            sub = s[i:j+1]
            if sub == sub[::-1] and len(sub) > len(res): res = sub
    return res

# 10. Valid Parentheses
def valid_parens(s):
    stack, match = [], {')':'(', ']':'[', '}':'{'}
    for ch in s:
        if ch in match.values(): stack.append(ch)
        elif ch in match:
            if not stack or stack.pop() != match[ch]: return False
    return not stack

# 1. Longest Palindromic Substring
def longest_palindrome(s):
    res = ""
    for i in range(len(s)):
        for j in range(i, len(s)):
            temp = s[i:j+1]
            if temp == temp[::-1] and len(temp) > len(res):
                res = temp
    return res
print(longest_palindrome("babad"))  # "bab" or "aba"


# 2. Check if Two Strings are Rotations of Each Other
def is_rotation(s1, s2):
    return len(s1) == len(s2) and s2 in s1 + s1
print(is_rotation("abcd", "cdab"))  # True


# 3. Longest Common Prefix
def longest_common_prefix(strs):
    if not strs: return ""
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix: return ""
    return prefix
print(longest_common_prefix(["flower","flow","flight"]))  # "fl"


# 4. Group Anagrams
from collections import defaultdict
def group_anagrams(words):
    groups = defaultdict(list)
    for w in words:
        groups["".join(sorted(w))].append(w)
    return list(groups.values())
print(group_anagrams(["eat","tea","tan","ate","nat","bat"]))
# [["eat","tea","ate"], ["tan","nat"], ["bat"]]


# 5. Remove Duplicate Characters but Keep Order
def remove_duplicates(s):
    seen, result = set(), ""
    for ch in s:
        if ch not in seen:
            seen.add(ch)
            result += ch
    return result
print(remove_duplicates("programming"))  # "progamin"


# 6. First Non-Repeating Character
from collections import Counter
def first_non_repeating(s):
    count = Counter(s)
    for ch in s:
        if count[ch] == 1:
            return ch
    return None
print(first_non_repeating("swiss"))  # "w"


# 7. Count Palindromic Substrings
def count_palindromes(s):
    count = 0
    for i in range(len(s)):
        for j in range(i, len(s)):
            if s[i:j+1] == s[i:j+1][::-1]:
                count += 1
    return count
print(count_palindromes("aaa"))  # 6


# 8. Check if String Can Be Rearranged to a Palindrome
def can_form_palindrome(s):
    odd = sum(1 for c in set(s) if s.count(c) % 2 != 0)
    return odd <= 1
print(can_form_palindrome("civic"))  # True
print(can_form_palindrome("civil"))  # False



# 1. Longest Substring Without Repeating Characters
def lengthOfLongestSubstring(s):
    seen, l, res = {}, 0, 0
    for r, ch in enumerate(s):
        if ch in seen and seen[ch] >= l:
            l = seen[ch] + 1
        seen[ch] = r
        res = max(res, r - l + 1)
    return res
print(lengthOfLongestSubstring("abcabcbb"))  # 3


# 2. Group Anagrams
from collections import defaultdict
def groupAnagrams(words):
    groups = defaultdict(list)
    for w in words:
        groups["".join(sorted(w))].append(w)
    return list(groups.values())
print(groupAnagrams(["eat","tea","tan","ate","nat","bat"]))


# 3. Minimum Window Substring
from collections import Counter
def minWindow(s, t):
    need, window, have, need_count = Counter(t), {}, 0, len(Counter(t))
    res, res_len, l = [-1, -1], float("inf"), 0
    for r, ch in enumerate(s):
        window[ch] = window.get(ch, 0) + 1
        if ch in need and window[ch] == need[ch]:
            have += 1
        while have == need_count:
            if (r - l + 1) < res_len:
                res, res_len = [l, r], r - l + 1
            window[s[l]] -= 1
            if s[l] in need and window[s[l]] < need[s[l]]:
                have -= 1
            l += 1
    l, r = res
    return s[l:r+1] if res_len != float("inf") else ""
print(minWindow("ADOBECODEBANC", "ABC"))  # "BANC"


# 4. Edit Distance (Levenshtein)
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0: dp[i][j] = j
            elif j == 0: dp[i][j] = i
            elif s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
print(edit_distance("horse", "ros"))  # 3


# 5. Word Break
def word_break(s, wordDict):
    dp = [False]*(len(s)+1)
    dp[0] = True
    for i in range(1, len(s)+1):
        for w in wordDict:
            if dp[i-len(w)] and s[i-len(w):i] == w:
                dp[i] = True
    return dp[-1]
print(word_break("leetcode", ["leet","code"]))  # True


# 6. Ransom Note
from collections import Counter
def canConstruct(ransomNote, magazine):
    return not (Counter(ransomNote) - Counter(magazine))
print(canConstruct("aa", "aab"))  # True


# 7. Isomorphic Strings
def isIsomorphic(s, t):
    return len(set(zip(s,t))) == len(set(s)) == len(set(t))
print(isIsomorphic("egg","add"))  # True


# 8. Longest Repeating Subsequence
def longestRepeatingSubseq(s):
    n = len(s)
    dp = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, n+1):
            if s[i-1]==s[j-1] and i!=j:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][n]
print(longestRepeatingSubseq("aabb"))  # 2


# 9. ZigZag Conversion
def zigzag_convert(s, numRows):
    if numRows==1 or numRows>=len(s): return s
    rows, idx, step = [""]*numRows, 0, 1
    for ch in s:
        rows[idx]+=ch
        if idx==0: step=1
        elif idx==numRows-1: step=-1
        idx+=step
    return "".join(rows)
print(zigzag_convert("PAYPALISHIRING", 3))  # "PAHNAPLSIIGYIR"


# 10. Count and Say
def countAndSay(n):
    s="1"
    for _ in range(n-1):
        res, i="", 0
        while i<len(s):
            count=1
            while i+1<len(s) and s[i]==s[i+1]:
                i+=1; count+=1
            res+=str(count)+s[i]
            i+=1
        s=res
    return s
print(countAndSay(5))  # "111221"


# 11. Pangram Check
def isPangram(s):
    return set("abcdefghijklmnopqrstuvwxyz") <= set(s.lower())
print(isPangram("The quick brown fox jumps over the lazy dog"))  # True


# 12. Roman to Integer
def romanToInt(s):
    val={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    res, prev=0,0
    for ch in reversed(s):
        if val[ch]<prev: res-=val[ch]
        else: res+=val[ch]
        prev=val[ch]
    return res
print(romanToInt("MCMXCIV"))  # 1994


# 13. Valid Palindrome with One Deletion
def validPalindrome(s):
    def isPal(st): return st==st[::-1]
    l, r = 0, len(s)-1
    while l<r:
        if s[l]!=s[r]:
            return isPal(s[l+1:r+1]) or isPal(s[l:r])
        l+=1; r-=1
    return True
print(validPalindrome("abca"))  # True


# 14. Permutations
def permutations(s, chosen=""):
    if not s:
        print(chosen)
    for i in range(len(s)):
        permutations(s[:i]+s[i+1:], chosen+s[i])
permutations("abc")  # all permutations


# 15. Longest Repeated Substring
def longestRepeatedSubstring(s):
    n=len(s); lrs=""
    for i in range(n):
        for j in range(i+1,n):
            k=0
            while j+k<n and s[i+k]==s[j+k]:
                k+=1
                if k>len(lrs): lrs=s[i:i+k]
    return lrs
print(longestRepeatedSubstring("banana"))  # "ana"


# 16. Word Ladder (length only)
from collections import deque
def ladderLength(beginWord, endWord, wordList):
    wordSet=set(wordList)
    q=deque([(beginWord,1)])
    while q:
        word,steps=q.popleft()
        if word==endWord: return steps
        for i in range(len(word)):
            for c in "abcdefghijklmnopqrstuvwxyz":
                nxt=word[:i]+c+word[i+1:]
                if nxt in wordSet:
                    wordSet.remove(nxt)
                    q.append((nxt,steps+1))
    return 0
print(ladderLength("hit","cog",["hot","dot","dog","lot","log","cog"]))  # 5


# 17. Balanced String
def isBalanced(s):
    stack, pairs = [], {')':'(', ']':'[', '}':'{'}
    for ch in s:
        if ch in pairs.values(): stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1]!=pairs[ch]: return False
            stack.pop()
    return not stack
print(isBalanced("{[()]}"))  # True


# 18. Remove Invalid Parentheses
def removeInvalidParentheses(s):
    from collections import deque
    def isValid(st):
        cnt=0
        for c in st:
            if c=='(': cnt+=1
            elif c==')':
                cnt-=1
                if cnt<0: return False
        return cnt==0
    q, visited, res = deque([s]), {s}, []
    found=False
    while q:
        st=q.popleft()
        if isValid(st):
            res.append(st); found=True
        if found: continue
        for i in range(len(st)):
            if st[i] not in "()": continue
            new=st[:i]+st[i+1:]
            if new not in visited:
                visited.add(new)
                q.append(new)
    return res
print(removeInvalidParentheses("()())()"))  # ["(())()","()()()"]
    