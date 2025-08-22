# Easy

# Given a list of numbers, return a new list with each number doubled.

# Find the largest number in a list.

# Count how many times a specific element appears in a list.

# Reverse a given list.

# Merge two lists into one.

# Find the sum of all elements in a list.

# Remove duplicates from a list.

# Check if a list is sorted in ascending order.

# Given a string, return it in reverse.

# Count the number of vowels in a string.

# Medium

# Given a list of words, return a list of words that are palindromes.

# Flatten a list of lists into a single list.

# Find the second largest number in a list.

# Given a string, count the frequency of each character and return a dictionary.

# Write a function that returns all unique pairs in a list whose sum is equal to a target number.

# Check if two strings are anagrams of each other.

# Given a list of numbers, move all zeros to the end without changing the order of other elements.

# Given a list of integers, return the sublist with the maximum sum (contiguous).

# Implement a function to rotate a list by k positions to the right.

# Given a dictionary, return a new dictionary with keys and values swapped. 



def double_num(l):
    new =[]
    for i in l :
        new.append(i*2)
    return new 
q = [1,2,3,4,5]
print(double_num(q))


def largest_num(l):
    return max(l)
w= [1,2,3,4,5]
print(largest_num(w))

def num_app(l):
    freq={}
    for item in l :
        if item in freq:
            freq[item] +=1
        else :
            freq[item] =1
    return max(freq)
e =[1,1,11,1,1,1,1,1,1,2,2,2,2,5,5,5,8,85,26,65,4,962,58,5,58,2,2,6,5]
print(num_app(e))

def reverse_list(l):
    return l[::-1]

r=[1,2,3,4,5,6]
print(reverse_list(r))

def merge_list(l1,l2):
    return l1+l2
t= [1,2,3,4]
y=[5,6,7,8,9]
print(merge_list(t,y))


def sum_ele(l):
    return sum(l)

u=[1,2,3,4,5]
print(sum_ele(u))


def remove_dupes(l):
    freq={}
    result =[]
    for i in l :
       if i not in freq:
           freq[i]= 1
           result.append(i)
    return result
p = [1,2,32,5,5,25,2,5,5]
print(remove_dupes(p))


def ascen_ord(l):
    return sorted(l)
a = [5,4,3,21,5,5,6,558,5,4,52,5,2]
print(ascen_ord(a))

def rever_se(l):
    return l[::-1]
s = "tuhsar"
print(rever_se(s))

def vowels_count(s):
    count = 0
    vowels ="aeiou"
    for i in s :
        if i in vowels :
            count +=1
        
    return count

d ="tushar"
print(vowels_count(d))




def paild_rome(l):
   paildrome =[]
   for i in l :
       if i == i[::-1]:
           paildrome.append(i) 
   return paildrome

f = ["oto", "hello", "madam", "world", "racecar"]
print(paild_rome(f))



def flatten_list(lst):
    lists=[]
    for i in lst :
        for p  in i:
            lists.append(p)
    return lists
g = [[1, 2], [3, 4], [5, 6]]
print(flatten_list(g))




def second_larget(l):
    l = sorted(l)
    return l[-2]
h =[1,2,3,4,5]
print(second_larget(g))


def freq_words(l):
    freq ={}
    for i in l :
        if i in freq:
            freq[i] +=1
        else:
            freq[i] =1
    return freq

j = "tushar"
print(freq_words(j))


def unique_sum(lst,target):
    pairs =[]
    seen = set()
    for i in range (len(lst)):
        for j in range(i+1, len(lst)):
            if lst[i]+lst[j] ==target :
                pair = tuple(sorted([lst[i], lst[j]]))
                if pair not in seen :
                    pairs.append(pair)
                    seen.add(pair)
    return pairs


num = [1, 2, 3, 4, 2, 5]
target_sum = 5

print(unique_sum(num,target_sum))



def anagram(lst):
    anagrams=[]
    for i in lst :
        for j in lst :
            if i != j and sorted(i.strip()) == sorted(i.strip()):
                anagrams.append(i)
    return anagrams

k = ["listen", "silent"]
print(anagram(k))

def zero_sum(nums):
    result =[]
    zeros =0

    for n in nums :
         if n == 0:
             zeros +=1
         else :
             result.append(n)
    result.extend([0]*zeros )
    return result 
z = [1,2,0,3,41,6,0,52,0,52,0,5,2,5,5,0,2,2,0]
print(zero_sum(z))



def max_sums(lst):
    current_sum = lst[0]
    max_sum = lst[0]
    for i in lst[1:]:
        current_sum = max(i, current_sum + i )
        max_sum =max(max_sum,current_sum)
    return max_sum

v = [1,3,3,56,3,56,5,65,3,3,5,5,6,51,545,8]
print(max_sums(v))


def rotate_lst(lst,k):
    k = k%len(lst)
    rotated =  lst[-k:]+lst [:-k]
    return rotated 
k = [1,2,3,4,5,6]
print(rotate_lst(k,2))