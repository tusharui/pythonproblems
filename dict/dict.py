# program to count the number of keys in a dictionary.
def count_numofkey(d):
    count =0
    for i in d :
        count +=1
    return count 
my_dict = {"a": 1, "b": 2, "c": 3}
print(count_numofkey(my_dict))  

# Given a dictionary, check if a key exists.
def key_in_dict (d,key):
    if key in d :
        return True 
    else:
        return False
my_dict = {"name": "Alice", "age": 25}
print(key_in_dict(my_dict, "age"))   
print(key_in_dict(my_dict, "gender")) 


# program to get all keys of a dictionary as a list.
def all_keys(d):
    all =[]
    for key in d :
        all.append(key)
    return all

my_dict = {"a": 10, "b": 20, "c": 30}
print(all_keys(my_dict))
    

# function to sum all values in a dictionary.
def sum_val(d):
    total =0
    for i in d.values():
        total += i
    return total 
d = {"x": 10, "y": 20, "z": 30}
print(sum_val(d))  


# program to find the key with the maximum value.
def max_val(d):
    return max(d, key = d.get)
d = {"a": 10, "b": 50, "c": 30}
print(max_val(d))

# program to merge two dictionaries.
def merge_dict(d1,d2):
    merged = d1.copy()
    merged.update(d2)
    return merged
d1 = {"a": 1, "b": 2}
d2 = {"c": 3, "d": 4}
print(merge_dict(d1,d2))

# function to invert a dictionary (swap keys and values).
def swap_dict(d):
    swapped = {}
    for key, value in d.items():
        swapped[value] = key
    return swapped


d = {"a": 1, "b": 2, "c": 3}
print(swap_dict(d)) 


#find above 50 score 
def high_scorers(students):
    result =[]
    for name , marks in students.items():
        if marks >50:
            result.append(name)
    return result 

students = {"Alice": 45, "Bob": 78, "Charlie": 90}
print(high_scorers(students))  


#remove key
def remove_key(d, key):
    d.pop(key, None)
    return d


d = {"a": 1, "b": 2, "c": 3}
print(remove_key(d, "b"))  


#count freq of word
def count_freq(d):
    freq={}
    for i in d :
        if i in freq:
            freq[i] +=1
        else :
            freq[i] =1
    return freq


s = "banana"
print(count_freq(s))  


# Medium-Hard Dictionary Questions

# Merge dictionaries and sum values for duplicate keys

def merge_dict(d1,d2):
    merged = d1.copy()
    for key , value in d2.items():
        if key in merged:
            merged[key] += value 
        else :
            merged[key] = value
    return merged    

d1 = {"a": 1, "b": 2}
d2 = {"b": 3, "d": 4}
print(merge_dict(d1,d2))



def top_n_keys(d, n):
    sorted_keys = sorted(d, key=d.get, reverse=True)
    return sorted_keys[:n]

d = {"a": 10, "b": 50, "c": 30}
print(top_n_keys(d, 2))  


# # Group words by their first letter
# def first_letter(words):
#     result ={}
#     for  word in words :
#         first = word[0]
#         if first in result :
#             result[first].append(word)
#         else:
#             result[first] =word 
#     return result 

# words = ["apple","banana","apricot","blueberry"]
# print(first_letter(words))

# Count frequency of words in a list
def freq_count (words):
    result ={}
    for word in words :
        if word in result :
            result[word] +=1
        else:
            result [word] = 1
    return result 

words = ["apple","banana","apple","banana","cherry"]
print(freq_count(words))

# d = {"a":10,"b":50,"c":30} → ["b","c"]
#  Find all keys whose values are greater than the average value
def avg_grt(d):
    result =[]
    avg = sum(d.values())/len(d)
    for key in d :
        if d[key]>avg:
            result.append(key)
    return result 

d = {"a":10,"b":50,"c":30} 
print(avg_grt(d))


def missing_keys(d1, d2):
    # keys in d2 that are not in d1
    return [key for key in d2 if key not in d1]

d1 = {"a": 1, "b": 2}
d2 = {"a": 10, "b": 2, "c": 5}

print(missing_keys(d1, d2))  


def sort_by_values(d):
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True))

d = {"a": 10, "b": 50, "c": 30}
print(sort_by_values(d))  



def flatten_dict(d, parent_key='', sep='_'):
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

d = {"a": 1, "b": {"x": 10, "y": 20}}
print(flatten_dict(d)) 



def dict_intersection_sum(d1, d2):
    result = {}
    for key in d1:
        if key in d2:
            result[key] = d1[key] + d2[key]
    return result

d1 = {"a": 1, "b": 2, "c": 3}
d2 = {"b": 3, "c": 4, "d": 5}

print(dict_intersection_sum(d1, d2))  


