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
