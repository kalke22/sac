"""
=============================================================================
PYTHON BASICS CHEAT SHEET
=============================================================================
A quick reference guide for core Python concepts, data structures, and features.
"""

# =============================================================================
# 1. LISTS (Mutable, Ordered)
# =============================================================================
my_list = [1, 2, 3, 'a', 'b']

# Operations
my_list.append(4)           # Add to end: [1, 2, 3, 'a', 'b', 4]
my_list.insert(0, 0)        # Insert at index: [0, 1, 2, 3, 'a', 'b', 4]
my_list.extend([5, 6])      # Append multiple: [0, 1, 2, 3, 'a', 'b', 4, 5, 6]
my_list.pop()               # Remove & return last item (6)
my_list.pop(1)              # Remove & return item at index 1 (1)
my_list.remove('a')         # Remove first occurrence of 'a'
my_list.reverse()           # Reverse in place
# my_list.sort()            # Sort in place (requires same types)
# sorted(my_list)           # Return new sorted list
my_list.clear()             # Empty the list
count = my_list.count(2)    # Count occurrences
idx = my_list.index(3)      # Find index of first occurrence

# Slicing: list[start:stop:step]
# my_list[1:4] (index 1 to 3), my_list[::-1] (reverse)

# =============================================================================
# 2. SETS (Mutable, Unordered, Unique Elements)
# =============================================================================
my_set = {1, 2, 3}
empty_set = set()           # Note: {} creates an empty dict, not a set

# Operations
my_set.add(4)               # Add element
my_set.update([5, 6])       # Add multiple elements
my_set.remove(6)            # Remove element (raises KeyError if not found)
my_set.discard(10)          # Remove element (safe, no error if not found)
my_set.pop()                # Remove & return arbitrary element
my_set.clear()              # Empty the set

set_a, set_b = {1, 2}, {2, 3}
union = set_a | set_b               # {1, 2, 3} (or set_a.union(set_b))
intersection = set_a & set_b        # {2} (or set_a.intersection(set_b))
diff = set_a - set_b                # {1} (or set_a.difference(set_b))
sym_diff = set_a ^ set_b            # {1, 3} (or set_a.symmetric_difference(set_b))

# =============================================================================
# 3. TUPLES (Immutable, Ordered)
# =============================================================================
my_tuple = (1, 2, 3, 2)
single_tuple = (1,)         # Comma needed for single-element tuple

# Operations (Very limited since immutable)
count = my_tuple.count(2)   # Count occurrences (2)
idx = my_tuple.index(3)     # Find index of first occurrence (2)
# Tuples support unpacking: a, b, c, d = my_tuple

# =============================================================================
# 4. DICTIONARIES (Mutable, Key-Value Pairs, Unordered before Python 3.7)
# =============================================================================
my_dict = {'name': 'Alice', 'age': 25}

# Operations
my_dict['city'] = 'NYC'     # Add or update key
val = my_dict.get('age')    # Safe get (returns None if not found, instead of KeyError)
val = my_dict.get('x', 0)   # Safe get with default value
keys = my_dict.keys()       # dict_keys(['name', 'age', 'city'])
values = my_dict.values()   # dict_values(['Alice', 25, 'NYC'])
items = my_dict.items()     # dict_items([('name', 'Alice'), ...])

# Removal
popped_val = my_dict.pop('age')         # Remove key 'age' and return value
popped_item = my_dict.popitem()         # Remove & return last key-value pair as tuple
# del my_dict['name']                   # Delete key
my_dict.clear()                         # Empty dict
my_dict.update({'a': 1, 'b': 2})        # Merge / Update with another dict

# =============================================================================
# 5. LIST / DICT / SET COMPREHENSIONS
# =============================================================================
# List Comprehension: [expression for item in iterable if condition]
squares = [x**2 for x in range(10) if x % 2 == 0]     # [0, 4, 16, 36, 64]

# Dict Comprehension: {key_expr: val_expr for item in iterable if condition}
sq_dict = {x: x**2 for x in range(5)}                 # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Set Comprehension: {expression for item in iterable if condition}
sq_set = {x**2 for x in [-1, 1, 2]}                   # {1, 4}

# Generator Expression: (expression for item in iterable if condition)
gen = (x**2 for x in range(10))                       # Lazy evaluation

# =============================================================================
# 6. LAMBDA FUNCTIONS, MAP, FILTER, REDUCE
# =============================================================================
# lambda arguments: expression
add = lambda x, y: x + y
print(add(2, 3))  # 5

nums = [1, 2, 3, 4]
# map: apply function to all items
mapped = list(map(lambda x: x*2, nums))               # [2, 4, 6, 8]

# filter: keep items where function returns True
filtered = list(filter(lambda x: x % 2 == 0, nums))   # [2, 4]

# reduce (requires functools): cumulative application
from functools import reduce
product = reduce(lambda x, y: x * y, nums)            # 24

# Sort with lambda key
words = ["apple", "banana", "cherry"]
words.sort(key=lambda w: len(w))                      # Sort by length

# =============================================================================
# 7. CLASSES AND OBJECTS (OOP)
# =============================================================================
class Animal:
    """Base class for animals."""
    species_count = 0  # Class attribute

    def __init__(self, name):
        self.name = name  # Instance attribute
        Animal.species_count += 1

    def speak(self):
        """Instance method"""
        return "Some sound"

    @classmethod
    def get_count(cls):
        """Class method: takes class as first arg"""
        return cls.species_count

    @staticmethod
    def is_alive():
        """Static method: no implicit self or cls args"""
        return True

# Inheritance
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # Call parent constructor
        self.breed = breed
    
    def speak(self):            # Method Overriding
        return "Woof!"

dog = Dog("Buddy", "Golden Retriever")
print(dog.speak())              # "Woof!"
print(Animal.get_count())       # 1

# =============================================================================
# 8. FILE HANDLING
# =============================================================================
# Using 'with' is a best practice, as it automatically closes the file
# Modes: 'r' (read), 'w' (write, truncates), 'a' (append), 'r+' (read & write), 'b' (binary)

# Write to file
with open("example.txt", "w", encoding="utf-8") as file:
    file.write("Hello World\nLine 2")

# Read from file
with open("example.txt", "r", encoding="utf-8") as file:
    content = file.read()       # Read entire file
    # file.seek(0)              # Reset cursor to start
    # lines = file.readlines()  # Read lines into a list
    # for line in file:         # Iterate line by line (memory efficient)
    #     print(line.strip())

# Note: file is automatically closed outside the 'with' block.
