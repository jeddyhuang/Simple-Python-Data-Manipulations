#!/usr/bin/env python
# coding: utf-8

# In[5]:


import math
import re
from bs4 import BeautifulSoup
import urllib.request
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# In[ ]:


def sin_taylor(x, n):
    val = 0
    for temp in range(1, n+1):
        step = (temp * 2 - 1)
        if temp % 2 == 1:
            val = val + x**step / math.factorial(step)
        else:
            val = val - x**step / math.factorial(step)
    return val


# In[ ]:


print(sin_taylor(math.pi, 1))
print(sin_taylor(math.pi, 4))
print(sin_taylor(math.pi/2, 1))
print(sin_taylor(math.pi/2, 3))


# In[ ]:


def pythagorean_triple(a, b, c):
    if a <= 0 or b <= 0 or c <= 0:
        return False
    if a**2 + b**2 == c**2:
        return True
    elif a**2 + c**2 == b**2:
        return True
    elif b**2 + c**2 == a**2:
        return True
    return False


# In[ ]:


print(pythagorean_triple(3, 4, 5))
print(pythagorean_triple(6, 4, 7))
print(pythagorean_triple(-1, 3, 2))
print(pythagorean_triple(13, 12, 5))
print(pythagorean_triple(0, 2, 2))


# In[6]:


def triangle_area(a, b, c):
    if a <= 0 or b <= 0 or c <= 0:
        return 'None'
    if a + b > c and b + c > a and c + a > b:
        p = (a + b + c)/2
        return math.sqrt(p*(p-a)*(p-b)*(p-c))
    return 'None'


# In[7]:


print(triangle_area(1, 1, 2))
print(triangle_area(3, 4, 5))
print(triangle_area(3, 2, 4))
print(triangle_area(0, 3, 3))
print(triangle_area(7, 10, 6))


# In[3]:


def decode(secret):
    final_string = ""
    for x in secret:
        curr = chr(x).swapcase()
        final_string = final_string + curr
    return final_string


# In[4]:


print(decode((104, 69, 76, 76, 111)))
print(decode((100, 101, 102, 103)))
print(decode((112, 121, 84, 72, 111, 78)))
print(decode((90, 88, 89)))


# In[13]:


def unique_intersection(list1, list2, list3):
    int1 = list(set(list1) & set(list2))
    int2 = list(set(int1) & set(list3))
    return int2


# In[14]:


print(unique_intersection([1, 2, 3], [2, 3], [2, 3, 4]))
print(unique_intersection([], [1, 2], [2]))
print(unique_intersection([3, 3, 3], [3, 3, 3], [3, 3, 3]))
print(unique_intersection([1, 2, 3, 4], [5, 4, 3, 2], [3, 4, 5, 6]))


# In[7]:


def str_duplicate(word):
    duplicates = ""
    for char in word:
        if word.count(char) > 1:
            if duplicates.count(char) < 1:
                duplicates = duplicates  + char
    return duplicates


# In[9]:


print(str_duplicate('just'))
print(str_duplicate('banana'))
print(str_duplicate('Alexander'))
print(str_duplicate('Abcacdbcecde'))


# In[23]:


def combine_dict(dict_list):
    final = {}
    for subdict in dict_list:
        for key in subdict:
            if key not in final:
                final[key] = subdict[key]
            else:
                final[key] = final[key] * subdict[key]
    return final


# In[24]:


print(combine_dict([{'A': 4, 'B': 2}, {'A': 2}]))
print(combine_dict([{'A': 4, 'B': 2}, {'A': 3, 'B': 6}]))
print(combine_dict([{'A': 4, 'B': 2}, {'A': 3, 'B': 3}, {'A': 2, 'C': 1}]))
print(combine_dict([{'A': 2, 'B': 5, 'C': 12, 'F': 20}, {'B': 4, 'E': 12}, {'A': 10, 'D': 26}]))


# In[29]:


def swap_dict(dictionary):
    final = {}
    for key in dictionary:
        if dictionary[key] not in final:
            final[dictionary[key]] = [key]
        else:
            final[dictionary[key]] = final[dictionary[key]] + [key]
    return final


# In[30]:


print(swap_dict({}))
print(swap_dict({'A': 1, 'B': 2, 'C': 3}))
print(swap_dict({'A': 1, 'B': 1, 'C': 3}))
print(swap_dict({'A': 1, 'B': 1, 'C': 1, 'D': 1}))


# In[5]:


def swap_chars(word):
    if not word:
        return -1
    sum = 0
    for char in word:
        sum = sum + ord(char)
    ind = sum % len(word)
    strlist = list(word)
    strlist[ind], strlist[len(word)-1-ind] = strlist[len(word)-1-ind], strlist[ind]
    return ''.join(strlist)


# In[6]:


print(swap_chars(''))
print(swap_chars('Purdue'))
print(swap_chars('BoilerUp!'))


# In[26]:


def restaurant(dict1, dict2, x):
    temp = set()
    final = set()
    for key in dict1:
        if key <= x:
            for value in dict1[key]:
                temp.add(value)
    for key in dict2:
        if key <= x:
            for value in dict2[key]:
                if value in temp:
                    final.add(value)
    return final


# In[27]:


print(restaurant({}, {10: ['The Owlery', 'Lucky']}, 10))
print(restaurant({15: ['The Owlery', 'BK']}, {10: ['The Owlery', 'Lucky']}, 12))
print(restaurant({5:['Lucky'], 10:['the Tap', 'BK']}, {10:['The Owlery', 'Lucky'], 20:['BK']}, 15))
print(restaurant({15: ['The Owlery', 'BK'], 7: ['Lucky']}, {10: ['The Owlery', 'Lucky']}, 20))


# In[28]:


def gradebook(information):
    final = {}
    for student in information:
        Asum = 0
        Psum = 0
        Esum = 0
        for value in student["assignments"]:
            Asum = Asum + value
        Asum = Asum / len(student["assignments"])
        for value in student["projects"]:
            Psum = Psum + value
        Psum = Psum / len(student["projects"])
        for value in student["exams"]:
            Esum = Esum + value
        Esum = Esum / len(student["exams"])
        total = 0.35 * Asum + 0.30 * Esum + 0.35 * Psum
        if total < 50:
            final[student["name"]] = 'F'
        elif 50 <= total and total < 60:
            final[student["name"]] = 'E'
        elif 60 <= total and total < 70:
            final[student["name"]] = 'D'
        elif 70 <= total and total < 80:
            final[student["name"]] = 'C'
        elif 80 <= total and total < 90:
            final[student["name"]] = 'B'
        elif 90 <= total:
            final[student["name"]] = 'A'
    return final


# In[29]:


information = [{ "name": "Lloyd", "assignments": [90.0,97.0,75.0,92.0],
                "projects": [88.0,40.0,94.0], "exams": [75.0,90.0] }]
print(gradebook(information))
information = [{"name": "Bob", "assignments": [80, 90, 93], "projects": [83, 87, 85], "exams": [84]},
               {"name": "David", "assignments": [90, 95, 83], "projects": [87, 89, 91], "exams": [95]}]
print(gradebook(information))
information = [{"name": "Nan", "assignments": [82, 91, 93], "projects": [86, 87], "exams": [93, 94]},
               {"name": "Iliad", "assignments": [80, 75, 76], "projects": [87, 79, 81], "exams": [75, 85]},
               {"name": "Yodel", "assignments": [70, 75, 66], "projects": [77, 69, 71], "exams": [65, 70]}]
print(gradebook(information))


# In[42]:


def list_flatten(list1):
    if not list1:
        return []
    if type(list1[0]) == type(list()):
        return list_flatten(list1[0]) + list_flatten(list1[1:])
    return [list1[0]] + list_flatten(list1[1:])


# In[43]:


print(list_flatten([[1,2], [3,4], [5,6]]))
print(list_flatten(['a', ['a', 'b'],['c', ['d']], [[['d']]]]))


# In[11]:


class Student:
    def __init__(self, name):
        self.name = name
        self.record = []
        self.GPA = None

    def get_name(self):
        return self.name

    def take_course(self, course_id, credit_hours):
        course = (course_id, credit_hours, 'in taking')
        self.record.append(course)

    def release_grade(self, course_id, grade):
        for (i, item) in enumerate(self.record):
            if self.record[i][0] == course_id:
                update = list(self.record[i])
                update[2] = grade
                self.record[i] = tuple(update)

    def drop_course(self, course_id):
        for (i, item) in enumerate(self.record):
            if self.record[i][0] == course_id:
                self.record.remove(self.record[i])

    def get_record(self):
        return self.record

    def get_gpa(self):
        credits = 0
        total = 0
        for grade in self.record:
            if type(grade[2]) != str:
                credits += grade[1]
                total += grade[1] * grade[2]
        if credits != 0:
            self.GPA = total / credits
        return self.GPA


# In[12]:


student = Student('Alice')
print(student.get_name())
student.take_course('CS380', 1)
print(student.get_gpa())
print(student.get_record())
student.take_course('CS373', 3)
student.release_grade('CS380', 90.0)
print(student.get_record())
student.take_course('CS354', 3)
student.release_grade('CS373', 87.0)
print(student.get_gpa())
print(student.get_record())
student.drop_course('CS354')
print(student.get_record())


# In[4]:


def course_registration(filename, courses):
    students = {}
    infile = open(filename, "r")
    for line in infile:
        array = line.split()
        if len(array) != 0:
            students[array[0]] = array[1:]
    infile.close()
    final = {}
    for course in courses:
        final[course] = []
        for student in students:
            for value in students[student]:
                if value == course:
                    final[course] = final[course] + [student]
    return final


# In[5]:


print(course_registration("test.txt", ['CS352']))
print(course_registration("test.txt", ['CS352', 'CS180']))
print(course_registration("test.txt", ['CS402', 'CS307']))
print(course_registration("test.txt", ['CS307', 'CS380']))


# In[ ]:


def extract_numbers(filename):
    final = []
    infile = open(filename, "r")
    for line in infile:
        final += re.findall("[-]?\d*\.\d+|[-]?\d+", line)
    infile.close()
    return final


# In[24]:


def find_phone_number(url='https://www.python-course.eu/simpsons_phone_book.txt'):
    r = requests.get(url).text
    people = [[]]
    array = r.split()
    for value in array:
        people[-1] += [value]
        if re.search("\d+-|-\d+", value):
            people = people + [[]]
    found = []
    for person in people:
        if len(person) > 0:
            if len(person[0]) >= 4 and person[1] == "Neu":
                found += [" ".join(person)]
    return found


# In[25]:


print(find_phone_number())


# In[6]:


def normalize_matrix(A, B, C):
    M1 = np.dot(A, B)
    M2 = np.dot(M1, C)
    mean = np.mean(C)
    M3 = M2 * mean;
    return M3


# In[7]:


A = np.array([[2], [1]])
B = np.array([[3, 4]])
C = np.array([[1], [2]])
print(normalize_matrix(A, B, C))
A = np.ones((3, 2))
B = np.ones((2, 3))
C = np.ones((3, 4))
print(normalize_matrix(A, B, C))
A = np.array([[2, 3], [1, 4]])
B = np.array([[1, 1], [1, 0]])
C = np.array([[0, 1], [1, 1]])
print(normalize_matrix(A, B, C))


# In[26]:


def compute_mean(filename, district):
    data = pd.read_csv(filename)
    data = data.dropna()
    data = data.drop_duplicates(subset=['DISTRICT','COVID_DEATHS'], keep='last')
    subset = data[data["DISTRICT"] == district]
    return subset.max()


# In[28]:


print(compute_mean("test.csv","District 2"))
print(compute_mean("test.csv","District 4"))
print(compute_mean("test.csv","District 6"))


# In[38]:


def inspect_data(filename, district, max_test, min_test):
    data = pd.read_csv(filename)
    data = data.dropna()
    data = data.drop_duplicates(keep='last')
    count_average = data["COVID_COUNT"].mean()
    death_average = data["COVID_DEATHS"].mean()
    subset = data[(data["DISTRICT"] == district) &
                  (min_test <= data["COVID_TEST"]) & (max_test >= data["COVID_TEST"]) &
                  (data["COVID_COUNT"] > count_average/2) & (data["COVID_DEATHS"] < death_average)]
    if subset.empty:
        return None
    else: return subset.mean()


# In[39]:


print(inspect_data("test.csv", "District 1", 1000, 10))
print(inspect_data("test.csv", "District 3", 10000, 10))
print(inspect_data("test.csv", "District 5", 10000, 1000))
print(inspect_data("test.csv", "District 6", 10000, 100))


# In[12]:


def plot_func():
    t = np.arange(-2.5, 2.5, 0.01)
    s1 = np.exp(-np.pi*t**2)*np.cos(2*np.pi*t)*np.sin(2*np.pi*t)
    s2 = 5*np.exp(-t**2)
    
    plt.plot(t, s1, label = 'f(t)')
    plt.plot(t, s2, label = 'g(t)')
    plt.xlabel('t-axes')
    plt.ylabel('y-axes')
    plt.legend()
    
    plt.show()


# In[13]:


plot_func()


# In[38]:


def estimate(outcome, n):
    nums = []
    for num in range(n):
        nums += [random.randint(1,6)]
    counter = 0
    for num in nums:
        if num >= outcome:
            counter += 1
    return counter/n


# In[39]:


random.seed(50)
print(estimate(2,10))
print(estimate(5, 6))
print(estimate(2, 8))


# In[40]:


def generate(n, x, y):
    if x < 0 or y > 255:
        return None
    curr = ""
    for num in range(n):
        curr += chr(random.randint(x,y))
    return curr


# In[41]:


print(generate(5, 20, 300))
print(generate(5, -5, 128))
random.seed(1)
print(generate(7, 67, 90))
random.seed(2)
print(generate(4, 55, 109))
random.seed(3)
print(generate(5, 35, 129))


# In[3]:


def find_sub_list(list1, list2):
    if len(list2) == 0:
        return -1
    for i in range(len(list1)-len(list2)+1):
        latch = True
        for j in range(len(list2)):
            if list1[i+j] != list2[j]:
                latch = False
        if latch:
            return i
    return -1


# In[4]:


print(find_sub_list([1, 2, 3], []))
print(find_sub_list([4, 2, 3, 1, 2, 3], [2, 3]))
print(find_sub_list([3, 4, 1, 2, 5], [4, 2, 5]))
print(find_sub_list([1, 2, 5, 4, 3, 6, 7], [3, 6, 7]))


# In[35]:


def normalize_matrix(A):
    m = np.mean(A)
    v = A.var()
    row,col = A.shape
    B = np.ndarray(A.shape)
    for x in range(row):
        for y in range(col):
            B[x, y] = (A[x, y] - m)/(np.sqrt(v + 10**-9))
    return B


# In[36]:


A = np.array([[4], [2]])
print(A)
print(normalize_matrix(A))

A = np.array([[1, 2, 3, 4, 5, 6]])
print(A)
print(normalize_matrix(A))

A = np.array([[1, 2, 3], [0, 4, 2]])
print(A)
print(normalize_matrix(A))

A = np.array([[1, 2], [0, 0]])
print(A)
print(normalize_matrix(A))


# In[56]:


special_word = '<unk>'
class Vocabulary:
    def __init__(self, filepath):
        self.dictionary = {special_word: 1}
        infile = open(filepath, "r")
        for line in infile:
            array = line.lower().split()
            if len(array) != 0:
                for value in array:
                    if value not in self.dictionary:
                        self.dictionary[value] = len(self.dictionary) + 1
        infile.close()
        
    def index(self, word):
        word = word.lower()
        if word in self.dictionary:
            return self.dictionary[word]
        return self.dictionary[special_word]
        
    def encode_text(self, text):
        words = text.lower().split()
        final = []
        for word in words:
            if word in self.dictionary:
                final += [self.dictionary[word]]
            else: final += [self.dictionary[special_word]]
        return final
        
    def decode_index(self, index):
        final = ""
        for num in index:
            for key in self.dictionary:
                if num == self.dictionary[key]:
                    final = final + key + " "
        return final.rstrip()


# In[57]:


voc = Vocabulary('corpus.txt')
print(voc.index('I'), voc.index('in'), voc.index('Indiana'))
print(voc.encode_text('This sentence is irrelevant to the corpus .'))
print(voc.decode_index([2, 3, 4, 1, 7, 15, 1, 20]))
print(voc.encode_text('who ruled the country of normandy ?'))
print(voc.decode_index([398, 526, 15, 509, 48, 70, 511]))
print(voc.encode_text('there are too many words not in the corpus , we need to expand our data .'))
print(voc.decode_index([117, 15, 118]))


# In[76]:


def positive_death_rate(filepath, covid_test):
    data = pd.read_csv(filepath)
    data = data.dropna()
    data = data.drop_duplicates(keep='first')
    data = data[data["COVID_TEST"] >= covid_test]
    data["POSITIVE_RATE"] = data["COVID_COUNT"] / data["COVID_TEST"]
    data["DEATH_RATE"] = data["COVID_DEATHS"] / data["COVID_COUNT"]
    return data[["COUNTY_NAME", "POSITIVE_RATE", "DEATH_RATE"]]


# In[77]:


print(positive_death_rate('test.csv', 60000))
print(positive_death_rate('test.csv', 50000))
print(positive_death_rate('test.csv', 47920))


# In[ ]:




