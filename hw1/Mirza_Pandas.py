#!/usr/bin/python3
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt


def question_1():
    """
    1. Import pandas package
    """
    pass


def question_2():
    """
    2. Create a pandas series s as [4,5,6].
       (1) Print s.
       (2) Print the type of s.
       (3) Print the shape of s.
    """
    arr = [4, 5, 6]
    s = pd.Series(arr)
    print("Question 2")
    print(type(s))
    print(s.shape)
    print()


def question_3():
    """
    3. Create a DataFrame d based on the given dict.
       (1) Print the shape of d.
       (2) Print the type of values in d.
       (3) Print the index of d.
       (4) Print the columns of d.
       (5) Print the summary of d (use function describe()).
    """
    dic = {
        "name": ["Andy", "James", "Lucy"],
        "age": [18, 20, 22],
        "gender": ["male", "male", "female"],
    }
    d = pd.DataFrame(dic)
    print("Question 3")
    print(d.shape)
    print(d.dtypes)
    print(d.index)
    print(d.columns)
    print(d.describe())
    print()
    pass


def question_4():
    """
    4. Select data in DataFrame.
       (1) Select and print one column (name) from the previous DataFrame d.
       (2) Select and print the first two rows from d.
       (3) Select and print the first two columns from d.
    """
    dic = {
        "name": ["Andy", "James", "Lucy"],
        "age": [18, 20, 22],
        "gender": ["male", "male", "female"],
    }
    d = pd.DataFrame(dic)
    print("Question 4")
    print(d["name"])
    print(d[0:2])
    print(d.iloc[:, :2])
    print()

    pass


def question_5():
    """
    5. Read csv file.
       (1) Read salary.csv file using read_csv() function.
    """
    df = pd.read_csv(r"salary.csv")
    print("Question 5")
    print("shape: ", df.shape)
    print()
    pass


def question_6():
    """
    6. Plot DataFrame.
       (1) Import plot package.
       (2) Plot Salary in data from question 5.
    """
    df = pd.read_csv(r"salary.csv")
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    plt.plot(x, y)
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.show()
    print("Question 6")
    print("Plot")

    pass


if __name__ == "__main__":
    question_1()
    question_2()
    question_3()
    question_4()
    question_5()
    question_6()
