# Practice using numpy

#

# ###  Array operations in numpy

# #### 1. Import numpy package
import numpy as np
import matplotlib.pyplot as plt


def q2():
    """
    2. Create a one-dimensional array a and initialize as [4,5,6].
    (1) Print the type of a
    (2) Print the shape of a
    (3) Print the first element in a (the value should be 4)
    """
    a = [4, 5, 6]
    variable_type = type(a)
    shape = np.shape(a)
    first_element = a[0]

    print(
        "Question 2:\n",
        f"variable_type = {variable_type} \n",
        f"shape = {shape} \n",
        f"first element={first_element}\n",
        "\n",
    )


def q3():
    """
    3. Create a two-dimensional array b and initialize as [ [4,5,6], [1,2,3] ].
    (1) Print the shape of b
    (2) Print b(0,0), b(0,1), b(1,1) (the values should be 4,5,2)
    """
    b = [
        [4, 5, 6],
        [1, 2, 3],
    ]
    shape = np.shape(b)
    values = [b[0][0], b[0][1], b[1][1]]

    print(
        "Question 3:\n",
        f"shape = {shape} \n",
        f"elements = {values}\n",
        "\n",
    )


def q4():
    """
    4.
    (1) Create a matrix a, which is all 0, of size 3x3
    (2) Create a matrix b, which is all 1, of size 4x5
    (3) Create a unit matrix c, of size 4x4
    (4) Create a random matrix d, of size 3x2
    """
    a = np.zeros((3, 3))
    b = np.ones((4, 5))
    c = np.identity(4)
    d = np.random.rand(3, 2)

    print(
        "Question 4:\n",
        f"All zeros 3x3:\n{a}\n",
        f"All ones 4x5:\n{b}\n",
        f"Unit matrix 4x4:\n {c}\n",
        f"Random matrix 3x2:\n{d}\n\n",
    )


def q5():
    """
    5. Create an array a and initialize as [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]].
    (1) Print a
    (2) Put the 0th and 1st rows, 2nd and 3rd columns of  array a into  array b, then print b.
    (3) Print b(0,0)
    """
    a = np.arange(1, 13).reshape((4, 3))
    b = a[0:2, 1:3]
    print(
        "Question 5:\n",
        f"array 'a':\n{a}\n",
        f"b[0,0]: {b[0][0]}\n\n",
    )



def q6():
    """
    6. Put all the elements of the last two rows of array a (question 5) into an new array c
    (1) Print c
    (2) Print the last element of the first row in c (hint: Use -1 for the last element)
    """
    a = np.arange(1, 13).reshape((4, 3))
    c = a[-2:]

    last_element = c[0][-1]
    print(
        "Question 6:\n",
        f"array 'c':\n{c}\n",
        f"Last Element: {last_element}\n\n",
    )  


# ###  Array arithmetical operation in numpy
def q7():
    """
    7.
    (1) Create  an array x and initialize as [[1, 2], [3, 4]], dtype=np.float64.
    (2) Create  an array y and initialize as [[5, 6], [7, 8]], dtype=np.float64.
    (3) Print x + y and np.add(x, y)
    (4) Print x - y and np.subtract(x, y）
    (5) Print x * y, np.multiply(x, y）and np.dot(x,y), and compare the results
    (6) Print x / y and np.divide(x,y)
    (7) Print  the sqaure of x (hint: np.sqrt())
    (8) Print x.dot(y) and np.dot(x,y)
    """
    x = np.array([[1,2],[3,4]],dtype=np.float64)
    y = np.array([[5,6],[7,8]],dtype=np.float64)
    add = x+y,np.add(x,y)
    subtract = x-y,np.subtract(x,y)
    multi = x*y,np.multiply(x,y)
    divide = x/y,np.divide(x,y)
    sqre = np.sqrt(x)
    dot = x.dot(y), np.add(x,y)

    print(
        "Question 7:\n",
        f"Add: {add}\n",
        f"Subtract: {subtract}\n",
        f"Multi: {multi}\n",
        f"Divide: {divide}\n",
        f"Square: {sqre}\n",
        f"Dot: {dot}\n\n",
    )
    return x


def q8():
    """
    8. sum (use the array x in question 7)
    (1) print the sum of x
    (2) print the sum of the rows of x  (hint: axis = 0)
    (3) print the sum of the columns of x  (hint: axis = 1)
    """
    x = np.array([[1,2],[3,4]],dtype=np.float64)
    s = np.sum(x)
    sum_rows = np.sum(x,axis=0) 
    sum_cols = np.sum(x,axis=1)
    print(
        "Question 8:\n",
        f"Total Sum: {s}\n",
        f"Row Sum: {sum_rows}\n",
        f"Col Sum: {sum_cols}\n",


    )

def q9():
    """
    9. mean (use the array x in question 7)
    (1) print the mean of x
    (2) print the mean of the rows of x  (hint: axis = 0)
    (3) print the mean of the columns of x  (hint: axis = 1)
    """
    x = np.array([[1,2],[3,4]],dtype=np.float64)
    mean = np.mean(x)
    mean_rows = np.mean(x,axis=0)
    mean_cols = np.mean(x,axis=1)


def q10():
    """
    10. Using the array x in question 7 to get the matrix transpose of x
    """
    x = np.array([[1,2],[3,4]],dtype=np.float64)
    transposed = np.transpose(x)



def q11():
    """
    11. Get the index of the max elements in x in question 7
    (1) print the index of the max element of x
    (2) print the index of the max elementof in the rows of x  (hint: axis = 0)
    (3) print the index of the max elementof in the columns of x  (hint: axis = 1)
    """
    x = np.array([[1,2],[3,4]],dtype=np.float64)
    max_element = np.argmax(x)
    max_rows = np.argmax(x,axis=0)
    max_colms = np.argmax(x,axis=1)



def q12():
    """
    12. Plot
     axis X:  x = np.arange(0, 100, 0.1)    axis Y: y=x*x （hint: use  matplotlib.pyplot package）
    """
    x = np.arange(0, 100, 0.1)
    y=x*x

    plt.plot(x,y)
    plt.show()
    

def q13():
    """
    13. Plot sin() and cos()   (hint: use np.sin(), np.cos() and matplotlib.pyplot package）
    (1) axis X: x = np.arange(0, 3 * np.pi, 0.1)   axis Y: y=sin(x)
    (2) axis X: x = np.arange(0, 3 * np.pi, 0.1)   axis Y: y=cos(x)
    """

    x = np.arange(0, 3 * np.pi, 0.1)   
    y=np.sin(x)
    plt.title("Sine Wave")
    plt.plot(x,y)
    plt.show()
    x = np.arange(0, 3 * np.pi, 0.1)   
    y=np.cos(x)
    plt.title("Cosine Wave")
    plt.plot(x,y)
    plt.show()
    


if __name__ == "__main__":
    q2()
    q3()
    q4()
    q5()
    q6() 
    q7()
    q8()
    q9()
    q10()
    q11()
    q12()
    q13()
