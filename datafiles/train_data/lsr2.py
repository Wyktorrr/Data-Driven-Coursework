import os
import sys
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from utilities import load_points_from_file as lpf
from utilities import view_data_segments as vds

#global number_of_segments_to_be_ploted, ax, xs, ys, cross_validation_x, cross_validation_y

"""
def create_path(selectedFile): 
    #universalPath = "C:/Users/UserDell/Documents/University/Data Driven CW/coursework/datafiles/train_data/"  
    
    universalPath = os.path.abspath("datafiles/train_data/") 
    universalPath += selectedFile
    return universalPath
    
    return os.path.abspath(selectedFile)
"""    


#xs, ys = lpf(os.path.abspath("basic_1.csv"))  


"""
def create_path(selectedFile): 
    universalPath = "C:/Users/UserDell/Documents/University/Data Driven CW/coursework/datafiles/train_data/"   
    universalPath += selectedFile
    return universalPath
    
xs, ys = lpf(create_path("basic_1.csv"))  

#filename = "datafiles"
#filename.append("\")
#filename.append(str(sys.argv[1]))
#xs, ys = lpf(str(sys.argv[1]))

fig, ax = plt.subplots()
ax.set_xlabel("$x_\lambda$")
ax.set_ylabel("$y_\lambda$")
ax.scatter(xs, ys, label="data", s = 200)

number_of_segments_to_be_ploted = len(xs) // 20

def create_chunks_of_data(current_point):
    created_list_x = []
    created_list_y = []
    for i in range(20):
        created_list_x = np.append(created_list_x, xs[20 * current_point + i])
        created_list_y = np.append(created_list_y, ys[20 * current_point + i])
    return created_list_x, created_list_y    
       
#Method using Chebyshev-polynomials - code adapted from week 15 - overfitting#

"""
"""
def least_squares(x, y):
    wh = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return wh
"""

"""
def least_squares(x, y):
    mul_x_xT = np.matmul(x, x.T)
    mul_v2 = np.matmul(mul_x_xT, x.T)
    mul_final = np.matmul(mul_x_xT, mul_v2)
    return np.linalg.inv(mul_final)
"""
"""
def least_squares(X, Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

def cheb(xs, c):
    # c is int
    coefs = c*[0] + [1]
    return np.polynomial.chebyshev.chebval(xs, coefs)

def chebx(x, order):
    xs = cheb(x, 0)
    for c in range(order - 1):
        xs = np.vstack([xs, cheb(x, c + 1)])
    return xs.T

for i in range(number_of_segments_to_be_ploted):
    chunk_xs, chunk_ys = create_chunks_of_data(i)
    x = chebx(chunk_xs, 4)
    wh = least_squares(x, chunk_ys)
    ax.plot(chunk_xs, chebx(chunk_xs, 4).dot(wh), 'r', lw = 4, label="fitted line")  
    ax.legend()

vds(xs, ys)     

def square_error(true_y, estimated_y):
    return np.sum((true_y - estimated_y) ** 2)

def shuffle_chunks(array):
        random.shuffle(array)
        return array    

def cross_validation():
    chunk_data = np.linspace(0, 20, num=20, endpoint=True, retstep=False, dtype=int, axis=0)
    #chunk_data = np.array([0, 1, 2, 3, 4])
    #chunk_data = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,16,19])
    #chunk_data = np.linspace(0, 20, 20)
    shuffle_chunks(chunk_data)
    chunk_data_random_x = []
    chunk_data_random_y = []
    
    for i in range(20):
        chunk_data_random_x = np.append(chunk_data_random_x, chunk_data[i])
        chunk_data_random_y = np.append(chunk_data_random_y, chunk_data[i])

    train_x_data = np.zeros((15,), dtype=int)
    test_x_data = np.zeros((5,), dtype=int)
    train_y_data = np.zeros((15,), dtype=int)
    test_y_data = np.zeros((5,), dtype=int)

    cross_validation_x = chunk_data_random_x[:5]
    cross_validation_x = np.vstack([cross_validation_x, chunk_data_random_x[5:10]])
    cross_validation_x = np.vstack([cross_validation_x, chunk_data_random_x[10:15]])
    cross_validation_x = np.vstack([cross_validation_x, chunk_data_random_x[15:]])

    cross_validation_y = chunk_data_random_y[:5]
    cross_validation_y = np.vstack([cross_validation_y, chunk_data_random_y[5:10]])
    cross_validation_y = np.vstack([cross_validation_y, chunk_data_random_y[10:15]])
    cross_validation_y = np.vstack([cross_validation_y, chunk_data_random_y[15:]])

    for i in range(4):
        new_train_x_data = []
        new_test_x_data = []
        new_train_y_data = []
        new_test_y_data = []
        new_test_x_data = cross_validation_x[i]
        new_test_y_data = cross_validation_y[i]
        for j in range(4):
            if (j != i):
                new_train_x_data = np.append(new_train_x_data, cross_validation_x[j])
                new_train_y_data = np.append(new_train_y_data, cross_validation_y[j])
        train_x_data = np.vstack([train_x_data, new_train_x_data])  
        train_y_data = np.vstack([train_y_data, new_train_y_data])      
        test_x_data = np.vstack([test_x_data, new_test_x_data])  
        test_y_data = np.vstack([test_y_data, new_test_y_data]) 
    return train_x_data, train_y_data, test_x_data, test_y_data       

for i in range(number_of_segments_to_be_ploted):
    x_current = []
    y_current = []
    for j in range(20):
        x_current = np.append(x_current, xs[20 * i + j])
        y_current = np.append(y_current, ys[20 * i + j])

    cross_validation_error_list = []
    xs_train, ys_train, xs_test, ys_test = cross_validation()

    order = 2

    while(order < 20):
        wh = least_squares(chebx(xs_train, order), ys_train)
        output = chebx(xs_test, order).dot(wh)
        cross_validation_error = square_error(ys_test, output).mean()
        cross_validation_error_list = np.append(cross_validation_error_list, cross_validation_error)
        order += 1

    output = chebx(np.sin(xs_test), 2).dot(least_squares(chebx(np.sin(xs_train), 2), ys_train))
    cross_validation_error = square_error(ys_test, output).mean()

#increase the degree (order) of the polynomial until error becomes smaller
    degree = 1
    first_error = cross_validation_error_list[0]
    while(cross_validation_error_list[degree] < first_error):
        first_error = cross_validation_error_list[degree]
        degree += 1

    if (first_error < cross_validation_error):
        wh = least_squares(chebx(xs_train, i + 1), ys_train)
        output = chebx(x_current, i + 1).dot(wh)
        print(output)
        ax.plot(x_current, output, 'r', label = "fitted")
        total_error = square_error(y_current, output)
        print(total_error)
    else:
        inp = chebx(np.sin(xs_train), 2)
        wh = least_squares(inp, ys_train)
        ax.plot(x_current, chebx(np.sin(x_current), 2).dot(wh), 'r', label = "fitted")
        total_error = square_error(y_current, chebx(np.sin(x_current), 2).dot(wh))
        print(total_error)    

plt.show()
"""

"""
def cheb(xs, c):
    coefs = c * [0] + [1]
    return np.polynomial.chebyshev.chebval(xs, coefs)

def chebx(X, order):
    xs = cheb(X, 0)
    for c in range(order - 1):
        xs = np.vstack([xs, cheb(X, c + 1)])
    return xs.T 

def square_error(true_output, estimated_output):
    return np.sum((true_output - estimated_output) ** 2)
"""   

"""

total_reconstruction_error = 0

number_of_segments_to_be_ploted = len(xs) // 20

for i in range(number_of_segments_to_be_ploted):
    create_chunk_list_xs = []
    create_chunk_list_ys = []
    for j in range(20):
        create_chunk_list_xs = np.append(create_chunk_list_xs, xs[20 * i + j])
        create_chunk_list_ys = np.append(create_chunk_list_xs, xs[20 * i + j])

    reconstruction_error_list = []

    xs_train = create_chunk_list_xs[:15]
    xs_test = create_chunk_list_xs[15:]
    ys_test = create_chunk_list_ys[:15]
    ys_train = create_chunk_list_ys[15:]

    for degree in range(2, 20):
        wh = least_squares(chebx(xs_train, degree), ys_train)
        estimated_output = chebx(xs_test, degree).dot(wh)
        cross_validation_error = square_error(ys_test, estimated_output).mean()
        reconstruction_error_list = np.append(reconstruction_error_list, cross_validation_error)

    estimated_output = chebx(np.sin(xs_test), 2).dot(least_squares(np.sin(xs_train), 2), ys_train)
    cross_validation_error = square_error(ys_test, estimated_output).mean()

    first_error = reconstruction_error_list[0]
    order = 1
    while(reconstruction_error_list[i] < first_error):
        first_error = reconstruction_error_list[i]
        i += 1    
"""   
#######################

"""
def least_squares(X, Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
command_arguments_list = sys.argv
xs, ys = lpf(str(command_arguments_list[1]))

fig, ax = plt.subplots()
ax.set_xlabel("$x_\lambda$")
ax.set_ylabel("$y_\lambda$")
ax.scatter(xs, ys)

number_of_segments_to_be_ploted = len(xs) // 20

n = len(xs) 

def plot_linear(X, Y):
    #return the slope and the vertical offset of a linear function
    ones = np.ones(X.shape)
    x_e = np.column_stack((ones, X))
    v = np.linalg.inv(x_e.transpose().dot(x_e)).dot(x_e.transpose()).dot(Y)
    return v[:]

#a, b = plot_linear(xs, ys)
#print(a, b)

def find_function(linear_error, polynomial_error, unknown_error):
    if linear_error < polynomial_error:
        if linear_error < unknown_error:
            function = "linear"
        else:
            function = "unknown function"
    elif polynomial_error < unknown_error:
        function = "polynomial"
    else:
        function = "unknown function"

    return function                    

def plot_polynomial(X, Y):
    
    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0
    y1 = 0
    xy = 0
    xxy = 0
    y2 = 0
    for i in range(len(X)):
        x1 += X[i]
        x2 += X[i]*X[i]
        y2 += Y[i]*Y[i]
        y1 += Y[i]
        x3 += X[i]*X[i]*X[i]
        xy += X[i]*Y[i]
        xxy += X[i]*X[i]*Y[i]
        x4 += X[i]*X[i]*X[i]*X[i]

    A = np.array([[len(X), x1, x2], [x1, x2, x3], [x2, x3, x4]]) 
    B = np.array([y1, xy, xxy])
    coef = np.linalg.inv(A).dot(B)

    return coef

def unknown_function(X, Y):
    y_i = np.array(Y)
    M = np.column_stack(((np.ones(X.shape)), X, X ** 2, X ** 3))
    x_i = np.column_stack(((np.ones(X.shape)), X, X ** 2, X ** 3))
    a = np.linalg.inv(x_i.T.dot(M)).dot(x_i.T).dot(y_i)
    return a


def test_data(X, Y, j):
    X_test = X[j:j+5]
    Y_test = Y[j:j+5]
    X_train = X[j+5:20]
    Y_train = Y[j+5:20]
    X_train = np.append(X[:j], X_test)
    Y_train = np.append(Y[:j], Y_test)

    return X_train, X_test, Y_train, Y_test
    

def calculate_line(X, Y, function):
    if function == "polynomial":
            coef = plot_polynomial(X, Y)
            x = X
            y_hat = coef[2]*x**2 + coef[1]*x + coef[0]

    elif function == "linear":
        slope, offset = plot_linear(X, Y)
        x = X
        y_hat = slope * x + offset

    else:
        coef = unknown_function(X, Y)
        x = X
        y_hat = coef[0] + coef[1]*x + coef[2]*x**2 + coef[3]**3

    return x, y_hat 

def plot_line(x, y_hat):
    plt.plot(x, y_hat, c = "g")           

X_test = []
X_train = []
Y_test = []
Y_train = []
error = 0

def shuffle_chunks(array):
        random.shuffle(array)
        return array   

def cross_validation():
    chunk_data = np.linspace(0, 20, num=20, endpoint=True, retstep=False, dtype=int, axis=0)
    #chunk_data = np.array([0, 1, 2, 3, 4])
    #chunk_data = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,16,19])
    #chunk_data = np.linspace(0, 20, 20)
    shuffle_chunks(chunk_data)
    chunk_data_random_x = []
    chunk_data_random_y = []
    
    for i in range(20):
        chunk_data_random_x = np.append(chunk_data_random_x, chunk_data[i])
        chunk_data_random_y = np.append(chunk_data_random_y, chunk_data[i])

    train_x_data = np.zeros((15,), dtype=int)
    test_x_data = np.zeros((5,), dtype=int)
    train_y_data = np.zeros((15,), dtype=int)
    test_y_data = np.zeros((5,), dtype=int)

    cross_validation_x = chunk_data_random_x[:5]
    cross_validation_x = np.vstack([cross_validation_x, chunk_data_random_x[5:10]])
    cross_validation_x = np.vstack([cross_validation_x, chunk_data_random_x[10:15]])
    cross_validation_x = np.vstack([cross_validation_x, chunk_data_random_x[15:]])

    cross_validation_y = chunk_data_random_y[:5]
    cross_validation_y = np.vstack([cross_validation_y, chunk_data_random_y[5:10]])
    cross_validation_y = np.vstack([cross_validation_y, chunk_data_random_y[10:15]])
    cross_validation_y = np.vstack([cross_validation_y, chunk_data_random_y[15:]])

    for i in range(4):
        new_train_x_data = []
        new_test_x_data = []
        new_train_y_data = []
        new_test_y_data = []
        new_test_x_data = cross_validation_x[i]
        new_test_y_data = cross_validation_y[i]
        for j in range(4):
            if (j != i):
                new_train_x_data = np.append(new_train_x_data, cross_validation_x[j])
                new_train_y_data = np.append(new_train_y_data, cross_validation_y[j])
        train_x_data = np.vstack([train_x_data, new_train_x_data])  
        train_y_data = np.vstack([train_y_data, new_train_y_data])      
        test_x_data = np.vstack([test_x_data, new_test_x_data])  
        test_y_data = np.vstack([test_y_data, new_test_y_data]) 
    return train_x_data, train_y_data, test_x_data, test_y_data       

#Split in chanks of 20 points
for i in range (0, n, 20):
    slice_x = xs[i:i+20]
    slice_y = ys[i:i+20]

    error_linear = 0
    error_polynomial = 0
    error_unknown = 0

    for j in range(0, 20, 5):
        xs_train, xs_test, ys_train, ys_test = test_data(slice_x, slice_y, j)

        x, y_hat = calculate_line(xs_train, ys_train, "linear")
        error_linear += np.sum((ys_test - y_hat[j:j + 5]) ** 2)
        error_linear = error_linear / ((j + 5) / 5)

        x, y_hat = calculate_line(xs_train, ys_train, "polynomial")
        error_polynomial += np.sum((ys_test - y_hat[j:j + 5]) ** 2)
        error_polynomial = error_polynomial / ((j + 5) / 5)

        x, y_hat = calculate_line(xs_train, ys_train, "unknown")
        error_unknown += np.sum((ys_test - y_hat[j:j + 5]) ** 2)
        error_unknown = error_unknown / ((j + 5) / 5)

    type = find_function(error_linear, error_polynomial, error_unknown)    

    x, y_hat = calculate_line(slice_x, slice_y, type)
    error += np.sum((slice_y - y_hat) ** 2)

    if sys.argv.__contains__("--plot"):
        plot_line(x, y_hat)


error = error / (len(ys) // 20)
print(error)

if sys.argv.__contains__("--plot"):
    num_segments = n // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c = colour)
    plt.show()

"""    
