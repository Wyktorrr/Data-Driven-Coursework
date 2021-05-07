import os
import sys
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from utilities import load_points_from_file as lpf
from utilities import view_data_segments as vds

def least_squares(X, Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

command_arguments_list = sys.argv
xs, ys = lpf(str(command_arguments_list[1]))

fig, ax = plt.subplots()
ax.set_xlabel("$x_\lambda$")
ax.set_ylabel("$y_\lambda$")
ax.scatter(xs, ys)

number_of_input_points = len(xs) 
number_of_segments_to_be_ploted = number_of_input_points // 20

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
       
def plot_linear(X, Y):
    #return the slope and the vertical offset of a linear function
    ones = np.ones(X.shape)
    x_e = np.column_stack((ones, X))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(Y)
    return v[:]

#a, b = plot_linear(xs, ys)
#print(a, b)                    

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
    y_coordinates = np.array(Y)
    extended_matrice = np.column_stack(((np.ones(X.shape)), X, X ** 2, X ** 3))
    x_coordinates = np.column_stack(((np.ones(X.shape)), X, X ** 2, X ** 3))
    estimated_parameters = np.linalg.inv(x_coordinates.T.dot(extended_matrice)).dot(x_coordinates.T).dot(y_coordinates)
    return estimated_parameters    

def test_data(X, Y, j):
    xs_test = X[j:j+5]
    ys_test = Y[j:j+5]
    xs_train = X[j+5:20]
    ys_train = Y[j+5:20]
    xs_train = np.append(X[:j], xs_test)
    ys_train = np.append(Y[:j], ys_test)

    return xs_train, xs_test, ys_train, ys_test

def calculate_line(X, Y, function_type):
    if function_type == "polynomial":
            coefficients = plot_polynomial(X, Y)
            x = X
            estimated_output = coefficients[2]*x**2 + coefficients[1]*x + coefficients[0]

    elif function_type == "linear":
        slope, offset = plot_linear(X, Y)
        x = X
        estimated_output = slope * x + offset

    else:
        parameters = unknown_function(X, Y)
        x = X
        estimated_output = parameters[0] + parameters[1]*x + parameters[2]*x**2 + parameters[3]*x**3

    return x, estimated_output 

def plot_line(x, estimated_output):
    plt.plot(x, estimated_output, c = "r")           

reconstruction_error = 0

#Split in chanks of 20 points
for i in range (0, number_of_input_points, 20):
    chunk_xs = xs[i:i+20]
    chunk_ys = ys[i:i+20]

    error_linear = 0
    error_polynomial = 0
    error_unknown = 0

    for j in range(0, 20, 5):
        xs_train, xs_test, ys_train, ys_test = test_data(chunk_xs, chunk_ys, j)

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

    x, y_hat = calculate_line(chunk_xs, chunk_ys, type)
    reconstruction_error += np.sum((chunk_ys - y_hat) ** 2)

    if sys.argv.__contains__("--plot"):
        plot_line(x, y_hat)


reconstruction_error = reconstruction_error / (len(ys) // 20)
print(reconstruction_error)

if sys.argv.__contains__("--plot"):
    colour = np.concatenate([[i] * 20 for i in range(number_of_segments_to_be_ploted)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c = colour)
    plt.show()