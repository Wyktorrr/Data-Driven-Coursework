import os
import sys
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from utilities import load_points_from_file as lpf
from utilities import view_data_segments as vds

command_arguments_list = sys.argv
xs, ys = lpf(str(command_arguments_list[1]))

fig, ax = plt.subplots()
ax.set_xlabel("$x_\lambda$")
ax.set_ylabel("$y_\lambda$")
ax.scatter(xs, ys)

"""
number_of_input_points = len(xs) 
number_of_segments_to_be_ploted = number_of_input_points // 20
"""
n = len(xs)

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
            estimated_output = coefficients[2]*x*2 + coefficients[1]*x + coefficients[0]

    elif function_type == "linear":
        slope, offset = plot_linear(X, Y)
        x = X
        estimated_output = slope * x + offset

    else:
        parameters = unknown_function(X, Y)
        x = X
        estimated_output = parameters[0] + parameters[1]*x + parameters[2]*x**2 + parameters[3]*x**3

    return x, estimated_output 

