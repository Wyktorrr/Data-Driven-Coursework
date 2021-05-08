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

def square_error(y, y_hat):
    return np.sum((y - y_hat) ** 2)    

command_arguments_list = sys.argv
xs, ys = lpf(str(command_arguments_list[1]))

fig, ax = plt.subplots()
ax.set_xlabel("$x_lambda$")
ax.set_ylabel("$y_lambda$")
ax.scatter(xs, ys)

number_of_input_points = len(xs) 
number_of_segments_to_be_ploted = number_of_input_points // 20

def find_function(linear_error, polynomial_error, unknown_error):
    list_errors = [linear_error, polynomial_error, unknown_error]
    min_error = min(list_errors)
    if(min_error == linear_error):
        return "linear"
    elif(min_error == polynomial_error):
        return "polynomial"
    return "unknown"       
       
def plot_linear(X, Y):
    #return the gradient and the vertical offset of a linear function
    ones = np.ones(X.shape)
    x_e = np.column_stack((ones, X))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(Y)
    return v[:]                  

def plot_polynomial(X, Y, degree):
    estimated_parameters = []
    extended_matrix = np.column_stack((np.ones(X.shape), X))
    Y = np.reshape(Y, (1, len(Y)))
    for i in range(2, degree + 1):
        extended_matrix = np.column_stack((X**i))
        estimated_parameters = np.linalg.inv(extended_matrix.T.dot(extended_matrix)).dot(extended_matrix.T).dot(Y)
    return estimated_parameters

def unknown_function(X, Y):
     extended_matrix = np.column_stack((np.ones(X.shape), X, np.sin(X)))
     estimated_parameters = np.linalg.inv(extended_matrix.T.dot(extended_matrix)).dot(extended_matrix.T).dot(Y)
     return estimated_parameters  

def test_data(X, Y, j):
    xs_test = X[j:j+5]
    ys_test = Y[j:j+5]
    xs_train = X[j+5:20]
    ys_train = Y[j+5:20]
    xs_train = np.append(X[:j], xs_test)
    ys_train = np.append(Y[:j], ys_test)

    return xs_train, xs_test, ys_train, ys_test

def compute_polynomial(X, Y, degree):
    coefficients = []
    coefficients = plot_polynomial(X, Y, degree) 
    result = [0 for i in range(len(coefficients))]
    for i in range(degree + 1):
        result += coefficients[i]*X**i  #Y_hat -> estimated_output
    return result    

def calculate_line(X, Y, function_type, degree):
    estimated_output = []
    if function_type == "polynomial":
            estimated_output = compute_polynomial(X, Y, degree)

    elif function_type == "linear":
        slope, offset = plot_linear(X, Y)
        estimated_output = slope * X + offset

    else:
        parameters = unknown_function(X, Y)
        estimated_output = parameters[0] + parameters[1]*X + parameters[2]*np.sin(X)

    return estimated_output 


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

    default_degree = 2

  #for de la 0 la 100 
      #shuffle chunks
      #mean sqare error in the end
    def shuffle(x, y):
        sh = np.random.permutation(len(x))
        return x[sh], y[sh]  
        
    for j in range(0, 20, 5):
        xs_train, xs_test, ys_train, ys_test = test_data(chunk_xs, chunk_ys, j)
        y_hat = calculate_line(xs_train, ys_train, "linear", default_degree)
        error_linear += square_error(ys_test, y_hat[j:j + 5])
        
        #Calculez squared error pt fiecare grad de polynom
        initial_error_polynomial = 99999

        y_hat = calculate_line(xs_train, ys_train, "polynomial", default_degree)
        error_polynomial += square_error(ys_test, y_hat[j:j + 5])

        while(error_polynomial < initial_error_polynomial):
              initial_error_polynomial = error_polynomial
              default_degree += 1
              error_polynomial += square_error(ys_test, y_hat[j:j + 5])
              y_hat = calculate_line(xs_train, ys_train, "polynomial", default_degree)

        y_hat = calculate_line(xs_train, ys_train, "unknown", default_degree)
        error_unknown += square_error(ys_test, y_hat[j:j + 5])


    function_type = find_function(error_linear, error_polynomial, error_unknown) 

    if (function_type == "polynomial"):
        print(default_degree)   

    y_hat = calculate_line(chunk_xs, chunk_ys, function_type, default_degree)
    reconstruction_error += square_error(chunk_ys, y_hat)

    if sys.argv.__contains__("--plot"):
        plot_line(chunk_xs, y_hat)


reconstruction_error = reconstruction_error / number_of_segments_to_be_ploted
print(reconstruction_error)

if sys.argv.__contains__("--plot"):
    colour = np.concatenate([[i] * 20 for i in range(number_of_segments_to_be_ploted)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c = colour)
    plt.show()