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
ax.set_xlabel("$x_lambda$")
ax.set_ylabel("$y_lambda$")
ax.scatter(xs, ys)

number_of_input_points = len(xs) 
number_of_segments_to_be_ploted = number_of_input_points // 20

#Equations for linear regresion aiming to minimize the sum squared error
#Determine the line of best fit for each set of data       
def plot_linear(X, Y):
    #Return the gradient and the vertical offset of a linear function
    ones = np.ones(X.shape)
    x_e = np.column_stack((ones, X))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(Y)
    return v[:]  

def plot_polynomial(X, Y, degree):
    #Returns estimates of the coefficients for the polynomial with respect to its order.
    estimated_parameters = []
    extended_matrix = np.column_stack((np.ones(X.shape), X))
    for i in range(2, degree + 1):
        extended_matrix = np.column_stack((extended_matrix, X**i))
        estimated_parameters = np.linalg.inv(extended_matrix.T.dot(extended_matrix)).dot(extended_matrix.T).dot(Y)
    return estimated_parameters

def unknown_function_sin(X, Y):
    #Returns estimates of the parameters for sinus function.
     extended_matrix = np.column_stack((np.ones(X.shape),np.sin(X)))
     estimated_parameters = np.linalg.inv(extended_matrix.T.dot(extended_matrix)).dot(extended_matrix.T).dot(Y)
     return estimated_parameters

def unknown_function_cos(X, Y):
    #Returns estimates of the parameters for cosinus function. 
     extended_matrix = np.column_stack((np.ones(X.shape),np.cos(X)))
     estimated_parameters = np.linalg.inv(extended_matrix.T.dot(extended_matrix)).dot(extended_matrix.T).dot(Y)
     return estimated_parameters   

def unknown_function_exp(X, Y):
    #Returns estimates of the parameters for exponential function.
     extended_matrix = np.column_stack((np.ones(X.shape), np.exp(X)))
     estimated_parameters = np.linalg.inv(extended_matrix.T.dot(extended_matrix)).dot(extended_matrix.T).dot(Y)   
     return estimated_parameters  

#Compute the estimated output for polynomial given its input
def compute_polynomial(X, X_test, Y, degree):
    coefficients = []
    coefficients = plot_polynomial(X, Y, degree) 
    result = [0 for i in range(len(X_test))]
    for i in range(degree + 1):
        result += coefficients[i]*X_test**i  #Y_hat -> estimated_output
    return result    

#Based on the square error that has been computed, decide the type of the function.
#The vertical distance from the data points to the regression line needs to be minimized.
def find_function(linear_error, polynomial_error, unknown_error_sin, unknown_error_cos, unknown_error_exp):
    list_errors = [linear_error, polynomial_error, unknown_error_sin, unknown_error_cos]
    min_error = min(list_errors)
    if(min_error == linear_error):
        return "linear"
    elif(min_error == polynomial_error):
        return "polynomial"
    elif(min_error == unknown_error_sin):    
           return "sinus" 
    elif(min_error == unknown_error_exp):
           return "exponential"         
    return "cosinus"           

#Compute output (the fitted line) based on the function type.
#I use a limited sample of testing points in order to estimate how the model is expected to behave when making 
#predictions on data that does not serve as training purpouses. 
#I evaluate the model on the test set and fit it on the training set.
def calculate_line(X, X_test, Y, function_type, degree):
    estimated_output = []
    if function_type == "polynomial":
            estimated_output = compute_polynomial(X, X_test, Y, degree)

    elif function_type == "linear":
        offset, slope = plot_linear(X, Y)
        estimated_output = slope * X_test + offset

    elif function_type == "sinus":
        parameters = unknown_function_sin(X, Y)
        estimated_output = parameters[0] + parameters[1]*np.sin(X_test)

    elif function_type == "cosinus":
         parameters = unknown_function_cos(X, Y)
         estimated_output = parameters[0] + parameters[1]*np.cos(X_test)

    else:
          parameters = unknown_function_exp(X, Y)
          estimated_output = parameters[0] + parameters[1]*np.exp(X_test)

    return estimated_output 


def plot_line(x, estimated_output):
    plt.plot(x, estimated_output, c = "r")           

reconstruction_error = 0

def shuffle(x, y):
    #This function keeps an association between the input and output points and randomly permutates them
    sh = np.random.permutation(len(x))
    return x[sh], y[sh]  

#In order to estimate the skill of the model, I acquired test and data points for k-fold validation. 
#I considered 5 points of test and 15 of train, my choice being an implementation of a 5-fold cross validation.
#The parameter j is increasing by 5 at each iteration for each portion of 20 points (each function).
#As a consequence, for the first iteration, the first 5 points are test ponts and the last 15 train points, 
#for the second iteration, points starting from the fifth up the tenth will be testing and the others will be train points and so on.
#Therefore, the error for each type of function and for each degree for the polynomial is computed and summed four times.
#This is why I divide each error by four before deciding what function describes each chunk of data.   
def test_train_data(X, Y, j):
    xs_test = X[j:j+5]
    ys_test = Y[j:j+5]
    xs_train = X[j+5:20]
    ys_train = Y[j+5:20]
    xs_train = np.append(X[:j], xs_train)
    ys_train = np.append(Y[:j], ys_train)

    return xs_train, xs_test, ys_train, ys_test 

#I have also tested the model using 10 points as test and 10 points for train. A version of 10-fold cross validation.
#The values for the total reconstruction error were very similar. 
#I fitted the line using 10-fold cross validation as well to check if a bias-variance tradeoff discussion would be appropiate.
"""
def test_train_data(X, Y, j):
    xs_test = X[j:j+10]
    ys_test = Y[j:j+10]
    xs_train = X[j+10:20]
    ys_train = Y[j+10:20]
    xs_train = np.append(X[:j], xs_train)
    ys_train = np.append(Y[:j], ys_train)

    return xs_train, xs_test, ys_train, ys_test 
"""    


#Sum of offsets / residuals from the plotted curve that need to be minimized. These are minimized thanks to the least squares method. 
def square_error(y, y_hat):
    return np.sum((y - y_hat) ** 2)

#Split data in chanks of 20 points to delimit each function from the input data
for i in range (0, number_of_input_points, 20):
    chunk_xs = xs[i:i+20]
    chunk_ys = ys[i:i+20]

    #Shuffle the dataset randomly in order to avoid choosing the points for test and respectively training in a biased manner.
    shuffle_chunk_xs, shuffle_chunk_ys = shuffle(chunk_xs, chunk_ys) 

    #Initialize errors for each type of function 
    error_linear = 0
    error_polynomial = 0
    error_unknown_sin = 0
    error_unknown_cos = 0
    error_unknown_exp = 0

    #Consider the order of the polynomial to be at least 2. 
    default_degree = 2

    #Initialize list of errors for each order of the polynomial
    list_degree = np.zeros(7) 

    #Because I shaffle the x and y coordinates for the k-fold validation, the errors will end up being slightly different.
    #In order to mitigate this, I compute the estimated outputs and keep on adding the errors multiple times,
    #and, in the end, before chosing the suitable function for the 20 points data set, I take the mean of these errors. 
    #The reason for this approach is to compensate for the randomness.
    for i in range(0, 100):  
        for j in range(0, 20, 5):
            
            #Acquire test and train data for each data set
            xs_train, xs_test, ys_train, ys_test = test_train_data(shuffle_chunk_xs, shuffle_chunk_ys, j)
        
            #Fit the lines based on the estimated values for the parameters for each function type

            y_hat = calculate_line(xs_train, xs_test, ys_train, "linear", default_degree)
            error_linear += square_error(ys_test, y_hat)
        
            #Create a list of squared errors for each degree of the polynomial 
            for i in range(2, 7):
                y_hat = calculate_line(xs_train, xs_test, ys_train, "polynomial", i)
                error_polynomial = square_error(ys_test, y_hat)
                list_degree[i] += error_polynomial

            y_hat = calculate_line(xs_train, xs_test, ys_train, "sinus", default_degree)
            error_unknown_sin += square_error(ys_test, y_hat)

            y_hat = calculate_line(xs_train, xs_test, ys_train, "cosinus", default_degree)
            error_unknown_cos += square_error(ys_test, y_hat)

            y_hat = calculate_line(xs_train, xs_test, ys_train, "exponential", default_degree)
            error_unknown_exp += square_error(ys_test, y_hat)


    error_linear = error_linear.mean() / 4
    error_linear = error_linear / 4
    error_unknown_sin = error_unknown_sin.mean() / 4
    error_unknown_cos = error_unknown_cos.mean() / 4
    error_unknown_exp = error_unknown_exp.mean() / 4
    for i in range(2, 7):
        list_degree[i] = list_degree[i].mean() / 4

    #Pick the suitable degree for the polynomial based on the error for each degree. (smallest error -> the suitable degree)
    index = 2
    for i in range(2, 7):
        if(list_degree[i] < list_degree[index]):
            index = i

    default_degree = index        
    
    print("LINEAR ERROR IS: ", error_linear)
    print()
    print("ERROR POLYNOMIAL: ", list_degree[index])
    print()
    print("ERROR SINUS: ", error_unknown_sin)
    print()
    print("ERROR COSINUS: ", error_unknown_cos)
    print()
    print("ERROR EXPONENTIAL:", error_unknown_exp)
    print()
    print("LIST OF ERRORS FOR EACH DEGREE FOR POLYNOMIAL: ", list_degree)
    print()

    #Based on the error decide the function that best fits the current chunk of data
    function_type = find_function(error_linear, list_degree[index], error_unknown_sin, error_unknown_cos, error_unknown_exp) 

    if (function_type == "polynomial"):
        print("Degree of the polynomial is: ", default_degree)  
        print() 
        print()
    
    #Fit the line according to the function type
    y_hat = calculate_line(chunk_xs, chunk_xs, chunk_ys, function_type, default_degree)
    reconstruction_error += square_error(chunk_ys, y_hat)

    #Plot the line for each chunk of data based on the predicted line.
    if sys.argv.__contains__("--plot"):
        plot_line(chunk_xs, y_hat)

print("Total reconstruction error is: ", reconstruction_error)

if sys.argv.__contains__("--plot"):
    colour = np.concatenate([[i] * 20 for i in range(number_of_segments_to_be_ploted)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c = colour)
    plt.show()