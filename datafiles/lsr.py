import os
import sys
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from utilities import load_points_from_file as lpf
from utilities import view_data_segments as vds

global number_of_segments_to_be_ploted, ax, xs, ys, cross_validation_x, cross_validation_y

"""
def create_path(selectedFile): 
    #universalPath = "C:/Users/UserDell/Documents/University/Data Driven CW/coursework/datafiles/train_data/"  
    
    universalPath = os.path.abspath("datafiles/train_data/") 
    universalPath += selectedFile
    return universalPath
    
    return os.path.abspath(selectedFile)
"""    


#xs, ys = lpf(os.path.abspath("basic_1.csv"))  

def create_path(selectedFile): 
    universalPath = "C:/Users/UserDell/Documents/University/Data Driven CW/coursework/datafiles/train_data/"   
    universalPath += selectedFile
    return universalPath
    
xs, ys = lpf(create_path("noise_3.csv"))  

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

def least_squares(x, y):
    wh = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return wh

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
    """
    switch (i) {
        case 1: colour = 'r'; break;
        case 2: colour = 'g'; break;
        case 3: colour = 'c'; break;
    }
    """
    chunk_xs, chunk_ys = create_chunks_of_data(i)
    x = chebx(chunk_xs, 4)
    wh = least_squares(x, chunk_ys)
    ax.plot(chunk_xs, chebx(chunk_xs, 4).dot(wh), 'r', lw = 4, label="fitted line")  
    ax.legend()

vds(xs, ys)     

def square_error(true_y, estimated_y):
    return np.sum((y - y_hat) ** 2)

def shuffle_chunks(array):
        random.shuffle(array)
        return array    

def cross_validation():
    chunk_data = np.linspace(0, 20, 20)
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
    cross_validation_x = np.vstack(cross_validation_x, chunk_data_random_x[5:10])
    cross_validation_x = np.vstack(cross_validation_x, chunk_data_random_x[10:15])
    cross_validation_x = np.vstack(cross_validation_x, chunk_data_random_x[15:])

    cross_validation_y = chunk_data_random_y[:5]
    cross_validation_y = np.vstack(cross_validation_y, chunk_data_random_y[5:10])
    cross_validation_y = np.vstack(cross_validation_y, chunk_data_random_y[10:15])
    cross_validation_y = np.vstack(cross_validation_y, chunk_data_random_y[15:])

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