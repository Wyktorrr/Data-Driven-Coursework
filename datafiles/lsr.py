import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utilities import load_points_from_file as lpf
from utilities import view_data_segments as vds

global number_of_segments_to_be_ploted, ax, xs, ys

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
    
xs, ys = lpf(create_path("basic_1.csv"))  

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