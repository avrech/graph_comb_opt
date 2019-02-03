import os
import numpy as np 
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets, linear_model
TESTSET = 'tsplib' #or 'tsp2d_clustered'
if __name__ == "__main__":
    with open(TESTSET + "_performance.txt") as f:
        lines = [line.rstrip('\n').split() for line in f]
    results = np.array([[float(num) for num in line[1:]] for line in lines])
    graph_size = results[:,0]
    aprx_ratio = results[:,1]
    time_ratio = results[:,2]
    
    # Aprroximation Ratio vs. graph size
    plt.figure(1)
    plt.xlabel("Graph Size [#nodes]")
    plt.ylabel("Approximation Ratio")
    plt.title("Approximation Ratio vs. Graph Size\nAverage = %.3f" % np.mean(aprx_ratio))
    plt.scatter(graph_size,aprx_ratio)
    X = graph_size.reshape(-1,1)
    y = aprx_ratio.reshape(-1,1)
    # Fit line using all data
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    # Predict data of estimated models
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y = lr.predict(line_X)
    lw = 2
    plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
    
    # Time Ratio vs. graph size:
    plt.figure(2)
    plt.xlabel("Graph Size [#nodes]")
    plt.ylabel("Time Ratio")
    plt.title("Execution Time Ratio vs. Graph Size\nAverage = %.2f" % np.mean(time_ratio))
    plt.scatter(graph_size,time_ratio) 
    y = time_ratio.reshape(-1,1)
    # Fit line using all data
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    # Predict data of estimated models
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y = lr.predict(line_X)
    lw = 2
    plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
    
    # 3D presentation
    c = ['b' if a<1 else 'r' for a in time_ratio]
    m = ['o' if t<1.1 else '^' for t in aprx_ratio]
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    [ax.scatter( time_ratio[ii], aprx_ratio[ii], graph_size[ii], c=c[ii], marker=m[ii]) for ii in range(len(c))]
    ax.set_zlabel('Graph Size')
    ax.set_ylabel('Approximation Ratio')
    ax.set_xlabel('Time Ratio')
    plt.show()
    weighted_aprx_ratio = sum(aprx_ratio * graph_size) / sum(graph_size)
    weighted_time_ratio = sum(time_ratio * graph_size) / sum(graph_size)

    print "approximation ratio: " , np.mean(aprx_ratio)
    print "execution time ratio: " , np.mean(time_ratio)
    print "weighted approximation ratio: ",  weighted_aprx_ratio
    print "weighted execution time ratio: ",  weighted_time_ratio
    
    # plt.figure(4)
    # X = graph_size.reshape(-1,1)
    # y = aprx_ratio.reshape(-1,1)
    # # Fit line using all data
    # lr = linear_model.LinearRegression()
    # lr.fit(X, y)

    # # Predict data of estimated models
    # line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    # line_y = lr.predict(line_X)
    # lw = 2
    # plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
    
    # plt.show()

    print "Goodbye!"
