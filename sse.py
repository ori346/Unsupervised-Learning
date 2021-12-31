
from dpmmpython.dpmmwrapper import DPMMPython
from dpmmpython.priors import niw
import numpy as np
from dpmeans import dp_means 
import matplotlib.pyplot as plt, random as rnd
from sklearn import datasets

def generate_data(n=100):
        mu_x = np.array((0, 5, 10, 15))
        mu_y = np.array((0, 5, 0, -5))
        data = np.empty(shape=(n, 2))
        classes = []
        for i in range(n):
            c = rnd.choice(range(4))
            data[i] = (np.random.normal(mu_x[c], 1), np.random.normal(mu_y[c], 1))
            classes.append(c)
        return data, classes

def rearnge_data(data):
    output = []
    for i in range(len(data[0])):
        output.append([data[0][i] , data[1][i]])
    return np.array(output)


def graphicTest(my_data ,our_res , k1 , predictions , k2 , title = 'res'):
    plt.subplot(221)
    plt.title(f"our means result\n The number of k is: {k1}")
    plt.scatter(my_data[:, 0], my_data[:, 1], c = our_res['assignments'] )
    plt.subplot(222)
    plt.title(f"Julia means result\n The number of k is: {k2}")
    plt.scatter(my_data[:, 0], my_data[:, 1], c= predictions )
    plt.show()
    print("this is debug message")
    #plt.savefig(title)

def run_test(N_sample , title):
    n_sample = N_sample
    max_iters = 100
    data,gt = DPMMPython.generate_gaussian_data(n_sample, 2, 10, 100.0)

    prior = niw(1,np.zeros(2),4,np.eye(2))
    labels,_,results= DPMMPython.fit(data,50,prior = prior,verbose = True, gt = gt , iterations = max_iters)
    my_data = rearnge_data(data)
    our_res = dp_means(my_data , 50)
    predictions = DPMMPython.predict(results[-1],data)
    graphicTest(my_data , our_res , our_res['k'] , predictions , max(predictions) , title)

for i in range ( 25 , 31):
    title = f"test number {i}.png"
    run_test(600 + (i* 200) , title)








