from numpy import *

def dp_means(data, Lambda, max_iters=100, tolerance=10e-3):
    n = len(data)
    k = 1
    assignments = ones(n)
    mu = []
    for d in range(data.ndim):
        mu.append(array((mean(data[:, d]),)))
    is_converged = False
    n_iters = 0
    ss_old = float('inf')
    ss_new = float('inf')
    while not is_converged and n_iters < max_iters:
        n_iters += 1
        for i in range(n):
            distances = repeat(None, k)
            for j in range(k):
                distances[j] = sum((data[i, d] - mu[d][j]) ** 2  for d in range(data.ndim))
            if min(distances) > Lambda:
                k += 1
                assignments[i] = k-1
                for d in range(data.ndim):                    
                    mu[d] = append(mu[d], data[i, d])
            else:
                assignments[i] = argmin(distances)
        for j in range(k):
            if len(where(assignments == j)) > 0:
                for d in range(data.ndim):
                    mu[d][j] = mean(data[assignments == j, d])
        ss_new = 0      
        for i in range(n):
            ss_new <- ss_new + sum((data[i, d] - mu[d][int(assignments[i])]) ** 2 for d in range(data.ndim))
        ss_change = ss_old - ss_new
        is_converged = ss_change < tolerance  
    return {'centers': column_stack([mu[d] for d in range(data.ndim)]),
            'assignments': assignments,
            'k': k, 'n_iters': n_iters}


                        
if __name__ == '__main__':

    import matplotlib.pyplot as plt, random as rnd

    data, classes = generate_data(10)
    #results = dp_means(data, 50)
    #print 'found %d clusters' % len(set(results['assignments']))
    #plt.scatter(data[:,0], data[:,1], c=results['assignments'])
    #plt.show()