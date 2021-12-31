import math
import time
import random
import sys
import matplotlib.pyplot as plt
import numpy as np 

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice


class Point(object):
	"""
	This is an object representation of the vector data set.
	"""
	def __init__(self, location):
		self.visited = False
		self.noise = False
		self.incluster = False
		self.location = location



def init_random_data(number_of_point ,demntion):
	#init random data with gaussion distrbotion 
	points = np.random.normal(0 , 1 ,size = (number_of_point , demntion))
	return points

def import_data(file):
	"""
	 This function imports the data into a list form a file name passed as an argument. 
	 This is a list of objects
	 The file should only the data seperated by a space.(or change the delimiter as required in split)
	"""
	data = []
	f = open(str(file), 'r')
	for line in f:
		current = line.split(',')	#enter your own delimiter like ","
		for j in range(0,len(current)):
			current[j] = int(current[j])
		data.append(Point(current))
	return data

def init_data_from_array(points): 
	data = []
	for point in points:
		data.append(Point(point))
	return data 

def print_data_matrix(data):
	for i in data:
		print (i.location)

def distance(point1, point2):
	#return np.linalg.norm(point1.location - point2.location)	
	list1 = point1.location
	list2 = point2.location
	distance = 0
	for i in range(0,len(list1)):
		distance += abs(list1[i] - list2[i]) ** 2
	return math.sqrt(distance)
	
def calaculate_distance_matrix(data):
	distance_matrix =[]
	for i in range(0,len(data)):
		current = []
		for j in range(0,len(data)):
			current.append(distance(data[i], data[j]))
		distance_matrix.append(current)
	return distance_matrix

def regional_query(P, data , distance_matrix , epsilon):
	neighbour = []
	for i in range(0,len(data)):
		if data[i] == P:
			for j in range(0,len(data)):
				if distance_matrix[i][j] < epsilon:
					neighbour.append(data[j])
			break
	return neighbour

def expand_cluster(P, neighbor_pts, Cluster, epsilon, MinPts, data, distance_matrix):
	Cluster.append(P)
	P.incluster = True
	for P_neigh in neighbor_pts:
		if P_neigh.visited != True:
			P_neigh.visited = True
			neighbor_pts_in = regional_query(P_neigh, data , distance_matrix , epsilon)
			if len(neighbor_pts_in) >= MinPts:
				neighbor_pts = neighbor_pts_in + neighbor_pts
		if P_neigh.incluster != True:
			Cluster.append(P_neigh)


def dbscan(data, epsilon, MinPts):
	C = []
	distance_matrix = calaculate_distance_matrix(data)
	for P in data:
		#print P.location

		if P.visited == True:	
			continue
		P.visited = True
		neighbor_pts = regional_query(P, data, distance_matrix, epsilon)
		#print neighbor_pts
		if len(neighbor_pts) < MinPts:
			P.noise = True
		else:
			C.append([])
			expand_cluster(P, neighbor_pts, C[-1], epsilon, MinPts, data, distance_matrix)
	return C

def color(cluster_number):
	colors = []
	for i in range(0,cluster_number):
		colors.append("#%06x" % random.randint(0,0xFFFFFF))
	return colors

def graphic(final , title ='' , img_name = 'result'):
	colors = color(len(final))
	plt.ion()
	plt.figure()
	i = 0
	for cluster in final:
		dum = []
		for point in cluster:
			dum.append(point.location)
		x_ = [x for [x,y] in dum]
		y_ = [y for [x,y] in dum]
		plt.plot(x_, y_ , colors[i] , marker='o', ls='')
		i += 1
	plt.title(title)
	plt.gca().set_aspect('equal', adjustable='box')
	plt.axis('equal')
	plt.savefig(f'{img_name}.png')


def print_cluster(final):
	for i in range(0,len(final)):
		print ("cluster ", i+1)
		print_data_matrix(final[i])





def run_test(data , cases , img_name):
	index = 1
	for case in cases:
		data_copy = data.copy()
		data_copy = init_data_from_array(data_copy)
		final = dbscan(data_copy, case["epsilon"] , case["MinPts"])
		graphic(final , f'1500 points, esp = {case["epsilon"]} , min point = {case["MinPts"]} ' , img_name + str(index))
		index = index + 1 

if __name__ == '__main__':
	#data = import_data(sys.argv[1])
	# ============
	# Generate datasets. We choose the size big enough to see the scalability
	# of the algorithms, but not too big to avoid too long running times
	# ============
	np.random.seed(0)
	n_samples = 1500
	#circl data
	(noisy_circles , _) = datasets.make_circles(n_samples=n_samples,factor = 0 , noise=0.05)
	#half moon
	(noisy_moons , _) = datasets.make_moons(n_samples=n_samples, noise=0.05)
	#bloobs
	(blobs , _) = datasets.make_blobs(n_samples=n_samples, random_state=8)
	#unifom distrubiton 
	no_structure = np.random.rand(n_samples, 2)
	#gaussion distrbiton
	gauss = init_random_data(n_samples , 2)
	#for each of the data set we will test the result of:
		# epsilon = 0.3 minimum point = 20
		# epsilon = 0.18 minimum point = 5
		# epsilon = 0.15 minimum point = 20 
	
	cases = [{"epsilon" : 0.3 , "MinPts" : 20 } ,{"epsilon" : 0.18 , "MinPts" : 5 } , {"epsilon" : 0.15 , "MinPts" : 20 } ]

	#run_test(noisy_circles , cases , 'noisy_circles')
	#run_test(blobs ,cases , 'blobs')
	run_test(noisy_moons ,cases , 'noisy_moons')
	run_test( blobs ,cases ,'blobs')
	run_test(no_structure ,cases ,'uniform_disrubiton')
	run_test(gauss ,cases ,'gaussian distrbutin')
	
	


	


	