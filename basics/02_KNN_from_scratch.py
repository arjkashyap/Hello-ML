"""
K Nearest Neighbors from scratch.

KNN is a supervised classfication algorithm which makes use of 
Euclidean distances. It plots dataset points on a N dimmensional plane and
looks for the relation between each point by calculating the Euclidian dist.

For feature set which we want to predict, we calculate K neighrest points from it and 
label it according to majority vote among k neighbors.

"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
from math import sqrt
from collections import Counter

train_data = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}

class KNearestNeighbors:

	def __init__(self, k=3):

		"""
		Constructor takes the number of k nearest voters 
		"""
		self.k=k
		self.dim = 0					# Number of dimensions / attributes
		self.data_points = []           # 2d List containing an array of all data points and their label
		self.distances = []

	def fit(self, data, plot=False):
		"""
		Fit k Nearest algorithm for the dataset.
		Data set is a dictionary where the key corosponds to the label 
		value represents a matrix of co - ordinates in that group
		"""
		for group in data:
			self.data_points.append([data[group], group])
			self.dim = len(data[group][0])
		
		if plot and self.dim < 3:
			for label in data:
				group_points = data[label]
				xs = [ group_points[x][0] for x in range(len(group_points))]
				ys = [ group_points[x][1] for x in range(len(group_points))]				
				plt.scatter(xs, ys)
			plt.plot()
			plt.show()

		elif plot and self.dim >= 3:
			warnings.warn(" Unable to plot datasets with more than 2 features.")
			pass


	def predict(self, predict):
		"""
		Function takes a data set input and predicts its class 
		based on k nearest algorithm
		"""
		for group in self.data_points:
			points = group[0]
			label = group[1]
			for point in points:
				euclidean_distance = np.linalg.norm(np.array(point)-np.array(predict))
				self.distances.append([euclidean_distance, label])

		print(list(sorted(self.distances)))
		votes = [ dist[1] for dist in  sorted(self.distances)[:self.k] ]
		vote_result = Counter(votes).most_common(1)[0][0]
		return vote_result


def main():
	new_features = [5,7]
	clf = KNearestNeighbors(3)
	clf.fit(train_data)
	print(clf.predict(new_features))

if __name__ == '__main__':
	main()

