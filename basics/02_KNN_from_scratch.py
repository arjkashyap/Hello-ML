"""
K Nearest Neighbors from scratch.

KNN is a supervised classfication algorithm which makes use of 
Euclidean distances. It plots dataset points on a N dimmensional plane and
looks for the relation between each point by calculating the Euclidian dist.

For feature set which we want to predict, we calculate K neighrest points from it and 
label it according to majority vote among k neighbors.

"""

class KNearestNeighbors:

	def __init__(self, k=3):
		self.k=k