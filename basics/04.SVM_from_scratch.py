import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

"""
SVMfrom scratch in raw python.

Support vector machine is basically a convex optimization problem used
for binnary classification in machine learning (and sometimes for regression purposes too).
The whole point of the SVM is to draw a hyperplane between support vectors of
two classes such that tha margin is maximum.

SVM represents dataset on N-dimensional co-ordinates where n is the number of 
attributes in the feature set.

The whole point is to find the two unknowns namely vector
	 w: largest vector perpendicular to the hyperplane ,
	 b: bias.

"""

data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}


class SupportVectorMachine:

	def __init__(self, visualization=False):
		self.visualization = visualization
		self.colors = {1:'r', -1:'b'}
		self.data = []

	def fit(self, data):
		self.data = data
		opt_dict = {}

		

		pass

	def predict(self, features):
		# sign( x.w + b )
		classification = np.sign(np.dot(self.features, self.w) + self.b)

		return classification