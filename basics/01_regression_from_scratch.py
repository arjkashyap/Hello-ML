"""
Linear Regression from scratch for 2D problem

The objective of Linear algebra in machine learning is to find the relation ship
between the points in the space.
Simple linear regression is used to find the best fit line of a dataset.

Line of an equation is given by: y = mx + c

To find the best fit line for N points in a space, we have to find the slope and the y intercept.
In this module we will try to plot a best fit line.
then we will calculate how good our best fit line is.

"""

from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt
import random


class LinearRegression:
	"""
	Linear regression class for single variable. 
	To initialize the object instance, pass the length of the features
	which you have or want to generate. 

	- You can generate random data after that by using the 
		LinearRegression.generate_data(plot_data=False) method.
			Pass True if you want to see the dataset plotted on the graph

	- Fit the dataset with regression model using
		LinearRegression.fit(X, y)
			Where X -> Feature set array
				  y -> Labels array
	
	"""
	random.seed(20)
	m = 0           # slope
	c = 0           # y intercept

	def __init__(self, n):
		self.n = n
		self.X = 0
		self.y = 0

	def genereate_data(self, plot_data=False):
		"""
		Function takes an arg n and returns two np arrays
		X -> Feature set of n co-ordinates
		y - > Labels for n features

		These are random dataset generated for testing
		"""
		x = [ random.random()*10 + x for x in range(self.n) ]
		y = [ random.random()*10  + x for x in range(self.n) ]

		xs = np.array(x, dtype=np.float64)
		ys = np.array(y, dtype=np.float64)

		if plot_data:
			plt.scatter(xs, ys)
			plt.show()

		return xs, ys

	def best_fit_mc(self, xs, ys):
		"""
		Function takes two args: 
		X -> Numpy array of Features
		y -> Numpy array of labels

		Returns the best fit slope and intercepts for the graph
		"""
		p = ((mean(xs)*mean(ys)) - mean(xs*ys))
		q = ((mean(xs)*mean(xs)) - mean(xs*xs))          # denomiator
		

		m = p/q
		c = mean(ys) - m*mean(xs)

		return m, c

	def fit(self, xs, ys):
		self.X = xs
		self.y = ys
		self.m, self.c = self.best_fit_mc(self.X, self.y)
		
		print(self.m, self.c)
		model_line = [ self.m*x + self.c for x in self.X ]
		plt.scatter(self.X,self.y,color='red')	
		plt.plot(self.X, model_line, color='#003F72')

		plt.show()


def main():
	model = LinearRegression(40)
	X, y = model.genereate_data()
	model.fit(X, y)


if __name__ == '__main__':
	main()