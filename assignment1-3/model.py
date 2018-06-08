import numpy as np 

class Network(object):

	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	def batch_gradient_descent(self, X, Y, eta, epochs): 
		np.random.seed(3)
		cost_data = [self.cost(X, Y)]
		for _ in range(epochs):
			dB, dW = self.gradient(X, Y)
			self.biases = self.biases - eta * np.array(dB)
			self.weights = self.weights - eta * np.array(dW)
			cost_data.append(self.cost(X, Y))
		return self.biases, self.weights, cost_data


	def cost(self, X, Y):
		m = X.shape[1]
		zs, activations = self.feedforward(X)
		a = activations[-1]
		inner = -(np.multiply(Y, np.log(a)) + np.multiply((1 - Y), np.log(1 - a)))
		return np.sum(inner) / m
		

	def feedforward(self, X):
		a = X
		zs = [a]
		activations = [a]
		for b, w in zip(self.biases[:-1], self.weights[:-1]):
			z = np.dot(w, a) + b
			a = tanh(z)
			zs.append(z)
			activations.append(a)
		z = np.dot(self.weights[-1], a) + self.biases[-1]
		a = sigmoid(z)
		zs.append(z)
		activations.append(a)
		return zs, activations



	def gradient(self, X, Y):
		m = X.shape[1]
		zs, activations = self.feedforward(X)
		dz2 = activations[2] - Y
		dw2 = np.dot(dz2, activations[1].T) / m
		db2 = np.sum(dz2, axis = 1, keepdims = True) / m
		dz1 = np.dot(self.weights[1].T, dz2) * tanh_prime(zs[1])
		dw1 = np.dot(dz1, (X.T)) / m
		db1 = np.sum(dz1, axis = 1, keepdims = True) / m
		dB = [db1, db2]
		dW = [dw1, dw2]
		return dB, dW


def tanh(z):
	a = np.exp(z)
	b = np.exp(-z)
	return (a - b) / (a + b)

def sigmoid(z):
	return (1 / (1 + np.exp(-z)))

def tanh_prime(z):
	return 1 - tanh(z)**2


"""
X, Y: matrix(numpy.ndarray)
      X: np.array([|, ..., |, |])
      Y: np.array([[a1, ..., a2, a3]])
weights, biases, dB, dW: list
	  weights, dW: [w1, w2, ..., wL](神经网络共L层)
	                wi: numpy.ndarray, j×i
	  biases, dB: [b1, b2, ..., bL]
			       b1: numpy.ndarray, j×1
activations: [X, a1, a2, ..., aL], list
			  ai: numpy.ndarray, j×1
"""