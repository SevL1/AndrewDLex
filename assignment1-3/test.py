import model
import numpy as np

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class，分为两类，每类是红或蓝
    D = 2 # dimensionality二维
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower 花的最大值
    #linspace以指定的时间间隔返回均匀间隔的数字。
    for j in range(2):
        ix = range(N*j,N*(j+1))#ix=（0，199）（200，399）
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta角度，产生200个角度并加入随机数，保证角度随机分开，图像开起来稀疏程度不一
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius半径，4sin(4*t),并加入一定的随机，图像轨道不平滑
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)] #生成坐标点
        Y[ix] = j #red or blue

    X = X.T
    Y = Y.T

    return X, Y

X, Y = load_planar_dataset()
sizes = [2, 4, 1]
network = model.Network(sizes)
final_biases, final_weights, cost_data = network.batch_gradient_descent(X, Y, 0.9, 20000)

def predict(weights, biases, X):
	a = X
	for b, w in zip(biases[:-1], weights[:-1]):
		z = np.dot(w, a) + b
		a = model.tanh(z)
	z = np.dot(weights[-1], a) + biases[-1]
	a = model.sigmoid(z)
	return a > 0.5

a = predict(final_weights, final_biases, X)
rights = np.sum(np.multiply(a, Y) + np.multiply((1 - a), (1 - Y)))
accuracy = rights / X.shape[1] * 100
print(cost_data[-20:])
print("Accuracy: %f%%" % accuracy)

