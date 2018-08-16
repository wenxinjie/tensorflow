import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

x = np.linspace(-7, 7, 180)

# def sigmoid(inputs):
# 	y = [1 / float(1 + np.exp(-x)) for x in inputs]
# 	return y

# def relu(inputs):
# 	y = [x * (x > 0) for x in inputs]
# 	return y

# def tanh(inputs):
# 	y = [ (np.exp(x) - np.exp(-x)) / float(np.exp(x) + np.exp(-x)) for x in inputs]
# 	return y

# def softplus(inputs):
# 	y = [np.log(1 + np.exp(x)) for x in inputs]
# 	return y

y_sigmoid = tf.nn.sigmoid(x)
y_relu = tf.nn.relu(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)

sess = tf.Session()

y_sigmoid, y_relu, y_tanh, y_softplus = sess.run([y_sigmoid, y_relu, y_tanh, y_softplus])

plt.figure(num = 1, figsize =  (8, 6))

plt.subplot(221)
plt.plot(x, y_sigmoid, "red", label = "sigmoid")
plt.ylim(-0.2, 1.2)
plt.legend(loc = "best")
plt.subplot(222)
plt.plot(x, y_relu, "red", label = "relu")
plt.ylim(-1, 6)
plt.legend(loc = "best")

plt.subplot(223)
plt.plot(x, y_tanh, "red", label = "tanh")
plt.ylim(-1.3, 1.3)
plt.legend(loc = "best")

plt.subplot(224)
plt.plot(x, y_softplus, "red", label = "softplus")
plt.ylim(-1, 6)
plt.legend(loc = "best")

plt.show()

sess.close()