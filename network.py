import os
import cv2
import random
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    f = 1 / (1 + np.exp(-x))
    return f * (1 - f)


# 以列表方式初始化网络结构
class Network:
    def __init__(self, net, learning_rate=0.5):
        self.weights = []
        self.biases = []
        self.delta = []
        self.learning_rate = learning_rate
        for i in range(1, len(net)):
            weight = np.random.normal(size=net[i - 1] * net[i]).reshape((net[i - 1], net[i]))
            bias = np.random.normal(size=net[i])
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, x):
        layers_output = []
        for i in range(len(self.weights)):
            x = sigmoid(np.dot(self.weights[i].T, x) + self.biases[i])
            layers_output.append(x)
        return layers_output

    def backpro(self, layers_output, y):
        delta = 1
        for i in reversed(range(len(layers_output))):
            layer_out = layers_output[i]
            if i == (len(layers_output) - 1):
                error = layer_out - y
                delta = delta * error * deriv_sigmoid(layer_out)
                self.delta.append(delta)
            else:
                error = np.dot((self.weights[i + 1]), delta)
                delta = deriv_sigmoid(layer_out) * error
                self.delta.append(delta)
        self.delta.reverse()

    def train(self, x, y):
        layers_output = self.forward(x)
        self.backpro(layers_output, y)
        for j in range(len(self.weights)):
            previous_layers_output = np.atleast_2d(x if j == 0 else layers_output[j - 1])
            self.weights[j] = self.weights[j] - self.delta[j] * previous_layers_output.T * self.learning_rate
            self.biases[j] = self.biases[j] - self.delta[j] * self.learning_rate


if __name__ == '__main__':

    train_data = []
    train_path = '/home/kv/Downloads/MINST图像/train'
    label_list = os.listdir(train_path)
    for i in range(len(label_list)):
        train_img_path = os.path.join(train_path, label_list[i])
        train_img_list = os.listdir(train_img_path)
        for j in range(len(train_img_list)):
            train_img = cv2.imread(os.path.join(train_img_path, train_img_list[j]), 0)
            train_img = np.reshape(train_img, (train_img.shape[0] * train_img.shape[1]))
            train_data.append([train_img, label_list[i]])
    random.shuffle(train_data)

    val_data = []
    val_path = '/home/kv/Downloads/MINST图像/test'
    label_list = os.listdir(val_path)
    for i in range(len(label_list)):
        val_img_path = os.path.join(val_path, label_list[i])
        val_img_list = os.listdir(val_img_path)
        for j in range(len(val_img_list)):
            val_img = cv2.imread(os.path.join(val_img_path, val_img_list[j]), 0)
            val_img = np.reshape(val_img, (val_img.shape[0] * val_img.shape[1]))
            val_data.append([val_img, label_list[i]])
    random.shuffle(val_data)

    network = Network([28 * 28, 50, 100, 10])
    for epoch in range(30):
        for i in range(len(train_data)):
            x = train_data[i][0]
            y = int(train_data[i][1])
            y = np.array(np.eye(10)[y], dtype=int)
            network.train(x, y)
            if (i + 1) % 100 == 0:
                print('epoch:', epoch + 1, 'train:', i + 1)
    num = 0
    for i in range(len(val_data)):
        x = val_data[i][0]
        y = int(val_data[i][1])
        output = network.forward(x)[-1]
        predict = np.argmax(output)
        if predict == y:
            num += 1
    print('acc:', num / len(val_data))
