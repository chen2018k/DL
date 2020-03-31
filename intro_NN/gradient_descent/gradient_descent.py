import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#绘图函数

def plot_points(X, y):
    # 把所有y==1的在y中索引找出来，用以讲X中对应的数据点输出
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]

    # plt.scatter用法
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')

def display(m, b, color='g--'):
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)

data = pd.read_csv('data.csv', header=None)
X = np.array(data[[0,1]])
y = np.array(data[2])
# plot_points(X,y)
# plt.show()

# 运算函数

# Activation (sigmoid) function 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Output (prediction) formula
def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)

# Error (log-loss) formula
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

# Gradient descent step
def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)
    d_error = y - output
    weights += learnrate * d_error * x
    bias += learnrate * d_error
    return weights, bias


# 训练过程

np.random.seed(44)

epochs = 100
learnrate = 0.01


def train(features, targets, epochs, learnrate, graph_lines=False):

    # 初始化
    errors = []
    last_loss = None
    bias = 0

    # 获得Features数量，X.shape 第一项返回行数，第二项返回列数
    n_records, n_features = features.shape

    # np.random.normal todo
    weights = np.random.normal(scale=1 / n_features ** .5, size=n_features)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)

        # zip todo
        for x, y in zip(features, targets):
            # 计算单层感知器的输出
            output = output_formula(x, weights, bias)
            # 计算其CE误差
            error = error_formula(y, output)
            # 对所有数据点做一次权重更新
            weights, bias = update_weights(x, y, weights, bias, learnrate)


        # Printing out the log-loss error on the training set
        # 输出单个感知器的结果
        out = output_formula(features, weights, bias)
        # 计算误差
        loss = np.mean(error_formula(targets, out))
        errors.append(loss)

        # 如果epochs为100的话，则每10次执行一次操作
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e, "==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            predictions = out > 0.5
            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
        if graph_lines and e % (epochs / 100) == 0:

            # display todo
            display(-weights[0] / weights[1], -bias / weights[1])

    # Plotting the solution boundary
    plt.title("Solution boundary")
    display(-weights[0] / weights[1], -bias / weights[1], 'black')

    # Plotting the data
    plot_points(features, targets)
    plt.show()

    # Plotting the error
    plt.title("Error Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.show()

train(X, y, epochs, learnrate, )