import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading the csv file into a pandas DataFrame
data = pd.read_csv('student_data.csv') # data的数据结构为pandas DataFrame

# Printing out the first 10 rows of our data
# print(data[:10])

# Function to help us plot
def plot_points(data):
    X = np.array(data[["gre", "gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y == 1)]
    rejected = X[np.argwhere(y == 0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s=25, color='red', edgecolor='k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s=25, color='cyan', edgecolor='k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')


# Plotting the points
# plot_points(data)
# plt.show()

# 将第三个维度，rank也考虑在内
# data_rank1 = data[data["rank"]==1]
# data_rank2 = data[data["rank"]==2]
# data_rank3 = data[data["rank"]==3]
# data_rank4 = data[data["rank"]==4]
#
# # Plotting the graphs
# plot_points(data_rank1)
# plt.title("Rank 1")
# plt.show()
# plot_points(data_rank2)
# plt.title("Rank 2")
# plt.show()
# plot_points(data_rank3)
# plt.title("Rank 3")
# plt.show()
# plot_points(data_rank4)
# plt.title("Rank 4")
# plt.show()

# get_dummies todo
# 通过pd.get_dummies对直接对数据实现one-hot encode
one_hot_data = pd.get_dummies(data, columns=['rank'])
# print(one_hot_data[:10])

# alt way:
# 先对rank列实现one-hot encode，然后再把之前的rank列删掉
# one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)
#
# # Drop the previous rank column
# one_hot_data = one_hot_data.drop('rank', axis=1)
#
# # Print the first 10 rows of our data
# print(one_hot_data[:10])

# 将数据规范，Scaling the data，都处理在[0,1]中
processed_data = one_hot_data[:]
processed_data['gpa'] = processed_data['gpa']/4.0
processed_data['gre'] = processed_data['gre']/800


# 将数据分为测试集和训练集
sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

# print("Number of training samples is", len(train_data))
# print("Number of testing samples is", len(test_data))
# print(train_data[:10])
# print(test_data[:10])

# 将数据分成features 和 targets （即X，Y）
features = train_data.drop('admit', axis=1)
targets = train_data['admit']
features_test = test_data.drop('admit', axis=1)
targets_test = test_data['admit']

# 训练一个2层的NN

# Activation (sigmoid) function
# 先列出一些辅助函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

# Write the error term formula
def error_term_formula(x, y, output):
    return (y - output)*sigmoid_prime(x)


# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5


# Training function
def train_nn(features, targets, epochs, learnrate):
    # Use to same seed to make debugging easier
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights

    # normal todo 随机取样
    weights = np.random.normal(scale=1 / n_features ** .5, size=n_features)

    for e in range(epochs):

        del_w = np.zeros(weights.shape)
        # 对于features，从dataframe到ndarray，需要用x.values,方便迭代
        for x, y in zip(features.values, targets):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here
            #   rather than storing h as a separate variable
            output = sigmoid(np.dot(x, weights))

            # The error, the target minus the network output
            error = error_formula(y, output)

            # The error term
            error_term = error_term_formula(x, y, output)

            # The gradient descent step, the error times the gradient times the inputs
            del_w += error_term * x

        # Update the weights here. The learning rate times the
        # change in weights, divided by the number of records to average
        weights += learnrate * del_w / n_records

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean((out - targets) ** 2)
            print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights


weights = train_nn(features, targets, epochs, learnrate)

test_out = sigmoid(np.dot(features_test, weights))

# 返回true、false的 ndarray
predictions = test_out > 0.5

# 计算符合条件的个数并除以总数
accuracy = np.mean(predictions == targets_test)

print("Prediction accuracy: {:.3f}".format(accuracy))