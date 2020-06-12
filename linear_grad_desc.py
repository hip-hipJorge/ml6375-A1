import numpy as np
LEARNING_RATE = 0.001


def optimize_weight(weight, training_data, y):
    # weights of data set and intercept
    w0 = weight[0]
    w1 = weight[1]
    w2 = weight[2]
    w3 = weight[3]
    w4 = weight[4]
    w5 = weight[5]
    b = weight[6]

    # initialize weight steps
    w0_step = 0
    w1_step = 0
    w2_step = 0
    w3_step = 0
    w4_step = 0
    w5_step = 0
    b_step = 0

    # number of instances, n
    n = len(training_data)

    # iterate through ith row
    for i in range(n):
        # read in ith row
        x0 = training_data[i][0]
        x1 = training_data[i][1]
        x2 = training_data[i][2]
        x3 = training_data[i][3]
        x4 = training_data[i][4]
        x5 = training_data[i][5]

        # calculate hypothesis
        h = w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + b

        # calculate step size
        w0_step += round(- (2/n) * x0 * (y[i][0] - h), 4)
        w1_step += round(- (2/n) * x1 * (y[i][0] - h), 4)
        w2_step += round(- (2/n) * x2 * (y[i][0] - h), 4)
        w3_step += round(- (2/n) * x3 * (y[i][0] - h), 4)
        w4_step += round(- (2/n) * x4 * (y[i][0] - h), 4)
        w5_step += round(- (2/n) * x5 * (y[i][0] - h), 4)
        b_step += round(- (2/n) * (y[i][0] - h), 4)

        # calculate new weight
        w0 = round(w0 - (LEARNING_RATE * w0_step), 4)
        w1 = round(w1 - LEARNING_RATE * w1_step, 4)
        w2 = round(w2 - LEARNING_RATE * w2_step, 4)
        w3 = round(w3 - LEARNING_RATE * w3_step, 4)
        w4 = round(w4 - LEARNING_RATE * w4_step, 4)
        w5 = round(w5 - LEARNING_RATE * w5_step, 4)
        b = round(b - LEARNING_RATE * b_step, 4)
    new_weight = [w0, w1, w2, w3, w4, w5, b]
    print(new_weight)
    return new_weight


def calc_error(weight, data_set, y):
    # weights of data set and intercept
    w0 = weight[0]
    w1 = weight[1]
    w2 = weight[2]
    w3 = weight[3]
    w4 = weight[4]
    w5 = weight[5]
    b = weight[6]

    # initialize data
    x0 = data_set[0]
    x1 = data_set[1]
    x2 = data_set[2]
    x3 = data_set[3]
    x4 = data_set[4]
    x5 = data_set[5]

    # calculate hypothesis
    h = w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + b

    return h - y


def gradient_descent(training_data, iterations, y):
    opt_weight = np.array([1] * 7)
    for i in range(iterations):
        opt_weight = optimize_weight(opt_weight, training_data, y)
    return opt_weight


def list_format(lst, dict):
    for i in range(len(lst)):
        lst[i] = dict[lst[i]]
    return lst

