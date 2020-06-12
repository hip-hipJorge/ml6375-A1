import numpy as np
LEARNING_RATE = 0.001


def optimize_weight(hs, weight, training_data, y):
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
        # read in hypothesis space
        h = w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + b
        hs[i] = h

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
    # return last weight
    new_weight = [w0, w1, w2, w3, w4, w5, b]
    return new_weight


def gradient_descent(hs, training_data, iterations, y):
    opt_weight = np.array([1] * 7)
    for i in range(iterations):
        opt_weight = optimize_weight(hs, opt_weight, training_data, y)
    return opt_weight


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


def calc_mse(hs, y):
    n = len(hs)
    error = np.empty(n)
    for i in range(n):
        error[i] = hs[i] - y[i]
    sum_of_error = sum(error)
    mse = (1/n) * (sum_of_error ** 2)
    return round(mse, 4)


def list_format(lst, attr):
    for i in range(len(lst)):
        lst[i] = attr[lst[i]]
    return lst


def prediction(hyp_weight, data_points):
    print(hyp_weight)
    print(data_points)
    h = hyp_weight[0]*data_points[0] + hyp_weight[1]*data_points[1]\
        + hyp_weight[2]*data_points[2] + hyp_weight[3]*data_points[3]\
        + hyp_weight[4]*data_points[4] + hyp_weight[5]*data_points[5]\
        + hyp_weight[6]
    return h
