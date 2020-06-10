import array as arr
import numpy as np

LEARNING_RATE = 0.1

class PopLine:
    def __init__(self, data_value, intercept=0):
        self.pop_size = len(data_value)
        self.data_value = arr.array('d', data_value)
        self.weight = arr.array('d', [1] * self.pop_size)
        self.intercept = intercept
        # output (y) for each line
        self.output = self.calc_output()

    # getters
    def get_pop_size(self):
        return self.pop_size

    def get_data_value(self):

        return self.data_value

    def get_weight(self):
        return self.weight

    def get_output(self):
        return self.output

    # setters
    def set_data_value(self, data_value):
        self.data_value = data_value

    def set_weight(self, weight):
        self.weight = weight
        # new output
        self.output = self.calc_output()

    def set_intercept(self, intercept):
        self.intercept = intercept

    def set_output(self, output):
        self.output = output

    # helper functions
    def calc_output(self):
        self.output = arr.array('d', [0] * self.pop_size)
        for i in range(self.pop_size):
            self.output[i] = self.weight[i] * self.data_value[i] + self.intercept
        return self.output


class LinearRegression:
    def __init__(self, line, actual_data):
        self.error = self.calc_error(line.output, actual_data)
        self.mse = self.mean_squared_error(self.error)
        self.step_size = 0

    # getters
    def get_error(self):
        return self.error

    def get_mse(self):
        return self.mse

    def get_step_size(self):
        return self.step_size

    # setters
    def set_error(self, error):
        self.error = error

    def set_mse(self, mse):
        self.mse = mse

    def set_step_size(self, step_size):
        self.step_size = step_size

    # helper functions
    def calc_error(self, hypothesis, actual_value):
        error = arr.array('d', [0] * len(actual_value))
        for i in range(len(actual_value)):
            error[i] = actual_value[i] - hypothesis[i]
        return error

    def mean_squared_error(self, error):
        e_squared_sum = 0
        n = len(error)
        for i in range(n):
            e_squared_sum += pow(error[i], 2)
        return (1 / (2 * n)) * e_squared_sum

    def new_weight(self, old_weight, data_values, error):
        n = len(data_values)
        new_weight = arr.array('d', [0] * n)
        sum_error_and_data = 0
        for i in range(n):
            sum_error_and_data += error[i] * data_values[i]
        self.set_step_size(LEARNING_RATE * ((-1/n) * sum_error_and_data))
        for i in range(n):
            new_weight[i] = old_weight[i] - self.step_size
        return new_weight



# test 1 data point
w = [0.64, 0.64, 0.64]
x = [0.5, 2.3, 2.9]
y = [1.4, 1.9, 3.2]

pl = PopLine(x)
pl.set_weight(w)
lin_reg = LinearRegression(pl, y)

print("actual:")
print(y)
print("output:")
print(*pl.output)
print("error:")
print(*lin_reg.error)
print("mse:")
print(lin_reg.mean_squared_error(lin_reg.get_error()))

print()

for i in range(10):
    print()
    print("Old weight:")
    print(*pl.get_weight())
    pl.set_weight(lin_reg.new_weight(pl.get_weight(),pl.get_data_value(),lin_reg.get_error()))
    print("New weight:")
    print(*pl.get_weight())
    print("new output:")
    print(*pl.get_output())
    print("step size:")
    print(lin_reg.get_step_size())
    print("new error:")
    lin_reg.set_error(lin_reg.calc_error(pl.get_output(),y))
    print(*lin_reg.get_error())
    print("mse:")
    print(lin_reg.mean_squared_error(lin_reg.get_error()))

