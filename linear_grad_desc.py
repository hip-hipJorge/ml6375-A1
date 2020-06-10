import array as arr
import numpy as np

LEARNING_RATE = 0.1


class HypothesisSpace:
    def __init__(self, data_values, real_values, intercept=0):
        self.pop_size = len(data_values)
        self.data_values = np.array(data_values, dtype='f')
        self.real_values = real_values
        self.weight = np.array([1] * self.pop_size, dtype='f')
        self.intercept = intercept
        # output (y) for each line
        self.output = self.calc_output()

    # getters
    def get_pop_size(self):
        return self.pop_size

    def get_data_values(self):
        return self.data_values

    def get_real_values(self):
        return self.real_values

    def get_weight(self):
        return self.weight

    def get_output(self):
        return self.output


    # setters
    def set_data_values(self, data_values):
        self.data_values = data_values

    def set_real_values(self, real_values):
        self.real_values = real_values

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
        self.output = np.array([0] * self.pop_size, dtype='f')
        for i in range(self.pop_size):
            self.output[i] = self.weight[i] * self.data_values[i] + self.intercept
        return np.round(self.output,4)


class LinearRegression:
    def __init__(self, line, actual_data):
        self.error = self.calc_error(line.output, actual_data)
        self.mse = self.mean_squared_error(self.error)
        self.step_size = 1

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
        self.step_size = round(step_size,4)

    # helper functions
    def calc_error(self, hypothesis, actual_value):
        error = np.array([0] * len(actual_value), dtype='f')
        for i in range(len(actual_value)):
            error[i] = actual_value[i] - hypothesis[i]
        return np.round(error,4)

    def mean_squared_error(self, error):
        e_squared_sum = 0
        n = len(error)
        for i in range(n):
            e_squared_sum += pow(error[i], 2)
        return round((1 / (2 * n)) * e_squared_sum,4)

    def new_weight(self, old_weight, data_values, error):
        n = len(data_values)
        new_weight = np.array([0] * n, dtype='f')
        sum_error_and_data = 0
        for i in range(n):
            sum_error_and_data += error[i] * data_values[i]
        self.set_step_size(LEARNING_RATE * ((-1/n) * sum_error_and_data))
        for i in range(n):
            new_weight[i] = old_weight[i] - self.step_size
        return new_weight

    def optimize(self, hypothesis):
        n = len(hypothesis.get_data_values())
        for i in range(n):
            print(i)
            if abs(self.get_step_size()) < 0.001:
                break
            hypothesis.set_weight(self.new_weight(hypothesis.get_weight(),
                                                  hypothesis.get_data_values(),self.get_error()))
            self.set_error(self.calc_error(hypothesis.get_output(),hypothesis.get_real_values()))
            print(self.get_step_size())






# test 1 data point
w = [0.64, 0.64, 0.64]
x = [0.5, 2.3, 2.9]
y = [1.4, 1.9, 3.2]

hs = HypothesisSpace(x,y)
hs.set_weight(w)
lin_reg = LinearRegression(hs, y)

print("initial weight:")
print(hs.get_weight())
print("inital actual:")
print(y)
print("initial output:")
print(*hs.output)
print("initial error:")
print(*lin_reg.error)
print("initail mse:")
print(lin_reg.mean_squared_error(lin_reg.get_error()))
print()

lin_reg.optimize(hs)

print("New weight:")
print(hs.get_weight())
print("New actual:")
print(y)
print("New output:")
print(*hs.output)
print("New error:")
print(*lin_reg.error)
print("New mse:")
print(lin_reg.mean_squared_error(lin_reg.get_error()))
