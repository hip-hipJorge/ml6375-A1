import array as arr


class Line:
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

    def set_intercept(self, intercept):
        self.intercept = intercept

    def set_output(self, output):
        self.output = output

    # helper functions
    def calc_output(self):
        self.output = arr.array('d', [0] * self.pop_size)
        for i in range(self.pop_size):
            self.output[i] = self.weight[i] * self.data_value[i]
        return self.output


class LinearRegression:
    def __init__(self, line, actual_data):
        self.error = self.calc_error(line.output, actual_data)
        self.mse = self.mean_squared_error(self.error)

    # getters
    def get_error(self):
        return self.error

    def get_mse(self):
        return self.mse

    # setters
    def set_error(self, error):
        self.error = error

    def set_mse(self, mse):
        self.mse = mse

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


# test 1 data point
s = [0.64, 0.64, 0.64]
x = [0.5, 2.3, 2.9]
y = [1.4, 1.9, 3.2]

l = Line(s, x)
lin_reg = LinearRegression(l, y)

print("Output print: %.2f" % (l.get_output()))
print("Error print:")
print(*lin_reg.get_error())
print("MSE print:")
print(lin_reg.get_mse())



