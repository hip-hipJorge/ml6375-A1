import pandas as pd
from linear_grad_desc import *
import requests
import io

# read data from...
# url = "https://www.utdallas.edu/~jxp175430/car.data"
# if that url does not read, comment out and use this one:
url = "https://raw.githubusercontent.com/hip-hipJorge/ml6375-A1/master/car.data"

# read data from public source
read_data = requests.get(url).content
df = pd.read_csv(io.StringIO(read_data.decode('utf-8')), delimiter='\t')

# open log file
log = open("log.txt", 'w')
print("Linear Regression Log Part 1\n", file=log)

# create hashmap for attributes
attr = {
    # buying/maint/safety
    'vhigh': 4,
    'high': 3,
    'med': 2,
    'low': 1,
    # lug_boot
    'big': 3,
    'small': 1,
    # doors/persons
    '2': 2,
    '3': 3,
    '4': 4,
    '5more': 5,
    # persons
    'more': 5,
    # class values
    'unacc': 1,
    'acc': 2,
    'good': 3,
    'vgood': 4
}

# lite pre-processing, feed data into hypothesis space

# n, the number of instances
# tn, the number of instances for training (75/25)
# hs, hypothesis space
n = df.shape[0]
tn = int(n/4)
hs = np.empty([tn, 1], dtype='f')

# training_data, the data used to build
# y, the real values
training_data = np.empty([tn, 6], dtype='i')
y = np.empty([tn, 1], dtype='i')

# add data instances to training data
for i in range(tn):
    training_data[i] = list_format(list(df.iloc[i, 0:6]), attr)
    y[i] = list_format([df.iloc[i, 6]], attr)


# work data to find best parameters
# prompt for number of iterations (assuming valid input)
iter = int(input("Number of iterations? (int): "))
print("Number of desired iterations is %i" % iter, file=log)

# train data
hypothesis = gradient_descent(hs, training_data, iter, y, log)

# select which instance of df to predict/compare class value (assuming valid input)
nth_row = int(input("What row do we use to evaluate our hypothesis? (0-1728): "))
print("Desired Data instance is number %i" % nth_row, file=log)
row = list_format(list(df.iloc[nth_row, 0:6]), attr)

# evaluate work
h = prediction(hypothesis, row)

# class value
class_val = df.iloc[nth_row, 6]

# results
print("--------------------------------------", file=log)
print("The estimated class value: %.3f" % h, file=log)
print("The real class value: %i\n" % attr[class_val], file=log)

if int(h+1) == attr[class_val] or int(h-1) == attr[class_val]:
    print("Hmm..not bad.\n\n", file=log)
else:
    print("Uh-oh. Needs work.", file=log)

print("Other important information:", file=log)
print("\tLearning rate: %.3f" % LEARNING_RATE, file=log)
print("\tOverall MSE: %.3f" % calc_mse(hs, y), file=log)
print("\tParameters after %i iterations:" % iter, file=log)
print(hypothesis, file=log)

log.close()
print("See report in log.txt")