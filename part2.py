import pandas as pd
import numpy as np
import requests
import io
from linear_grad_desc import list_format
from sklearn import linear_model
from sklearn.model_selection import train_test_split



# read data from...
# url = "https://www.utdallas.edu/~jxp175430/car.data"
# if that url does not read, comment out and use this one:
url = "https://raw.githubusercontent.com/hip-hipJorge/ml6375-A1/master/car.data"

# read data from public source
read_data = requests.get(url).content
df = pd.read_csv(io.StringIO(read_data.decode('utf-8')), delimiter='\t')

# open log file
log = open("log2.txt", 'w')
print("Linear Regression Log Part 2\n", file=log)

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
tn = int(n / 4)
hs = np.empty([tn, 1], dtype='f')

# training_data, the data used to build
# y, the real values
df_x = np.empty([tn, 6], dtype='i')
df_y = np.empty([tn, 1], dtype='i')

# add data instances to training data
for i in range(tn):
    df_x[i] = list_format(list(df.iloc[i, 0:6]), attr)
    df_y[i] = list_format([df.iloc[i, 6]], attr)

# create linear model
reg = linear_model.LinearRegression()

# split training data
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=.25, random_state=4)
reg.fit(x_train, y_train)

# print Results
print("Results of (25/75) data split:", file=log)
print(reg.coef_, file=log)

# hypothesis
h = reg.predict(x_test)

# test data
nth_row = int(input("What row do we use to evaluate our hypothesis? (0-1728): "))
print("Desired Data instance is number %i" % nth_row, file=log)
class_val = df.iloc[nth_row, 6]

print("Desired Data instance is number %i" % nth_row, file=log)
print("The estimated class value: %.3f" % h[nth_row], file=log)
print("The real class value: %i\n" % attr[class_val], file=log)

# evaluation
if int(h+1) == attr[class_val] or int(h-1) == attr[class_val]:
    print("Hmm..not bad.", file=log)
else:
    print("Uh-oh. Needs work.", file=log)

# mean squared error
print("MSE:", file=log)
print(np.mean((h - y_test)) ** 2)
print(file=log)


log.close()
print("See report in log.txt")
