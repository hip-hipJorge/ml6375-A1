import pandas as pd
from linear_grad_desc import *

# prepare data
url = "https://www.utdallas.edu/~jxp175430/car.data"

# df = pd.read_csv(url, delimiter='\t')
df = pd.read_csv("car.data", delimiter='\t')

# create hashmap for attributes
attr = {
    # buying/maint/safety
    'vhigh': 4,
    'high': 3,
    'med': 2,
    'low': 1,
    # lug_boot
    'big': 3,
    'med': 2,
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


# feed data into hypothesis space
# n, the number of instances
# tn, the number of instances for training
# hs, hypothesis space
n = df.shape[0]
tn = int(n/4)
hs = np.empty([tn,1], dtype='f')

# training_data, the data used to build
training_data = np.empty([tn, 6], dtype='i')
y = np.empty([tn, 1], dtype='i')


for i in range(tn):
    training_data[i] = list_format(list(df.iloc[i, 0:6]), attr)
    y[i] = list_format([df.iloc[i, 6]], attr)

ideal_weight = gradient_descent(hs, training_data, 200, y)
acc_test = list_format(list(df.iloc[500, 0:6]), attr)

h = prediction(ideal_weight, acc_test)

print(int(h))
y_test = df.iloc[500, 6]
print(y_test)
