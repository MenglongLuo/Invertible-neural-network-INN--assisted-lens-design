import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split

### The dataset of the lens ###
lens_info = {
    0: {'filename': 'LensV_PC_20deg.xlsx', 'sheet_name': 'LensV_PC_20deg', 'lensname': 'LensV_PC_20deg', 'usecols': 'A:E'},
    1: {'filename': 'LensV_PC_25deg.xlsx', 'sheet_name': 'LensV_PC_25deg', 'lensname': 'LensV_PC_25deg', 'usecols': 'A:E'},
    2: {'filename': 'LensV_HZF6_20deg.xlsx', 'sheet_name': 'LensV_HZF6_20deg', 'lensname': 'LensV_HZF6_20deg', 'usecols': 'A:E'},
    3: {'filename': 'LensV_HZF6_25deg.xlsx', 'sheet_name': 'LensV_HZF6_25deg', 'lensname': 'LensV_HZF6_25deg', 'usecols': 'A:E'},
    4: {'filename': 'LensH_PC.xlsx', 'sheet_name': 'LensH_PC', 'lensname': 'LensH_PC', 'usecols': 'A:F'},
    5: {'filename': 'LensH_HZF6.xlsx', 'sheet_name': 'LensH_HZF6', 'lensname': 'LensH_HZF6', 'usecols': 'A:F'}
}

########
lens = 0
########

## Load the dataset
filename = lens_info[lens]['filename']
sheet_name = lens_info[lens]['sheet_name']
lensname = lens_info[lens]['lensname']
usecols = lens_info[lens]['usecols']

df = np.array(pd.read_excel(filename, sheet_name=sheet_name, usecols=usecols))

performance = lensname + '_mse_mmd' + '.xlsx'
pklname = lensname + '.pkl'

### Pre-processing ###
input = 4 
output = 1 if lens in [0, 1, 2, 3] else 2
    
X = df[:,range(0, input)]
y = df[:,range(input, input + output)]

# Split the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True)

# Scaling X and y
scaler_x = StandardScaler().fit(X_train)
x_train = scaler_x.transform(X_train)
x_test = scaler_x.transform(X_test)

scaler_y = StandardScaler().fit(y_train)
y_train = scaler_y.transform(y_train)
y_test = scaler_y.transform(y_test)

x_train = torch.tensor(x_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
x_test = torch.tensor(x_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)

# Inverse_transform
def inverse_transform_x(x):
    actual_x = scaler_x.inverse_transform(x)
    return actual_x

def inverse_transform_y(y):
    actual_y = scaler_y.inverse_transform(y)
    return actual_y