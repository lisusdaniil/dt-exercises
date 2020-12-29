import json
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern
import pickle
import matplotlib.pyplot as plt

# Settings
load_model = False
save_model = True
augment_data = False

list_test_files = ["data_test.json", "data_set_2.json"]
#list_test_files = ["real_set_3.json"]
#list_train_files = ["data_set_1.json", "data_set_2.json", "data_set_3.json"]
list_train_files = ["data_set_1.json", "real_set_1.json", "real_set_2.json"]
model_save = 'trained_model.sav'
dir = '/home/daniil/Documents/School/Classes/Duckie/Duckietown_git/dt-exercises/lane_state_prediction/exercise_ws/data/'


print("Training starting")
data_train = []
for train_data_file in list_train_files:
    with open(dir+train_data_file, "r") as f:
        print()
        data_train = data_train + json.load(f)
data_test = []
for test_data_file in list_test_files:
    with open(dir+test_data_file, "r") as f:
        data_test = data_test + json.load(f)


# Load in training data
x = np.zeros([len(data_train), 12])
y = -np.ones([len(data_train)])
ii = 0
print(x.shape)
for pt in data_train:
    if pt['direction'] == 'straight':
        y[ii] = 0
    elif pt['direction'] == 'left':
        y[ii] = 1
    elif pt['direction'] == 'right':
        y[ii] = 2
    else:
        Exception("Ahhhh")
    w_avg_x = pt['white']['avg']['x']
    w_avg_y = pt['white']['avg']['y']
    w_min_x = pt['white']['min']['x']
    w_min_y = pt['white']['min']['y']
    w_max_x = pt['white']['max']['x']
    w_max_y = pt['white']['max']['y']

    y_avg_x = pt['yellow']['avg']['x']
    y_avg_y = pt['yellow']['avg']['y']
    y_min_x = pt['yellow']['min']['x']
    y_min_y = pt['yellow']['min']['y']
    y_max_x = pt['yellow']['max']['x']
    y_max_y = pt['yellow']['max']['y']
    x[ii] = np.array([w_avg_x, w_avg_y, w_min_x, w_min_y, w_max_x, w_max_y, y_avg_x, y_avg_y, y_min_x, y_min_y, y_max_x, y_max_y])

    # Augment data
    if augment_data:
        if not(w_avg_x == w_avg_y == w_min_x == w_min_y == 0):
            # white segments exist
            if not(y_avg_x == y_avg_y == y_min_x == y_min_y == 0):
                # yellow segments also exist
                x_aug = np.zeros([2, 12])
                y_aug = y[ii]*np.ones([2,])

                # add only white and only yellow detected segments
                x_aug[0] = np.array([w_avg_x, w_avg_y, w_min_x, w_min_y, w_max_x, w_max_y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                x_aug[1] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, y_avg_x, y_avg_y, y_min_x, y_min_y, y_max_x, y_max_y])
                x = np.vstack((x,x_aug))
                y = np.append(y,y_aug)
        # augment by flipping turns
        if y[ii] != 0:
            x_aug = np.zeros([1, 12])
            y_aug = y[ii]*np.ones([1,])
            x_aug[0] = np.array([w_avg_x, -w_avg_y, w_min_x, -w_min_y, w_max_x, -w_max_y, y_avg_x, -y_avg_y, y_min_x, -y_min_y, y_max_x, -y_max_y])
            x = np.vstack((x,x_aug))
            y = np.append(y,y_aug)

    ii += 1
print(x.shape)

# Load in testing data
x_star = np.zeros([len(data_test), 12])
y_star = -np.ones([len(data_test)])
for ii, pt in enumerate(data_test):
    if pt['direction'] == 'straight':
        y_star[ii] = 0
    elif pt['direction'] == 'left':
        y_star[ii] = 1
    elif pt['direction'] == 'right':
        y_star[ii] = 2
    else:
        Exception("Ahhhh")
    w_avg_x = pt['white']['avg']['x']
    w_avg_y = pt['white']['avg']['y']
    w_min_x = pt['white']['min']['x']
    w_min_y = pt['white']['min']['y']
    w_max_x = pt['white']['max']['x']
    w_max_y = pt['white']['max']['y']

    y_avg_x = pt['yellow']['avg']['x']
    y_avg_y = pt['yellow']['avg']['y']
    y_min_x = pt['yellow']['min']['x']
    y_min_y = pt['yellow']['min']['y']
    y_max_x = pt['yellow']['max']['x']
    y_max_y = pt['yellow']['max']['y']
    x_star[ii] = np.array([w_avg_x, w_avg_y, w_min_x, w_min_y, w_max_x, w_max_y, y_avg_x, y_avg_y, y_min_x, y_min_y, y_max_x, y_max_y])


# define model
if load_model:
    model = pickle.load(open(dir+model_save, 'rb'))
else:
    model = GaussianProcessClassifier(kernel=1.0*RBF(1.0), n_restarts_optimizer=2, multi_class='one_vs_one', n_jobs=-1, max_iter_predict=1000, warm_start=True, copy_X_train=True)
model.fit(x,y)

err = 0
print("Training set results")
print("True| Predicted")
for ii in range(x.shape[0]):
    row = x[ii,:]
    # make a prediction
    yhat = model.predict([row])
    # summarize prediction
    print(" %d  |    %d    " % (y[ii], yhat))

    if y[ii] != yhat:
        print("idx: " + str(ii))
        err += 1

print("Accuracy train: %.2f %%" % (1 - err/x.shape[0]))

print("Testing set results")
err_star = 0
print("True| Predicted")
for ii in range(x_star.shape[0]):
    row = x_star[ii,:]
    # make a prediction
    yhat = model.predict([row])
    # summarize prediction
    print(" %d  |    %d    " % (y_star[ii], yhat))

    if y_star[ii] != yhat:
        print("idx: " + str(ii))
        err_star += 1

print("Accuracy test: %.2f %%" % (1 - err_star/x_star.shape[0]))


if save_model:
    pickle.dump(model, open(dir+model_save, 'wb'))



idx = 19
x = []
y = []
print(data_test[idx]['direction'])
for seg in data_test[idx]['white']['segs']:
    x.append(seg['x'])
    y.append(-seg['y'])

x_y = []
y_y = []
for seg in data_test[idx]['yellow']['segs']:
    x_y.append(seg['x'])
    y_y.append(-seg['y'])

plt.plot(-data_test[idx]['white']['avg']['y'], data_test[idx]['white']['avg']['x'], 'x', color='black')
plt.plot(y, x, 'o', color='black')
plt.plot(-data_test[idx]['yellow']['avg']['y'], data_test[idx]['yellow']['avg']['x'], 'x', color='red')
plt.plot(y_y, x_y, 'o', color='red')
plt.show()