__author__ = 'bptripp'

import numpy as np
from os.path import join
import scipy
import cPickle
from keras.optimizers import Adam
from data import load_all_params
from keras.models import model_from_json
from overlap_model import get_input

model = model_from_json(open('o-model-architecture.json').read())
model.load_weights('o-model-weights.h5')

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy', optimizer=adam)

objects, gripper_pos, gripper_orient, labels = load_all_params('../../grasp-conv/data/output_data.csv')
seq_nums = np.arange(len(objects)) % 1000 #exactly 1000 per object in above file (dated March 18)

labels = np.array(labels)[:,np.newaxis]

n = len(objects)
validation_indices = np.random.randint(0, n, 500) #TODO:

Y_valid = labels[validation_indices,:]
X_valid = []
for ind in validation_indices:
    X_valid.append(get_input(objects[ind], seq_nums[ind]))
X_valid = np.array(X_valid)

