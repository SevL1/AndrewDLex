import numpy as np
import h5py

def load_dataset():
    train_set = h5py.File('train_catvnoncat.h5', 'r')
    train_set_x = np.array(train_set['train_set_x'][:])
    train_set_y = np.array(train_set['train_set_y'][:])
    
    test_set= h5py.File('test_catvnoncat.h5', 'r')
    test_set_x = np.array(test_set['test_set_x'][:])
    test_set_y = np.array(test_set['test_set_y'][:])
    
    classes = np.array(test_set['list_classes'][:])
    
    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))
    
    return train_set_x, train_set_y, test_set_x, test_set_y, classes