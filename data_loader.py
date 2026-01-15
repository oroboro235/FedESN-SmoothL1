# module for loading data from the dataset.

# libs
import os
import sys

import random

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from config import path, get_data_path


# Time series for regression

def read_data_mg():
    data = np.load(get_data_path("mg_t17")+"mackey_glass_t17.npy")
    data = data.reshape(-1, 1) # (10000, 1)

    len_train = 5000
    len_test = 1000

    seq_exp_u = data[:len_train+len_test] # discard the last one
    seq_exp_y = data[1:len_train+len_test+1] # shift one step to predict the next value

    idx_train = len_train
    idx_test = idx_train + len_test


    train_X = seq_exp_u[:idx_train]
    train_y = seq_exp_y[:idx_train]

    test_X = seq_exp_u[idx_train:idx_test]
    test_y = seq_exp_y[idx_train:idx_test]

    return train_X, train_y, test_X, test_y

def read_data_lorenz():
    data = np.load(get_data_path("lorenz")+"lorenz_full.npy") # (10000, 3)

    len_train = 5000
    len_test = 1000

    seq_exp_u = data[:len_train+len_test] # discard the last one
    seq_exp_y = data[1:len_train+len_test+1] # shift one step to predict the next value

    idx_train = len_train
    idx_test = idx_train + len_test


    train_X = seq_exp_u[:idx_train]
    train_y = seq_exp_y[:idx_train]

    test_X = seq_exp_u[idx_train:idx_test]
    test_y = seq_exp_y[idx_train:idx_test]

    return train_X, train_y, test_X, test_y




# Time series for classification

# UCI HAR dataset
def read_data_har():

    def preprocess_uci_har(data_dir):
        """Preprocess the UCI HAR dataset for sequence prediction task"""
        
        # list of sensor signal file names
        sensor_signals = [
            'body_acc_x', 'body_acc_y', 'body_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z'
        ]

        def load_dataset(subset):
            """loades the dataset of the given subset (train or test)"""
            features = []
            # load 9 sensor signals and stack them
            for signal in sensor_signals:
                file_path = os.path.join(
                    data_dir, subset, 'Inertial Signals', 
                    f'{signal}_{subset}.txt'
                )
                # load data and add channel dimension (samples, timesteps, 1)
                data = np.loadtxt(file_path)[:, :, np.newaxis]
                features.append(data)
            
            # concatenate all sensor features (samples, 128, 9)
            return np.concatenate(features, axis=-1)

        # load training and test sets
        X_train = load_dataset('train')
        X_test = load_dataset('test')

        # standardize: use statistics of training set
        # mean = X_train.mean(axis=(0,1))  # calculate mean along sample and timesteps
        # std = X_train.std(axis=(0,1))    # calculate std along sample and timesteps
        # X_train = (X_train - mean) / (std + 1e-8) 
        # X_test = (X_test - mean) / (std + 1e-8)

        # process labels (original labels are 1-6)
        def load_labels(subset):
            label_path = os.path.join(data_dir, subset, f'y_{subset}.txt')
            labels = np.loadtxt(label_path).astype(int) - 1  # convert to 0-5
            return labels
        # return np.eye(6)[labels]  # convert to one-hot encoding

        y_train = load_labels('train')
        y_test = load_labels('test')


        return X_train, y_train, X_test, y_test

    datapath = get_data_path("har")

    X_train, y_train, X_test, y_test = preprocess_uci_har(datapath)

    # get index of random samples
    train_idx = np.random.permutation(X_train.shape[0])
    test_idx = np.random.permutation(X_test.shape[0])

    # select subset of data
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]
    X_test = X_test[test_idx]
    y_test = y_test[test_idx]

    return X_train, y_train, X_test, y_test


# Written character dataset
def read_data_char():

    def preprocess_written_char(data_dir, fixed_len=128, num_classes=20, shuffle_seqs=True):
        """Preprocess the UCI written character dataset."""

        data = scipy.io.loadmat(data_dir)
        mixout = data["mixout"][0]
        consts = data["consts"][0,0]

        # acquire labels
        key = [item[0] for item in consts["key"][0]] # a, b, c,..., z in total 20 chars
        charlabels = consts["charlabels"][0] - 1 # 1, 1, 1, ... ,20, 20 in total 20 classes

        # padding the seqs
        padded_seqs = np.zeros((len(mixout), fixed_len, 3))
        for i in range(len(mixout)):
            seq = mixout[i].T
            seq_len = len(seq)
            if seq_len < fixed_len:
                padded_seqs[i, :seq_len, :] = seq
            else:
                padded_seqs[i] = seq[:fixed_len,:]
        mixout = padded_seqs
        
        # use previous num_classes characters
        if num_classes < 20 and num_classes > 0:
            classStartIdx_P1 = []
            classStartIdx_P1.append(0)
            classStartIdx_P2 = []
            cntClasses = 0
            maxClasses = 20
            for i in range(1, len(charlabels)):
                if charlabels[i] != charlabels[i-1]:
                    cntClasses += 1
                    if cntClasses < maxClasses:
                        classStartIdx_P1.append(i)
                    else:
                        classStartIdx_P2.append(i)
            new_charlabels = np.hstack([charlabels[classStartIdx_P1[0]:classStartIdx_P1[num_classes]],
                        charlabels[classStartIdx_P2[0]:classStartIdx_P2[num_classes]]])
            new_mixout = np.vstack([mixout[classStartIdx_P1[0]:classStartIdx_P1[num_classes]],
                        mixout[classStartIdx_P2[0]:classStartIdx_P2[num_classes]]])
            charlabels = new_charlabels
            mixout = new_mixout
            key = key[:num_classes]
        elif num_classes == 20:
            charlabels = charlabels
            mixout = mixout
            key = key
        else:
            raise ValueError("num_classes should be in [1, 20]")

        if shuffle_seqs:
            randomized_idx = np.random.permutation(len(mixout))
            mixout = mixout[randomized_idx]
            charlabels = charlabels[randomized_idx]

        return mixout, charlabels, key, data["consts"]

    def train_test_split(data, labels, test_size=0.2, shuffle=False):
        """Split the data into train and test sets."""
        if shuffle:
            randomized_idx = np.random.permutation(len(data))
            data = data[randomized_idx]
            labels = labels[randomized_idx]

        split_idx = int(len(data) * (1 - test_size))
        train_data = data[:split_idx]
        train_labels = labels[:split_idx]
        test_data = data[split_idx:]
        test_labels = labels[split_idx:]

        return train_data, train_labels, test_data, test_labels



    datapath = get_data_path("char")+"mixoutALL_shifted.mat"
    num_classes = 20

    X, y, _, _ = preprocess_written_char(datapath, num_classes=num_classes)


    # shuffle data
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    return X_train, y_train, X_test, y_test

# Univariate time series dataset
def read_data_uni(dataset_name):

    def load_classification_Uni_ts(dataset_name):
        from aeon.datasets import load_from_ts_file

        root_path = get_data_path("ucr")

        X_train, y_train = load_from_ts_file(root_path+dataset_name+"/"+dataset_name+"_TRAIN.ts", return_type="numpy3d")
        X_test, y_test = load_from_ts_file(root_path+dataset_name+"/"+dataset_name+"_TEST.ts", return_type="numpy3d")
        # switch dimension position
        X_train = np.swapaxes(X_train, 1, 2)
        X_test = np.swapaxes(X_test, 1, 2)
        # y
        if dataset_name in ["DistalPhalanxOutlineCorrect"]:
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
        else:
            y_train = y_train.astype(int)-1
            y_test = y_test.astype(int)-1
        
        return X_train, y_train, X_test, y_test

    X_train, y_train, X_test, y_test = load_classification_Uni_ts(dataset_name)

    train_idx = np.random.permutation(X_train.shape[0])
    test_idx = np.random.permutation(X_test.shape[0])

    X_train = X_train[train_idx]
    y_train = y_train[train_idx]
    X_test = X_test[test_idx]
    y_test = y_test[test_idx]

    return X_train, y_train, X_test, y_test

# read data from 


if __name__ == "__main__":
    # test
    X_train, y_train, X_test, y_test = read_data_char()
    # X_train, y_train, X_test, y_test = read_data_uni("Beef")

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)