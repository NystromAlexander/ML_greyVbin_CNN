import sys
import os
import numpy as np
import tensorflow as tf
import csv
import pickle
import tarfile
import zipfile as z
import threading
from scipy import ndimage
from scipy.misc import imsave
from skimage.transform import resize
from skimage import filters
from PIL import Image
import matplotlib.pyplot as plt


def load_class(folder, image_size, pixel_depth, binary=False):
    image_files = os.listdir(folder)
    num_of_images = len(image_files)
    dataset = np.ndarray(shape=(num_of_images, image_size, image_size),
                         dtype=np.float32)
    image_index = 0
    print('Started loading images from: ' + folder)
    for index, image in enumerate(image_files):

        sys.stdout.write('Loading image %d of %d\r' % (index + 1, num_of_images))
        sys.stdout.flush()

        image_file = os.path.join(folder, image)

        try:
            image_data = (ndimage.imread(image_file,flatten=True).astype(float)-
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                # print('Resizing image')
                image_data = resize(image_data, (image_size, image_size))

            if binary:
                val = filters.threshold_otsu(image_data)
                dataset[image_index, :, :] = image_data < val
            else:
                dataset[image_index, :, :] = image_data
            # plt.imshow(image_data < val, cmap='gray')
            # plt.show()
            image_index += 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    print('Finished loading data from: ' + folder)

    return dataset[0:image_index, :, :]

def reformat(data, image_size, num_of_channels, num_of_classes, flatten=True):
    if flatten:
        data.grey_train_dataset = data.grey_train_dataset.reshape((-1, image_size * image_size * num_of_channels)).astype(np.float32)
        # data.grey_valid_dataset = data.grey_valid_dataset.reshape((-1, image_size * image_size * num_of_channels)).astype(np.float32)
        data.grey_test_dataset = data.grey_test_dataset.reshape((-1, image_size * image_size * num_of_channels)).astype(np.float32)
        data.bin_train_dataset = data.bin_train_dataset.reshape((-1, image_size * image_size * num_of_channels)).astype(np.float32)
        # data.bin_valid_dataset = data.bin_valid_dataset.reshape((-1, image_size * image_size * num_of_channels)).astype(np.float32)
        data.bin_test_dataset = data.bin_test_dataset.reshape((-1, image_size * image_size * num_of_channels)).astype(np.float32)
    else:
        data.grey_train_dataset = data.grey_train_dataset.reshape((-1, image_size, image_size, num_of_channels)).astype(np.float32)
        # data.grey_valid_dataset = data.grey_valid_dataset.reshape((-1, image_size, image_size, num_of_channels)).astype(np.float32)
        data.grey_test_dataset = data.grey_test_dataset.reshape((-1, image_size, image_size, num_of_channels)).astype(np.float32)
        data.bin_train_dataset = data.bin_train_dataset.reshape((-1, image_size, image_size, num_of_channels)).astype(np.float32)
        # data.bin_valid_dataset = data.bin_valid_dataset.reshape((-1, image_size, image_size, num_of_channels)).astype(np.float32)
        data.bin_test_dataset = data.bin_test_dataset.reshape((-1, image_size, image_size, num_of_channels)).astype(np.float32)

    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    data.grey_train_labels = (np.arange(num_of_classes) == data.grey_train_labels[:, None]).astype(np.float32)
    # data.grey_valid_labels = (np.arange(num_of_classes) == data.grey_valid_labels[:, None]).astype(np.float32)
    data.grey_test_labels = (np.arange(num_of_classes) == data.grey_test_labels[:, None]).astype(np.float32)
    data.bin_train_labels = (np.arange(num_of_classes) == data.bin_train_labels[:, None]).astype(np.float32)
    # data.bin_valid_labels = (np.arange(num_of_classes) == data.bin_valid_labels[:, None]).astype(np.float32)
    data.bin_test_labels = (np.arange(num_of_classes) == data.bin_test_labels[:, None]).astype(np.float32)

    return data

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, image_size, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


def make_pickles(input_folder, output_dir, image_size, image_depth, FORCE=False, binary=False):
    directories = sorted([x for x in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, x))])
    pickle_files = [os.path.join(output_dir, x + '.pickle') for x in directories]

    for index, pickle_file in enumerate(pickle_files):

        if os.path.isfile(pickle_file) and not FORCE:
            print('\tPickle already exists: %s' % (pickle_file))
        else:
            folder_path = os.path.join(input_folder, directories[index])
            print('\tLoading from folder: ' + folder_path)
            data = load_class(folder_path, image_size, image_depth, binary=binary)

            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            print('\tStarted pickling: ' + directories[index])
            try:
                with open(pickle_file, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', pickle_file, ':', e)
            print('Finished pickling: ' + directories[index])

    return pickle_files

def pickle_whole(train_pickle_files, test_pickle_files, image_size,
                 train_size, valid_size, test_size, output_file_path, FORCE=False):
    if os.path.isfile(output_file_path) and not FORCE:
        print('Pickle file: %s already exist' % (output_file_path))

        with open(output_file_path, 'rb') as f:
            save = pickle.load(f)
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            valid_dataset = save['valid_dataset']
            valid_labels = save['valid_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
            del save  # hint to help gc free up memory
            print('Training set', train_dataset.shape, train_labels.shape)
            # print('Validation set', valid_dataset.shape, valid_labels.shape)
            print('Test set', test_dataset.shape, test_labels.shape)

        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
    else:
        print('Merging train, valid data')
        valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
            train_pickle_files, image_size, train_size, valid_size)
        print('Merging test data')
        _, _, test_dataset, test_labels = merge_datasets(test_pickle_files, image_size, test_size)
        print('Training set', train_dataset.shape, train_labels.shape)
        # print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)

        train_dataset, train_labels = randomize(train_dataset, train_labels)
        test_dataset, test_labels = randomize(test_dataset, test_labels)
        # valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
        try:
            f = open(output_file_path, 'wb')
            save = {
                'train_dataset': train_dataset,
                'train_labels': train_labels,
                'valid_dataset': valid_dataset,
                'valid_labels': valid_labels,
                'test_dataset': test_dataset,
                'test_labels': test_labels,
            }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', output_file_path, ':', e)
            raise

        statinfo = os.stat(output_file_path)
        print('Compressed pickle size:', statinfo.st_size)

        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels






def prepare_data(root_dir="."):
    image_size = 300
    image_depth = 255
    num_of_classes = 6
    train_size = 60*num_of_classes
    test_size = 10*num_of_classes
    valid_size = 0*num_of_classes


    num_of_channels = 1

    dataset_path = os.path.realpath(os.path.join(root_dir, "dataset", "objects"))
    grey_train_path = os.path.join(dataset_path, "grey_train")
    grey_test_path = os.path.join(dataset_path, "grey_test")
    bin_train_path = os.path.join(dataset_path, "bin_train")
    bin_test_path = os.path.join(dataset_path, "bin_test")

    grey_train_pickle_files = make_pickles(os.path.join(dataset_path, "train"), grey_train_path, image_size, image_depth)
    grey_test_pickle_files = make_pickles(os.path.join(dataset_path, "test"), grey_test_path, image_size, image_depth)

    bin_train_pickle_files = make_pickles(os.path.join(dataset_path, "train"), bin_train_path, image_size, image_depth, binary=True)
    bin_test_pickle_files = make_pickles(os.path.join(dataset_path, "test"), bin_test_path, image_size, image_depth, binary=True)

    grey_train_dataset, grey_train_labels, grey_valid_dataset, grey_valid_labels, \
    grey_test_dataset, grey_test_labels = pickle_whole(grey_train_pickle_files, grey_test_pickle_files, image_size, train_size, valid_size,
                                             test_size, os.path.join(dataset_path, 'grey.pickle'))

    bin_train_dataset, bin_train_labels, bin_valid_dataset, bin_valid_labels, \
    bin_test_dataset, bin_test_labels = pickle_whole(bin_train_pickle_files, bin_test_pickle_files, image_size, train_size, valid_size,
                                             test_size, os.path.join(dataset_path, 'bin.pickle'))
    # print(bin_train_pickle_files)
    # for index, pickle_file in  enumerate(bin_train_pickle_files):
    #     print(index, pickle_file)

    def objects() : pass

    objects.grey_train_dataset = grey_train_dataset
    objects.grey_train_labels = grey_train_labels
    objects.grey_test_dataset = grey_test_dataset
    objects.grey_test_labels = grey_test_labels
    objects.grey_valid_dataset = grey_valid_dataset
    objects.grey_valid_labels = grey_valid_labels

    objects.bin_train_dataset = bin_train_dataset
    objects.bin_train_labels = bin_train_labels
    objects.bin_test_dataset = bin_test_dataset
    objects.bin_test_labels = bin_test_labels
    objects.bin_valid_dataset = bin_valid_dataset
    objects.bin_valid_labels = bin_valid_labels

    return objects, image_size, num_of_classes, num_of_channels


# prepare_data()
# fo = open(os.path.realpath("./dataset/bin_train/crayfish.pickle"), 'rb')
# dict = pickle.load(fo)
# fo.close()
# print(dict)
# plt.imshow(dict[1], cmap='gray')
# plt.show()
