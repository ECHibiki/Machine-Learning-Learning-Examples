'''
Exercise done in pycharm for logic and jupyter notebook for Ipython viewing

Assignment from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb
Help from https://leemengtaiwan.github.io/simple-image-recognition-using-notmnist-dataset.html
'''

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle



# Config the matplotlib backend as plotting inline in IPython
'''%matplotlib inline'''


url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.'  # Change me to store data elsewhere


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    global last_percent_reported # store change
    percent = int(count * blockSize * 100 / totalSize) #current stage

    if last_percent_reported != percent:
        if percent % 5 == 0: #when on a 5% mark make a report
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    dest_filename = os.path.join(data_root, filename) #take parts of a filename and combines
    if force or not os.path.exists(dest_filename): #check if path exists
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook) # the store, the desitnation and callback for what to do on every event
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception(
            'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename



train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

num_classes = 10 #10 files
np.random.seed(133) #seed numpy


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    #force = True
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

'''
for imagefile in os.listdir(test_folders[0]):
    test_folders_test_path = (test_folders[0].replace('.', "") + "\\" + imagefile)[1:]
    i = Image(filename=test_folders_test_path)
    display(i)
'''

image_size  = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32) #create 3D array of 32 bit float datatype
    num_images = 0
    for image in image_files: #read each image
        image_file = os.path.join(folder, image) #from path of
        try:
            image_data = (imageio.imread(image_file).astype(float) - #read as float
                          pixel_depth / 2) / pixel_depth #pixel colors are reduced by normalization function to black or white
#https://stackoverflow.com/questions/47185239/image-classification-what-is-pixel-depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except (IOError, ValueError) as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle' #file name for binary reduced data
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


''' #Read train test pickle set to verify all is in order
print('train set')
for train_dataset in train_datasets:
    train_dataset = train_dataset[2:]
    print(train_dataset)
    pickle_file = train_dataset # index 0 should be all As, 1 = all Bs, etc.
    with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)  # unpickle
        sample_idx = np.random.randint(len(letter_set))  # pick a random image index
        sample_image = letter_set[sample_idx, :, :]  # extract a 2D slice
        print('Number of samples in {}: {}'.format(train_dataset, letter_set.shape[0]))
        plt.figure()
        plt.imshow(sample_image, 'Blues')  # display it

print('test set') #read test pickle set to see all is good
for test_dataset in test_datasets:
    test_dataset = test_dataset[2:]
    print(test_dataset)
    pickle_file = test_dataset
    with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        sample_idx = np.random.randint(len(letter_set))
        sample_image = letter_set[sample_idx, :, :]
        print('number of samples in {}: {}'.format(test_dataset, letter_set.shape[0]))
        plt.figure()
        plt.imshow(sample_image, 'Greys')
'''

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32) #3D array for numeric index(A-J), and image XY
        labels = np.ndarray(nb_rows, dtype=np.int32) #The labels for all dataset to resolve to
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size) #create datasets for features and lables in the validation set
    train_dataset, train_labels = make_arrays(train_size, image_size) #create datasets for features and lables in the training set
    vsize_per_class = valid_size // num_classes #integer division for V size
    tsize_per_class = train_size // num_classes #integer division for T size

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files): #Nested loop going through pickle files as file and label as the hold point
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f) #read letters from file
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set) #randomize the letters
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :] #valid letter is a subset from 0 to size of the current V dataset
                    valid_dataset[start_v:end_v, :, :] = valid_letter #add the letter subset to a slice of that position
                    valid_labels[start_v:end_v] = label #label is in the slice
                    start_v += vsize_per_class #update for randomness
                    end_v += vsize_per_class #update for randomness

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


train_size = 200000
valid_size = 10000
test_size = 18724



valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

pickle_file = os.path.join(data_root, 'notMNIST.pickle') #this file will contain the shuffled data with  lables on the side
if not os.path.isfile(pickle_file):
    try:
      f = open(pickle_file, 'wb')
      save = {
        'train_dataset': train_dataset, #write the train dataset
        'train_labels': train_labels, #write train labels
        'valid_dataset': valid_dataset, #write valid data
        'valid_labels': valid_labels, #valid labels
        'test_dataset': test_dataset, #test data
        'test_labels': test_labels, #test labels
        }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL) #save as pickle file
      #pickle serialization https://docs.python.org/2/library/pickle.html
      f.close()
    except Exception as e:
      print('Unable to save data to', pickle_file, ':', e)
      raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size) # get size

'''
print('x to quit')
end = input() #heavy lifting here
if end is 'x':
    exit()
'''
import time
start_time = time.time()


#Use euclidian distance(cosine similarity) to determin how related two images are in
#valid or test datasets against train datasets
def get_duplicate_data(source_dataset, target_dataset, threshold=1, num_duplicate_to_show=0):
    X = source_dataset.reshape(source_dataset.shape[0], -1) #change the dim
    Y = target_dataset.reshape(target_dataset.shape[0], -1)
    assert (X.shape[1] == Y.shape[1])

    dim = X.shape[1]
    #The bellow will cause  a crash on most machines
    cosine_sim = np.inner(X, Y) / np.inner(np.abs(X), np.abs(Y)) #dot product and find cosine-sim
    assert (cosine_sim.shape == (X.shape[0], Y.shape[0]))

    # for each image in training set, find corresponding duplicate in test/valid set
    dup_target_indices = []
    show_duplicate_counter = 0

    for source_idx in range(cosine_sim.shape[0]):
        dup_indices = list(np.where(cosine_sim[source_idx, :] >= threshold)[0])
        '''
        # render duplicate images when is available. may omit if visual output is not required
        if dup_indices and num_duplicate_to_show and (show_duplicate_counter < num_duplicate_to_show):
            # show only non-redudent duplicate images
            for i in dup_indices:
                if i in dup_target_indices:
                    dup_indices.remove(i)
            if not dup_indices: continue

            if len(dup_indices) == 1:

                fig = plt.figure(figsize=(3, 15))
                fig.add_subplot(1, len(dup_indices) + 1, 1)
                plt.imshow(source_dataset[source_idx, :, :], cmap='gray')
                plt.title('Source: ' + str(source_idx))
                plt.axis('off')

                for i, target_idx in enumerate(dup_indices):
                    fig.add_subplot(1, len(dup_indices) + 1, i + 2)
                    plt.imshow(target_dataset[target_idx, :, :], cmap='gray')
                    plt.title('Target: ' + str(target_idx))
                    plt.axis('off')

                show_duplicate_counter += 1
            '''#
        dup_target_indices.extend(dup_indices)
    return list(set(dup_target_indices))

print('death?')
#input()

dup_indices_test_split = []
dup_indices_test = []
max_val=10
for i in range(0,max_val):
    dup_indices_test_split = get_duplicate_data(train_dataset[len(train_dataset)//max_val * (i):len(train_dataset)//max_val * (i+1)], test_dataset, num_duplicate_to_show=5)
    if dup_indices_test_split is not None:
        for data in dup_indices_test_split:
            dup_indices_test.append(data)


print('Number of duplicates in test dataset: {}'.format(len(dup_indices_test)))


dup_indices_valid_split = []
dup_indices_valid = []
max_val=10
for i in range(0,max_val):
    dup_indices_valid_split = get_duplicate_data(train_dataset[len(train_dataset)//max_val * (i):len(train_dataset)//max_val * (i+1)], valid_dataset, num_duplicate_to_show=5)
    if dup_indices_valid_split is not None:
        for data in dup_indices_valid_split:
            dup_indices_valid.append(data)

print('Number of duplicates in validation dataset: {}'.format(len(dup_indices_valid)))

non_duplicate_indices = [i for i in range(test_dataset.shape[0]) if not i in dup_indices_test]
sanitized_test_dataset = test_dataset[non_duplicate_indices, :, :]
sanitized_test_labels = test_labels[non_duplicate_indices]

non_duplicate_indices = [i for i in range(valid_dataset.shape[0]) if not i in dup_indices_valid]
sanitized_valid_dataset = valid_dataset[non_duplicate_indices, :, :]
sanitized_valid_labels = valid_labels[non_duplicate_indices]
pickle_file = os.path.join(data_root, 'notMNIST_sanitized.pickle')

if not os.path.isfile(pickle_file):
    try:
      f = open(pickle_file, 'wb')
      save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': sanitized_valid_dataset,
        'valid_labels': sanitized_valid_labels,
        'test_dataset': sanitized_test_dataset,
        'test_labels': sanitized_test_labels,
        }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print('Unable to save data to', pickle_file, ':', e)
      raise
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

print(time.time() - start_time)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
plt.style.use('ggplot')
np.random.seed(42)
train_sizes = [100, 1000, 50000, 100000, 200000]

# train models using different size of training set
test_scores, test_scores_sanitized = [[] for _ in range(2)]
for train_size in train_sizes:
    # random choose #train_size of training instances
    indices = np.random.randint(0, train_dataset.shape[0], train_size)

    # reshape images to (train_size, dim * dim) for easier processing
    X = train_dataset[indices, :, :]\
        .reshape(-1, train_dataset.shape[1] * train_dataset.shape[2])
    y = train_labels[indices]

    # train model
    clf = (LogisticRegression(random_state=10, solver='lbfgs', multi_class='multinomial')
                  .fit(X, y))

    # test on original test set and the sanitized one
    y_pred = clf.predict(test_dataset.reshape(test_dataset.shape[0], -1))
    y_pred_sanitized = clf.predict(sanitized_test_dataset.reshape(sanitized_test_dataset.shape[0], -1))

    test_score = accuracy_score(y_pred, test_labels)
    test_score_sanitized = accuracy_score(y_pred_sanitized, sanitized_test_labels)
    test_scores.append(test_score)
    test_scores_sanitized.append(test_score_sanitized)



#     print(classification_report(test_labels, y_pred))
#     print(accuracy_score(test_labels, y_pred))


plt.figure(figsize=(7, 7))
plt.xlabel('Training size', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
for x, y in zip(train_sizes, test_scores):
    plt.text(x + 50, y, '{:.2f}'.format(y))
for x, y in zip(train_sizes, test_scores_sanitized):
    plt.text(x + 50, y, '{:.2f}'.format(y))

plt.scatter(train_sizes, test_scores, label='Test score', color='blue');
plt.scatter(train_sizes, test_scores_sanitized, label='Test score(Sanitized)', color='red');
plt.legend(loc=4)
plt.title('Test set Accuracy vs Training size', fontsize=25);

print("done")
