'''
Exercise done in pycharm for logic and jupyter notebook for Ipython viewing

ISSUE: sanitized is probably broken

'''
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


# Config the matplotlib backend as plotting inline in IPython
'''%matplotlib inline'''

pickle_file  = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

pickle_file_sanitized = 'notMNIST_sanitized.pickle'
with open(pickle_file_sanitized, 'rb') as f:
    save_san = pickle.load(f)
    train_dataset_sanitized = save_san['train_dataset']
    train_labels_sanitized = save_san['train_labels']
    valid_dataset_sanitized = save_san['valid_dataset']
    valid_labels_sanitized = save_san['valid_labels']
    test_dataset_sanitized = save_san['test_dataset']
    test_labels_sanitized = save_san['test_labels']

image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32) #convert from 3D to 2D with index dimension
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('---\nTraining set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

train_dataset_sanitized, train_labels_sanitized = reformat(train_dataset_sanitized, train_labels_sanitized)
valid_dataset_sanitized, valid_labels_sanitized = reformat(valid_dataset_sanitized, valid_labels_sanitized)
test_dataset_sanitized, test_labels_sanitized = reformat(test_dataset_sanitized, test_labels_sanitized)
print('---\nTraining set Sant', train_dataset_sanitized.shape, train_labels_sanitized.shape)
print('Validation set Sant', valid_dataset_sanitized.shape, valid_labels_sanitized.shape)
print('Test set Sant', test_dataset_sanitized.shape, test_labels_sanitized.shape)

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 10000 #gets a 10.5%, as good as a random pick

learning_rate = 0.01

graph = tf.Graph() #Graph is a set of tf.Operation objects repreesnting units of compuitation and td.Tensor objects represting data to flow between operations
#Important note: This class is not thread-safe for graph construction. All operations should be created from a single thread, or external synchronization must be provided. Unless otherwise specified, all methods are not thread-safe.
with graph.as_default():
      # Input data.
      # Load the training, validation and test data into constants that are
      # attached to the graph.
      tf_train_dataset = tf.constant(train_dataset[:train_subset, :]) #a constant is given a dtype with arguments and optional shape
      '''
      # Constant 1-D Tensor populated with value list.
tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]

# Constant 2-D tensor populated with scalar value -1.
tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.]
                                             [-1. -1. -1.]]
                                              '''
      tf_train_labels = tf.constant(train_labels[:train_subset])
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)
      '''The above does not change'''

      # Variables.
      # These are the parameters that we are going to be training. The weight
      # matrix will be initialized using random values following a (truncated)
      # normal distribution. The biases get initialized to zero.
      weights = tf.Variable( #variable mainstainsts state when graph calls run(). Variable takes an initial value which can be a tensor of any type. After construction type and shape are fixed
        tf.truncated_normal([image_size * image_size, num_labels])) #truncated_normal outputs a random value from a truncated normal distribution
      biases = tf.Variable(tf.zeros([num_labels])) #variables are assigned to the graph
      '''These will adjust to give values'''

      # Training computation.
      # We multiply the inputs with the weight matrix, and add biases. We compute
      # the softmax and cross-entropy (it's one operation in TensorFlow, because
      # it's very common, and it can be optimized). We take the average of this
      # cross-entropy across all training examples: that's our loss.
      logits = tf.matmul(tf_train_dataset, weights) + biases #matrix multiply ojbect of  wieghts and Training with biases, stored for adjustment
      '''
      logit function is the inverse of the sigmoidal "logistic" function represents a probability p, logit function gives the log-odds, or the logarithm of the odds p/(1 − p).
      https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow
      Logit is a function that maps probabilities [0, 1] to [-inf, +inf].
      Softmax is a function that maps [-inf, +inf] to [0, 1] similar as Sigmoid. But Softmax also normalizes the sum of the values(output vector) to be 1.
      Tensorflow "with logit": It means that you are applying a softmax function to logit numbers to normalize it. The input_vector/logit is not normalized and can scale from [-inf, inf].
      '''
      # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels)) #Computes softmax cross entropy between logits and labels. (deprecated)
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf_train_labels)) #finds cross entropy between logits and lables. Excpects unscaled logits since performs softmax on logits for efficiency
      # Optimizer.
      # We are going to find the minimum of this loss using gradient descent.
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) # Optimizer for gradient descent.  Has learning rate and locks for updates. Follows by minimize with variable containing the value to minimize.

      # Predictions for the training, validation, and test data.
      # These are not part of training, but merely here so that we can report
      # accuracy figures as we train.
      train_prediction = tf.nn.softmax(logits) #computes softmax activations retuyrnign tensor of same shape as logits.
      valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases)
      test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 801

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])



with tf.Session(graph=graph) as session:
    # This is a one-time operation which ensures the parameters get initialized as
    # we described in the graph: random weights for the matrix, zeros for the
    # biases.
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy
        # arrays.
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if (step % 100 == 0):
          print('Loss at step %d: %f' % (step, l))
          print('Training accuracy: %.1f%%' % accuracy(
            predictions, train_labels[:train_subset, :]))
          # Calling .eval() on valid_prediction is basically like calling run(), but
          # just to get that one numpy array. Note that it recomputes all its graph
          # dependencies.
          print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


'''SGD'''
print("-----------\nSDG UNSANITIZED")

#learning_rate = 0.01
batch_size = 128 #63-66.3% test accuracy @ LL 0.01
#batch_size = 10000 #decent test accuracy of 74.2%,

graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, #placeholders are always fed.  Takes the type of elements to be fed. Optional shape of tensor. Name for tessor operation
                                    shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf_train_labels)) #finds cross entropy between logits and lables. Excpects unscaled logits since performs softmax on logits for efficiency


    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


print("------\nSDG SANITIZED")

#batch_size = 128 #59-63% test accuracy
#batch_size = 10000 #decent test accuracy of 71%,

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset_sanitized)
    tf_test_dataset = tf.constant(test_dataset_sanitized)

    weights = tf.Variable(tf.truncated_normal([image_size*image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf_train_labels))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 3001
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run() #Returns an Op that initializes global variables.
    print("Initialized")
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset_sanitized[offset:(offset + batch_size), :]
        batch_labels = train_labels_sanitized[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels_sanitized))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels_sanitized))

'''
Problem
Turn the logistic regression example with SGD into a 1-hidden layer neural network
with rectified linear units nn.relu() and 1024 hidden nodes. This model should improve
your validation / test accuracy.
'''











print("-----\n1 DEEP NN")
'''http://x-wei.github.io/dlMOOC_L2.html'''
batch_size = 128
num_hidden = 1024

learning_rate = 0.025

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables for linear layer 1
  W1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_hidden]))
  b1 = tf.Variable(tf.zeros([num_hidden]))

  # Hidden RELU input computation
  y1 = tf.matmul(tf_train_dataset, W1) + b1
  # Hidden RELU output computation
  X1 = tf.nn.relu(y1)

  # Variables for linear layer 2
  W2 = tf.Variable(
    tf.truncated_normal([num_hidden, num_labels]))#W2
  b2 = tf.Variable(tf.zeros([num_labels])) #b2
  # logit (y2) output
  logits = tf.matmul(X1, W2) + b2
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf_train_labels))

  def getlogits(X):
    y1 = tf.matmul(X, W1) + b1
    X1 = tf.nn.relu(y1)
    return tf.matmul(X1, W2) + b2

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax( getlogits(tf_valid_dataset) )
  test_prediction = tf.nn.softmax( getlogits(tf_test_dataset))

#run sgd optimization:

num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
print("done")
