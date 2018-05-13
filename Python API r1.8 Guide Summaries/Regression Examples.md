Regression Examples
This unit provides the following short examples demonstrating how to implement regression in Estimators:

Example	Demonstrates How To...
linear_regression.py	Use the tf.estimator.LinearRegressor Estimator to train a regression model on numeric data.
linear_regression_categorical.py	Use the tf.estimator.LinearRegressor Estimator to train a regression model on categorical data.
dnn_regression.py	Use the tf.estimator.DNNRegressor Estimator to train a regression model on discrete data with a deep neural network.
custom_regression.py	Use tf.estimator.Estimator to train a customized dnn regression model.
The preceding examples rely on the following data set utility:

Utility	Description
imports85.py	This program provides utility functions that load the imports85 data set into formats that other TensorFlow programs (for example, linear_regression.py and dnn_regression.py) can use.

Running the examples
You must install TensorFlow prior to running these examples. Depending on the way you've installed TensorFlow, you might also need to activate your TensorFlow environment. Then, do the following:

Clone the TensorFlow repository from github.
cd to the top of the downloaded tree.
Check out the branch for you current tensorflow version: git checkout rX.X
cd tensorflow/examples/get_started/regression.
You can now run any of the example TensorFlow programs in the tensorflow/examples/get_started/regression directory as you would run any Python program:

python linear_regressor.py
During training, all three programs output the following information:

The name of the checkpoint directory, which is important for TensorBoard.
The training loss after every 100 iterations, which helps you determine whether the model is converging.
For example, here's some possible output for the linear_regressor.py program:

INFO:tensorflow:Saving checkpoints for 1 into /tmp/tmpAObiz9/model.ckpt.
INFO:tensorflow:loss = 161.308, step = 1
INFO:tensorflow:global_step/sec: 1557.24
INFO:tensorflow:loss = 15.7937, step = 101 (0.065 sec)
INFO:tensorflow:global_step/sec: 1529.17
INFO:tensorflow:loss = 12.1988, step = 201 (0.065 sec)
INFO:tensorflow:global_step/sec: 1663.86
...
INFO:tensorflow:loss = 6.99378, step = 901 (0.058 sec)
INFO:tensorflow:Saving checkpoints for 1000 into /tmp/tmpAObiz9/model.ckpt.
INFO:tensorflow:Loss for final step: 5.12413.

linear_regressor.py
linear_regressor.py trains a model that predicts the price of a car from two numerical features.

Estimator	LinearRegressor, which is a pre-made Estimator for linear regression.
Features	Numerical: body-style and make.
Label	Numerical: price
Algorithm	Linear regression.
After training the model, the program concludes by outputting predicted car prices for two car models.


linear_regression_categorical.py
This program illustrates ways to represent categorical features. It also demonstrates how to train a linear model based on a mix of categorical and numerical features.

Estimator	LinearRegressor, which is a pre-made Estimator for linear regression.
Features	Categorical: curb-weight and highway-mpg.
Numerical: body-style and make.
Label	Numerical: price.
Algorithm	Linear regression.

dnn_regression.py
Like linear_regression_categorical.py, the dnn_regression.py example trains a model that predicts the price of a car from two features. Unlike linear_regression_categorical.py, the dnn_regression.py example uses a deep neural network to train the model. Both examples rely on the same features; dnn_regression.py demonstrates how to treat categorical features in a deep neural network.

Estimator	DNNRegressor, which is a pre-made Estimator for regression that relies on a deep neural network. The `hidden_units` parameter defines the topography of the network.
Features	Categorical: curb-weight and highway-mpg.
Numerical: body-style and make.
Label	Numerical: price.
Algorithm	Regression through a deep neural network.
After printing loss values, the program outputs the Mean Square Error on a test set.


custom_regression.py
The custom_regression.py example also trains a model that predicts the price of a car based on mixed real-valued and categorical input features, described by feature_columns. Unlike linear_regression_categorical.py, and dnn_regression.py this example does not use a pre-made estimator, but defines a custom model using the base Estimator class. The custom model is quite similar to the model defined by dnn_regression.py.

The custom model is defined by the model_fn argument to the constructor. The customization is made more reusable through params dictionary, which is later passed through to the model_fn when the model_fn is called.

The model_fn returns an EstimatorSpec which is a simple structure indicating to the Estimator which operations should be run to accomplish various tasks.