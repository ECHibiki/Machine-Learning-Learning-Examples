Metrics (contrib)
This module provides functions for computing streaming metrics:metrics computed on dynamically valued Tensors. 
To use any of these metrics, one need only declare the metric, call update_op repeatedly to accumulate data over the desired number of Tensor values (often each one is a single batch) and finally evaluate the value_tensor. 
Each metric function adds nodes to the graph that hold the state necessary to compute the value of the metric as well as a set of operations that actually perform the computation. 

Metric `Ops`
tf.contrib.metrics.streaming_mean_iou(predictions,labels,num_classes,weights=None,metrics_collections=None,updates_collections=None,name=None) // Calculate per step mean intersection over union
		// Returns mean_iou and update_op: An operation that increments the confusion matrix.
tf.contrib.metrics.streaming_mean_relative_error(predictions,labels,normalizer,weights=None,metrics_collections=None,updates_collections=None,name=None) // Computes mean relative error by normalizing with the given values
		// Returns the mean_relative_error and an update_op to update the values.
tf.contrib.metrics.streaming_mean_squared_error(predictions,labels,weights=None,metrics_collections=None,updates_collections=None,name=None)// Computes the mean squared error between the labels and predictions.
		// Returns the mean_squared_error and an update_op to update the values.
tf.contrib.metrics.streaming_root_mean_squared_error(predictions,labels,weights=None,metrics_collections=None,updates_collections=None,name=None)   //Computes the root mean squared error between the labels and predictions.
		// Returns as per usual
tf.contrib.metrics.streaming_covariance(predictions,labels,weights=None,metrics_collections=None,updates_collections=None,name=None)
		// Returns as per usual
		
**The following all behave as described. See docs for more specifics **
tf.contrib.metrics.streaming_pearson_correlation
		// Returns as per usual
tf.contrib.metrics.streaming_mean_cosine_distance
		// Returns as per usual
tf.contrib.metrics.streaming_percentage_less
		// Returns as per usual
tf.contrib.metrics.streaming_sensitivity_at_specificity
		// Returns as per usual
tf.contrib.metrics.streaming_sparse_average_precision_at_k
		// Returns as per usual
tf.contrib.metrics.streaming_sparse_precision_at_k
		// Returns as per usual
tf.contrib.metrics.streaming_sparse_precision_at_top_k
		// Returns as per usual
tf.contrib.metrics.streaming_sparse_recall_at_k
		// Returns as per usual
tf.contrib.metrics.streaming_specificity_at_sensitivity
		// Returns as per usual
tf.contrib.metrics.streaming_concat
		// Returns as per usual
tf.contrib.metrics.streaming_false_negatives_at_thresholds(predictions,labels,thresholds,weights=None)
tf.contrib.metrics.streaming_false_positives_at_thresholds(predictions,labels,thresholds,weights=None)
tf.contrib.metrics.streaming_true_negatives_at_thresholds(predictions,labels,thresholds,weights=None)
tf.contrib.metrics.streaming_true_positives_at_thresholds(predictions,labels,thresholds,weights=None)
** Ends **

tf.contrib.metrics.auc_using_histogram( boolean_labels, scores, score_range,nbins=100,collections=None,check_shape=True,name=None) // AUC through Histograms. Returns a float32 scalar tensor that converts internal histograms to an AUC value and OP function
tf.contrib.metrics.accuracy(predictions,labels,weights=None,name=None) // Percentage of times a prediction inmatched the labels. Returns the accuracy tensor
tf.contrib.metrics.aggregate_metrics(*value_update_tuples) 	// Aggregates metric value tensors and updates ops into two lists. Returns a list value tensor obects and list of update ops.
tf.contrib.metrics.aggregate_metric_map(names_to_tuples)	//Aggregates the metric names to tuple dictionary.
Set `Ops`
tf.contrib.metrics.set_difference( a, b, aminusb=True, validate_indices=True) // Compute set difference of elements in last dimension of a and b.
tf.contrib.metrics.set_intersection(    a,b,validate_indices=True) //Compute set intersection of elements in last dimension of a and b.
tf.contrib.metrics.set_size(a,validate_indices=True) //Compute number of unique elements along last dimension of a.int32 Tensor of set sizes. For a ranked n, this is a Tensor with rank n-1, and the same 1st n-1 dimensions as a. Each value is the number of unique elements in the corresponding [0...n-1] dimension of a.
tf.contrib.metrics.set_union(a,b,validate_indices=True) //Compute set union of elements in last dimension of a and b.