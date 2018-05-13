Training 
----ORIG----
tf.train provides a set of classes and functions that help train models.

Optimizers
The Optimizer base class provides methods to compute gradients for a loss and apply gradients to variables. A collection of subclasses implement classic optimization algorithms such as GradientDescent and Adagrad.

You never instantiate the Optimizer class itself, but instead instantiate one of the subclasses.
ALL CLASSES
tf.train.Optimizer // CLASS for API and Ops to train model. Used with subclasses.
	# Create an optimizer with the desired parameters.
	opt = GradientDescentOptimizer(learning_rate=0.1)
tf.train.GradientDescentOptimizer // implements the gradient descent algorithm.
tf.train.AdadeltaOptimizer // Adadelta algorithm.
tf.train.AdagradOptimizer //  Adagrad algorithm.
tf.train.AdagradDAOptimizer // Takes care of regularization in a minibatch from AdagradDA. Used where there is a need for large sparsity. Garuntees sparsity for linear models.
tf.train.MomentumOptimizer // Momuntum algorithm
tf.train.AdamOptimizer //https://arxiv.org/abs/1412.6980
tf.train.FtrlOptimizer // FTRL algorthm https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
tf.train.ProximalGradientDescentOptimizer // Proximal gradient descent algorithmhttp://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf
tf.train.ProximalAdagradOptimizer // Proximal Adagrad algorithm. 
tf.train.RMSPropOptimizer // http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
See tf.contrib.opt for more optimizers.

Gradient Computation
TensorFlow provides functions to compute the derivatives for a given TensorFlow computation graph, adding operations to the graph. The optimizer classes automatically compute derivatives on your graph, but creators of new Optimizers or expert users can call the lower-level functions below.

tf.gradients(ys,xs,grad_ys=None,name='gradients',colocate_gradients_with_ops=False,gate_gradients=False,aggregation_method=None,
stop_gradients=None ) // Gets derivatives between ys and xs. Returns A list of sum(dy/dx) for each x in xs.
tf.AggregationMethod // Computing partial derivatives to require aggregating gradient contributions.
tf.stop_gradient // Useful to compute a value with TF but need to pretend it were constant. EM algorithm, contrastive divergence trainnig of Botlzmann machines. Adverserial training with no backprop.
tf.hessians // Hessians adds to the graph to output the Hessian matrix of ys of xs.
Gradient Clipping
TensorFlow provides several operations that you can use to add clipping functions to your graph. You can use these functions to perform general data clipping, but they're particularly useful for handling exploding or vanishing gradients.

tf.clip_by_value  // Clips tensor values to a specified min and max.
tf.clip_by_norm // Clips tensor values to a maximum L2-norm.
tf.clip_by_average_norm // Clips tensor values to a maximum average L2-norm.
tf.clip_by_global_norm // Clips values of multiple tensors by the ratio of the sum of their norms.
tf.global_norm // global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
Decaying the learning rate 
tf.train.exponential_decay //  decayed_learning_rate = learning_rate *  decay_rate ^ (global_step / decay_steps)
tf.train.inverse_time_decay // decayed_learning_rate = learning_rate / (1 + decay_rate * global_step /decay_step)
tf.train.natural_exp_decay // etc.
tf.train.piecewise_constant // etc.
tf.train.polynomial_decay // etc.
tf.train.cosine_decay // etc.
tf.train.linear_cosine_decay // etc.
tf.train.noisy_linear_cosine_decay // etc.
Moving Averages
Some training algorithms, such as GradientDescent and Momentum often benefit from maintaining a moving average of variables during optimization. Using the moving averages for evaluations often improve results significantly.

tf.train.ExponentialMovingAverage
Coordinator and QueueRunner
See Threading and Queues for how to use threads and queues. For documentation on the Queue API, see Queues.

tf.train.Coordinator // See docs
tf.train.QueueRunner // See docs
tf.train.LooperThread // See docs
tf.train.add_queue_runner // See docs
tf.train.start_queue_runners // See docs
Distributed execution
See Distributed TensorFlow for more information about how to configure a distributed TensorFlow program.

tf.train.Server // See docs
tf.train.Supervisor // See docs
tf.train.SessionManager // See docs
tf.train.ClusterSpec // See docs
tf.train.replica_device_setter // See docs
tf.train.MonitoredTrainingSession // See docs
tf.train.MonitoredSession // See docs
tf.train.SingularMonitoredSession // See docs
tf.train.Scaffold // See docs
tf.train.SessionCreator // See docs
tf.train.ChiefSessionCreator // See docs
tf.train.WorkerSessionCreator // See docs
Reading Summaries from Event Files // See docs
See Summaries and TensorBoard for an overview of summaries, event files, and visualization in TensorBoard.

tf.train.summary_iterator
Training Hooks
Hooks are tools that run in the process of training/evaluation of the model.

tf.train.SessionRunHook // See docs
tf.train.SessionRunArgs // See docs
tf.train.SessionRunContext // See docs
tf.train.SessionRunValues // See docs
tf.train.LoggingTensorHook // See docs
tf.train.StopAtStepHook // See docs
tf.train.CheckpointSaverHook // See docs
tf.train.NewCheckpointReader // See docs
tf.train.StepCounterHook // See docs
tf.train.NanLossDuringTrainingError // See docs
tf.train.NanTensorHook // See docs
tf.train.SummarySaverHook // See docs
tf.train.GlobalStepWaiterHook // See docs
tf.train.FinalOpsHook // See docs
tf.train.FeedFnHook // See docs
Training Utilities // See docs
tf.train.global_step // See docs
tf.train.basic_train_loop // See docs
tf.train.get_global_step // See docs
tf.train.assert_global_step // See docs
tf.train.write_graph // See docs

----CONTRIB----

Splitting sequence inputs into minibatches with state saving 
Use tf.contrib.training.SequenceQueueingStateSaver or its wrapper tf.contrib.training.batch_sequences_with_states if you have input data with a dynamic primary time / frame count axis which you'd like to convert into fixed size segments during minibatching, and would like to store state in the forward direction across segments of an example. // 
* tf.contrib.training.batch_sequences_with_states(input_key,input_sequences,input_context,input_length,initial_states,num_unroll,batch_size,num_threads=3,capacity=1000,allow_small_batch=True,pad=True,make_keys_unique=False,make_keys_unique_seed=None,name=None) // 
	// Creates batches from segments of sequential input
* tf.contrib.training.NextQueuedSequenceBatch //  CLASS stores a deffered sequenceQueuingStateSaver's data
* tf.contrib.training.SequenceQueueingStateSaver // CLASS is used instead of a queue to split variable length sequences into segments of sequences with fixed length. Batches into mini-batches
Online data resampling
To resample data with replacement on a per-example basis, use tf.contrib.training.rejection_sample or tf.contrib.training.resample_at_rate. 
For rejection_sample, provide a boolean Tensor describing whether to accept or reject. Resulting batch sizes are always the same. 
For resample_at_rate, provide the desired rate for each example. Resulting batch sizes may vary. 
If you wish to specify relative rates, rather than absolute ones, use tf.contrib.training.weighted_resample (which also returns the actual resampling rate used for each output example). // 

Use tf.contrib.training.stratified_sample to resample without replacement from the data to achieve a desired mix of class proportions that the Tensorflow graph sees. For instance, if you have a binary classification dataset that is 99.9% class 1, a common approach is to resample from the data so that the data is more balanced.  
* tf.contrib.training.rejection_sample(tensors,accept_prob_fn,batch_size,queue_threads=1,enqueue_many=False,prebatch_capacity=16,prebatch_threads=1,runtime_checks=False,name=None) 
	// Creates batches by rejecting samples not accepted by a function
* tf.contrib.training.resample_at_rate(inputs,rates,scope=None,seed=None,back_prop=False) 
	// 
* tf.contrib.training.stratified_sample(tensors,labels,target_probs,batch_size,init_probs=None,enqueue_many=False,queue_capacity=16,threads_per_queue=1,name=None)
	// Resamples inputs at a rate returning a new resampled set
* tf.contrib.training.weighted_resample(inputs,weights,overall_rate,scope=None,mean_decay=0.999,seed=None) 
	// Creates batches based on probabilities
Bucketing
Use tf.contrib.training.bucket or tf.contrib.training.bucket_by_sequence_length to stratify minibatches into groups ("buckets"). 
Use bucket_by_sequence_length with the argument dynamic_pad=True to receive minibatches of similarly sized sequences for efficient training via dynamic_rnn. 
* tf.contrib.training.bucket(tensors,which_bucket,batch_size,num_buckets,num_threads=1,capacity=32,bucket_capacities=None,shapes=None,dynamic_pad=False,allow_smaller_final_batch=False,keep_input=True,shared_name=None,name=None) // 
	// An aproximate weighted resampling of inputs. Choses inputs where rate of selection is proportional to weights.
* tf.contrib.training.bucket_by_sequence_length(input_length,tensors,batch_size,bucket_boundaries,num_threads=1,capacity=32,bucket_capacities=None,shapes=None,dynamic_pad=False,allow_smaller_final_batch=False,keep_input=True,shared_name=None,name=None) //
	// Lazy bucketing of inputs according to their length. Calls tf.contrib.training.bucket and after subdividing bucket boundries identifies what bucks an input_length belongs to and uses that.