Neural Network

Activation Functions
The activation ops provide different types of nonlinearities for use in neural networks. These include smooth nonlinearities (sigmoid, tanh, elu, selu, softplus, and softsign), continuous but not everywhere differentiable functions (relu, relu6, crelu and relu_x), and random regularization (dropout).

All activation ops apply componentwise, and produce a tensor of the same shape as the input tensor.

tf.nn.relu // max(features, 0).
tf.nn.relu6 // min(max(features, 0), 6).
tf.nn.crelu // An improvement on relu selects only the negative part of the activation. https://arxiv.org/abs/1603.05201
tf.nn.elu // exp(features) - 1  if < 0, features https://arxiv.org/abs/1511.07289
tf.nn.selu // Scaled exponential linear: scale * alpha * (exp(features) - 1) if < 0, scale * feature
tf.nn.softplus //  log(exp(features) + 1).
tf.nn.softsign //  features / (abs(features) + 1).
tf.nn.dropout(x,keep_prob,noise_shape=None,seed=None,name=None) // computes the droupout 
tf.nn.bias_add // add where bias is restricted to 1D
tf.sigmoid // Specifically, y = 1 / (1 + exp(-x)).
tf.tanh // hyperbolic tangent of x element-wise.
Convolution
Sweeps 2D filter over batch. Applies filter to each window of each image of size.
They are cross corelations since filter combines with input window. Filter applies to image patches of filter size.

tf.nn.convolution
tf.nn.conv2d // Computes sums of nd convolutions. Also supports strdes
tf.nn.depthwise_conv2d // Depthwise 2-D convolution.
tf.nn.depthwise_conv2d_native // 
tf.nn.separable_conv2d
tf.nn.atrous_conv2d
tf.nn.atrous_conv2d_transpose
tf.nn.conv2d_transpose
tf.nn.conv1d
tf.nn.conv3d
tf.nn.conv3d_transpose
tf.nn.conv2d_backprop_filter
tf.nn.conv2d_backprop_input
tf.nn.conv3d_backprop_filter_v2
tf.nn.depthwise_conv2d_native_backprop_filter
tf.nn.depthwise_conv2d_native_backprop_input

Pooling
Pooling sweeps window over tensor finds reduction operation on each window. output[i] = reduce(value[strides * i:strides * i + ksize])

tf.nn.avg_pool
tf.nn.max_pool
tf.nn.max_pool_with_argmax
tf.nn.avg_pool3d
tf.nn.max_pool3d
tf.nn.fractional_avg_pool
tf.nn.fractional_max_pool
tf.nn.pool
Morphological filtering
Morphological operators are non-linear filters used in image processing.

Greyscale morphological dilation is the max-sum counterpart of standard sum-product convolution:

The filter is usually called structuring function. Max-pooling is a special case of greyscale morphological dilation when the filter assumes all-zero values (a.k.a. flat structuring function).

Greyscale morphological erosion is the min-sum counterpart of standard sum-product convolution:

Dilation and erosion are dual to each other. The dilation of the input signal f by the structuring signal g is equal to the negation of the erosion of -f by the reflected g, and vice versa.

Striding and padding is carried out in exactly the same way as in standard convolution. Please refer to the Convolution section for details.

tf.nn.dilation2d
tf.nn.erosion2d
tf.nn.with_space_to_batch
Normalization
Normalization is useful to prevent neurons from saturating when inputs may have varying scale, and to aid generalization.

tf.nn.l2_normalize
tf.nn.local_response_normalization
tf.nn.sufficient_statistics
tf.nn.normalize_moments
tf.nn.moments
tf.nn.weighted_moments
tf.nn.fused_batch_norm
tf.nn.batch_normalization
tf.nn.batch_norm_with_global_normalization
Losses
The loss ops measure error between two tensors, or between a tensor and zero. These can be used for measuring accuracy of a network in a regression task or for regularization purposes (weight decay).

tf.nn.l2_loss
tf.nn.log_poisson_loss
Classification
TensorFlow provides several operations that help you perform classification.

tf.nn.sigmoid_cross_entropy_with_logits
tf.nn.softmax
tf.nn.log_softmax
tf.nn.softmax_cross_entropy_with_logits
tf.nn.softmax_cross_entropy_with_logits_v2 - identical to the base version, except it allows gradient propagation into the labels.
tf.nn.sparse_softmax_cross_entropy_with_logits
tf.nn.weighted_cross_entropy_with_logits
Embeddings
TensorFlow provides library support for looking up values in embedding tensors.

tf.nn.embedding_lookup
tf.nn.embedding_lookup_sparse
Recurrent Neural Networks
TensorFlow provides a number of methods for constructing Recurrent Neural Networks. Most accept an RNNCell-subclassed object (see the documentation for tf.contrib.rnn).

tf.nn.dynamic_rnn
tf.nn.bidirectional_dynamic_rnn
tf.nn.raw_rnn
Connectionist Temporal Classification (CTC)
tf.nn.ctc_loss
tf.nn.ctc_greedy_decoder
tf.nn.ctc_beam_search_decoder
Evaluation
The evaluation ops are useful for measuring the performance of a network. They are typically used at evaluation time.

tf.nn.top_k
tf.nn.in_top_k
Candidate Sampling
Do you want to train a multiclass or multilabel model with thousands or millions of output classes (for example, a language model with a large vocabulary)? Training with a full Softmax is slow in this case, since all of the classes are evaluated for every training example. Candidate Sampling training algorithms can speed up your step times by only considering a small randomly-chosen subset of contrastive classes (called candidates) for each batch of training examples.

See our Candidate Sampling Algorithms Reference

Sampled Loss Functions
TensorFlow provides the following sampled loss functions for faster training.

tf.nn.nce_loss
tf.nn.sampled_softmax_loss
Candidate Samplers
TensorFlow provides the following samplers for randomly sampling candidate classes when using one of the sampled loss functions above.

tf.nn.uniform_candidate_sampler
tf.nn.log_uniform_candidate_sampler
tf.nn.learned_unigram_candidate_sampler
tf.nn.fixed_unigram_candidate_sampler
Miscellaneous candidate sampling utilities
tf.nn.compute_accidental_hits
Quantization ops
tf.nn.quantized_conv2d
tf.nn.quantized_relu_x
tf.nn.quantized_max_pool
tf.nn.quantized_avg_pool