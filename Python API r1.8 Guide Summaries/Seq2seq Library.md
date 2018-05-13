Seq2seq Library
Module for constructing seq2seq models and dynamic decoding. Builds on top of libraries in tf.contrib.rnn.
Attention wrappers are RNNCell objects that wrap other RNNCell objects and implement attention.
The two basic attention mechanisms are: tf.contrib.seq2seq.BahdanauAttention (additive attention, ref.) tf.contrib.seq2seq.LuongAttention (multiplicative attention, ref.)

Decoder base class and functions : 
 * tf.contrib.seq2seq.Decoder // See docs
 * tf.contrib.seq2seq.dynamic_decode(decoder,output_time_major=False,impute_finished=False,maximum_iterations=None,parallel_iterations=32,swap_memory=False,scope=None) 
	// Dynamic decoding
Basic Decoder
 * tf.contrib.seq2seq.BasicDecoderOutput // See docs 
 * tf.contrib.seq2seq.BasicDecoder  // See docs
Decoder Helpers
 * tf.contrib.seq2seq.Helper  // See docs 
 * tf.contrib.seq2seq.CustomHelper  // See docs
 * tf.contrib.seq2seq.GreedyEmbeddingHelper  // See docs
 * tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper  // See docs
 * tf.contrib.seq2seq.ScheduledOutputTrainingHelper  // See docs
 * tf.contrib.seq2seq.TrainingHelper  // See docs
