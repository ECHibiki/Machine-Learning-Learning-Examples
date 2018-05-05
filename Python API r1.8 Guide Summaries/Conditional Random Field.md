Conditional Random Fields(CRF) are a class of statstical modelling methods for patern learnign and machine learning using structered predictions. Linear Chain CRF(Natural Language Processsing) predicts sequences of labels for sequence of input samples.
Linear-chain CRF layer.

* tf.contrib.crf.crf_sequence_score(inputs, tag_indices, sequence_lengths, transition_params) // Computes an unnormalized score for a tag sequence
* tf.contrib.crf.crf_log_norm(inputs, sequence_lengths, transition_params) //Computes normalization for a CRF
* tf.contrib.crf.crf_log_likelihood( inputs, tag_indices, sequence_lengths, transition_params=None) // Computes log likelihood of a tag sequence in a CRF
* tf.contrib.crf.crf_unary_score( tag_indices,  sequence_lengths,  inputs) // Finds the unary scores of tag sequences
* tf.contrib.crf.crf_binary_score( tag_indices,  sequence_lengths,  transition_params) // Finds binary scores of tag sequnce
* tf.contrib.crf.CrfForwardRnnCell : Class Inherits From: RNNCell[Every RNNCell must have the properties below and implement call with the signature (output, next_state) = call(input, state). The optional third input argument, scope, is allowed for backwards compatibility purposes; but should be left off for new subclasses.]
	* Properties:
		* activity_regularizer: Optional regularizer for the output of layer
		* dtype 
		* graph
		* input: Retrieves the input tensor of a CRF layer iff the layer has one input/is connected to one incoming layer
		* input_shape : Gets teh input shape of a lwayer iff one input/connectd to one incoming layer or all inputs of same shape.
		* name
		* non_trainable_variables
		* non_trainable_weights
		* output : Gets output tensors of a layer. 
		* output_shape : Get output shapes of a layer
		* output_size :  Integer or TenshorShape size of outputs
		* Scoppe_name
		* state_size : size of states used by cell
		* trainable_varaibles
		* trainable_weights
		* updates
		* variables : list of all layer variables and weights
		* weights : returns all varibales and weights
	* Methods:
		* __init__(transition_params) : Initializes a CrfForwardRnnCell
		* __call__(inputs,state,scope=None) : Builds a CrfFrowardRnnCell
		* __deepcopy__(memo)
		* add_loss(losses,inputs=None) : Adds loss tensors that maybe pepend on layer inputs.
		* add_update(updates,inputs=None) : Adds updates 
		* add_variable(name,shape,dtype=None,initializer=None,regularizer=None,trainable=True,constraint=None, partitioner=None) :  Adds another variable and returns it.
		* apply(inputs,  *args,  **kwargs) : Applies a layer to an input
		* build(_) : Creates the variables of the layer.
		* call(inputs, **kwargs) : Causes logic of layer to be executed
		* compute_output_shape(input_shape) : Finds the output shape of a layer from a certain shape
		* count_params() : Finds all the scalars composing the weights
		* get_input_at(node_index) : Finds the input of layers at a node
		* get_input_shape_at(node_index) : Finds the input shape of a layer at a node
		* get_losses_for(inputs) : Losses of inputs
		* get_output_at(node_index) : Output tensors of a layer at a node
		* get_output_shape_at(node_index) : Shape of a layer at node
		* get_updates_for(inputs) : Find updates from a set of inputs
		* zero_state(batch_size,dtype) : gets zero-filled state tesnors
* tf.contrib.crf.viterbi_decode(score, transition_params) // Test function to decode the highest scoring sequence of tags outseide of the tensorflow environment
