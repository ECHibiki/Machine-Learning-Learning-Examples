RNN and Cells 

Base interface for all RNN Cells
 * tf.contrib.rnn.RNNCell //  CLASS from Layer.  Abstract object representing an RNN cell.
Core RNN Cells for use with TensorFlow's core RNN methods
 * tf.contrib.rnn.BasicRNNCell // CLASS from LayerRNNCell. The most basic RNN cell.
 * tf.contrib.rnn.BasicLSTMCell //CLASS from LayerRNNCell.  Basic LSTM recurrent network cell.
 * tf.contrib.rnn.GRUCell // CLASS from LayerRNNCell. Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
 * tf.contrib.rnn.LSTMCell // CLASS from LayerRNNCell.The default non-peephole implementation is based on: http://www.bioinf.jku.at/publications/older/2604.pdf . The peephole implementation is based on: https://research.google.com/pubs/archive/43905.pdf
 * tf.contrib.rnn.LayerNormBasicLSTMCell //  CLASS from RNNCell. class adds layer normalization and recurrent dropout to a basic LSTM unit.  https://arxiv.org/abs/1607.06450. https://arxiv.org/abs/1603.05118
Classes storing split `RNNCell` state
 * tf.contrib.rnn.LSTMStateTuple // CLASS. Stores two elements: (c, h), in that order. Where c is the hidden state and h is the output.
Core RNN Cell wrappers (RNNCells that wrap other RNNCells)
 * tf.contrib.rnn.MultiRNNCell //  CLASS from RNNCell. RNN cell composed sequentially of multiple simple cells.
 * tf.contrib.rnn.LSTMBlockWrapper // CLASS from Layer.  helper class that provides housekeeping for LSTM cells.
 * tf.contrib.rnn.DropoutWrapper // CLASS from RNNCell. adding dropout to inputs and outputs of the given cell.
 * tf.contrib.rnn.EmbeddingWrapper // CLASS from RNNCell.  adding input embedding to the given cell.
 * tf.contrib.rnn.InputProjectionWrapper //  CLASS from RNNCell. adding an input projection to the given cell.
 * tf.contrib.rnn.OutputProjectionWrapper //  CLASS from RNNCell.adding an output projection to the given cell.
 * tf.contrib.rnn.DeviceWrapper //  CLASS from RNNCell. ensures an RNNCell runs on a particular device.
 * tf.contrib.rnn.ResidualWrapper //  CLASS from RNNCell. ensures cell inputs are added to the outputs.
Block RNNCells
 * tf.contrib.rnn.LSTMBlockCell //CLASS from LayerRNNCell. Basic LSTM recurrent network cell.
 * tf.contrib.rnn.GRUBlockCell // CLASS from LayerRNNCell. Deprecated: use GRUBlockCellV2 instead.
 * tf.contrib.rnn.GRUBlockCellV2 // Only differs from GRUBlockCell by variable names.
Fused RNNCells
 * tf.contrib.rnn.FusedRNNCell // A fused RNN cell represents the entire RNN expanded over the time dimension. In effect, this represents an entire recurrent network.
 * tf.contrib.rnn.FusedRNNCellAdaptor // CLASS from FusedRNNCell. an adaptor for RNNCell classes to be used with FusedRNNCell.
 * tf.contrib.rnn.TimeReversedFusedRNN //CLASS from FusedRNNCell.  an adaptor to time-reverse a FusedRNNCell.
 * tf.contrib.rnn.LSTMBlockFusedCell //CLASS from FusedRNNCell.  extremely efficient LSTM implementation, that uses a single TF op for the entire LSTM.
LSTM-like cells
 * tf.contrib.rnn.CoupledInputForgetGateLSTMCell // CLASS from RNNCell. Lots of articles on the docs page
 * tf.contrib.rnn.TimeFreqLSTMCell // CLASS from RNNCell. Lots of articles on the docs page
 * tf.contrib.rnn.GridLSTMCell // CLASS from RNNCell. Lots of articles on the docs page
RNNCell wrappers
 * tf.contrib.rnn.AttentionCellWrapper // CLASS from RNNCell. https://arxiv.org/abs/1409.0473.
 * tf.contrib.rnn.CompiledWrapper //CLASS from RNNCell.  execution in an XLA JIT scope.
Recurrent Neural Networks
TensorFlow provides a number of methods for constructing Recurrent Neural Networks. 
* tf.contrib.rnn.static_rnn(cell,inputs,initial_state=None,dtype=None,sequence_length=None,scope=None) 
	// Create a recurrent neural network based on RNNCell. Various methods to setup
* tf.contrib.rnn.static_state_saving_rnn(cell,inputs,state_saver,state_name,sequence_length=None,scope=None) 
	// RNN accepting state save for time-truncated RNN
* tf.contrib.rnn.static_bidirectional_rnn(cell_fw,cell_bw,inputs,initial_state_fw=None,initial_state_bw=None,dtype=None,sequence_length=None,scope=None) 
	// Bidirectional neural network. Takes input and builds independent fore and back RNNs
* tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw,cells_bw,inputs,initial_states_fw=None,initial_states_bw=None,dtype=None,sequence_length=None,parallel_iterations=None,time_major=False,scope=None) 
	// dynamic bidirectional recurrent neural network. Combined fore and back are used in the next layer