Random variable transformations (contrib):
An API for invertible differntiable tranformations of random variables

Bijectors:

* tf.contrib.distributions.bijectors.Affine Inherits from Bijector[Bijectors can be used to represent any differentiable and injective (one to one) function defined on an open subset of R^n. Some non-injective transformations are also supported (see "Non Injective Transforms" below).]
	* Properties :
		* dtype : tensors transformable by distribution
		* event_ndims : Returns number of event dimensions in bijector
		* graph_parents : Returns bijector graph_parents as list
		* is_constant_jacobian : returns iff Jacobian is not an f of x
		* name : name of bijector
		* scale : scale LinearOperator in Y = scale @ X + shift.
		* shift : The shift Tensor in Y = scale @ X + shift.
		* validate_args : Returns True if Tensor arguments will be validated.
	* Methods :
		* __init__(shift=None,scale_identity_multiplier=None,scale_diag=None,scale_tril=None,scale_perturb_factor=None,scale_perturb_diag=None,validate_args=False,name='affine') 
			: Instantiates Affine class
		* forward(x,name='forward')
			: Returns forward bijector evaluation
		* forward_event_shape(input_shape)
			: Shape of a sample from a batch as a TensorShape
		* forward_event_shape_tensor(input_shape,name='forward_event_shape_tensor')
			: Shape of a sample from a single batch(int32 1D tensor)
		* forward_log_det_jacobian(x,name='forward_log_det_jacobian')
			: Returns two forward_log_det_jacobian
		* inverse(y,name='inverse')
			: Returns inverse Bijector evaluation.
		* inverse_event_shape(output_shape)
			: Shape of a single sample from a batch as TensorShape
		* inverse_event_shape_tensor(output_shape,name='inverse_event_shape_tensor')
			: Shape of sample from batch as int32 1D tensor
		* inverse_log_det_jacobian(y,name='inverse_log_det_jacobian')
			: Returns log(det(dX/dY))(Y)
* tf.contrib.distributions.bijectors.AffineLinearOperator
	* Read Documentation
* tf.contrib.distributions.bijectors.Bijector 
// Bijectors are smooth covering maps used by TransformeedDistribution to transform a tensor generated distribution.
// Jacobians are reductions over event dims
	* Properties :
		* dtype digit type of Tensors transformable by this distribution.
		* event_ndims : Number of event dimentsions
		* graph_parents : Bijectors graph parents as a list.
		* is_constant_jacobian : iff the jacobian is not f of x
		* name : Returns the string name of this Bijector
		* validate_args : true if tensor arguments will be validated.
	* Methods : 
		* __init__( event_ndims=None,graph_parents=None,is_constant_jacobian=False,validate_args=False,dtype=None,name=None) : Create a bijector to transform random variables to new randoms
		* forward(x,name='forward') : Returns forward bijector evaluation X = g(Y)
		* forward_event_shape(input_shape) : Shape of a single sample from a single batch(TensorShape) : Shape of a sample from single batch
		* forward_event_shape_tensor(input_shape,name='forward_event_shape_tensor')  : Shape of a single sample from a single batch as an int32 1D Tensor.
		* forward_log_det_jacobian(x,name='forward_log_det_jacobian') : returns both forward_log_det_jacobian
		* inverse(y,name='inverse') :  Returns inverse bijector   i.e., X = g^{-1}(Y).
		* inverse_event_shape(output_shape)	 : Shape of sample from single batch as TensorShape
		* inverse_event_shape_tensor(output_shape,name='inverse_event_shape_tensor')	: Shape of sample as in32 Tensor
		* inverse_log_det_jacobian(y,name='inverse_log_det_jacobian') : Mathematically, returns: log(det(dX/dY))(Y)
	
* tf.contrib.distributions.bijectors.Chain // A bijector applied to a sequence of bijectors
	* Properties : Same as bijector 
	* Methods : Same as bijector
* tf.contrib.distributions.bijectors.CholeskyOuterProduct
	* Read Documentation
* tf.contrib.distributions.bijectors.Exp
	* Read Documentation
* tf.contrib.distributions.bijectors.Identity
	* Read Documentation
* tf.contrib.distributions.bijectors.Inline
	* Read Documentation
* tf.contrib.distributions.bijectors.Invert
	* Read Documentation
* tf.contrib.distributions.bijectors.PowerTransform
	* Read Documentation
* tf.contrib.distributions.bijectors.SoftmaxCentered // The bijector computering Y = g(X) = exp([X 0]) / sum(exp([X 0])).
	* Properties : Same as bijector
	* Methods : Same as bijector
* tf.contrib.distributions.bijectors.Softplus // Bijector computing  Y = g(X) = Log[1 + exp(X)]
												The softplus Bijector has the following two useful properties:

												The domain is the positive real numbers
												softplus(x) approx x, for large x, so it does not overflow as easily as the Exp Bijector.
	* Properties :Same as bijector
	* Methods :  Same as bijector