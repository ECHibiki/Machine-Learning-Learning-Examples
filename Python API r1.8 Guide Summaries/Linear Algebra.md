Linear Algebra
Subclasses of LinearOperator provide a access to common methods on a (batch) matrix, without the need to materialize the matrix. This allows:

Base class
* tf.contrib.linalg.LinearOperator // Base class defining a [batch of] linear operator[s].
	* Parameters : 
		* batch_shape : TensorShape of batch dimensions of this LinearOperator.
		* domain_dimension : Dimension (in the sense of vector spaces) of the domain of this operator.
		* dtype : The DType of Tensors handled by this LinearOperator.
		* graph_parents : List of graph dependencies of this LinearOperator.
		* is_non_singular
		* is_positive_definite
		* is_self_adjoint
		* is_square : Return True/False depending on if this operator is square.
		* name : Name prepended to all ops created by this LinearOperator.
		* range_dimension : Dimension (in the sense of vector spaces) of the range of this operator.
		* shape : TensorShape of this LinearOperator.
		* tensor_rank : Rank (in the sense of tensors) of matrix corresponding to this operator.

	* Methods : 
		* __init__(dtype,graph_parents=None,is_non_singular=None,is_self_adjoint=None,is_positive_definite=None,is_square=None,name=None)
			: initializes class
		* add_to_tensor(x,name='add_to_tensor')
			: Adds matrix to x [A + x]
		* assert_non_singular(name='assert_non_singular')
			: Return an Op that asserts is not singular
		* assert_positive_definite(name='assert_positive_definite')
			: Returns an Op that asserts this operator is positive definite.
		* assert_self_adjoint(name='assert_self_adjoint')
			: Returns an Operator that asserts self-adjoin
		* batch_shape_tensor(name='batch_shape_tensor')
			: Shape of a batch of this Operator
		* determinant(name='det')
			: Determinant of every batch
		* diag_part(name='diag_part')
			:  Efficiently get the [batch] diagonal part of this operator.
		* domain_dimension_tensor(name='domain_dimension_tensor')
			: Dimension (in the sense of vector spaces) of the domain of this operator.
		* log_abs_determinant(name='log_abs_det')
			: Log absolute value of determinant for every batch member.
		* matmul(x,adjoint=False,adjoint_arg=False,name='matmul')
			: Transform [batch] matrix x with left multiplication: x --> Ax.
		* matvec(x,adjoint=False,name='matvec')
			: Transform [batch] vector x with left multiplication: x --> Ax.
		* range_dimension_tensor(name='range_dimension_tensor')
			: Dimension (in the sense of vector spaces) of the range of this operator. Determined at runtime.
		* shape_tensor(name='shape_tensor').
			: Shape of this LinearOperator, determined at runtime.
		* solve(rhs,adjoint=False,adjoint_arg=False,name='solve')
			: Solve (exact or approx) R (batch) systems of equations: A X = rhs.
		* solvevec(rhs,adjoint=False,name='solve')
			: Solve single equation with best effort: A X = rhs.
		* tensor_rank_tensor(name='tensor_rank_tensor')
			: Rank (in the sense of tensors) of matrix corresponding to this operator.
		* to_dense(name='to_dense')
			: Return a dense (batch) matrix representing this operator.
		* trace(name='trace')
			: Trace of the linear operator, equal to sum of self.diag_part().

Individual operators : A bunch of derivitave classes
	* tf.contrib.linalg.LinearOperatorDiag // A LinearOperator acting like a [batch] square diagonal matrix.
		* Properties :  See Documentation
		* Methods : See Documentation
	* tf.contrib.linalg.LinearOperatorIdentity // LinearOperator acting like a [batch] square identity matrix.
		* Properties :  See Documentation
		* Methods : See Documentation
	* tf.contrib.linalg.LinearOperatorScaledIdentity // LinearOperator  acting like a scaled [batch] identity matrix A = c I..
		* Properties :  See Documentation
		* Methods : See Documentation
	* tf.contrib.linalg.LinearOperatorFullMatrix // LinearOperator that wraps a [batch] matrix.
		* Properties :  See Documentation
		* Methods : See Documentation
	* tf.contrib.linalg.LinearOperatorLowerTriangular // LinearOperator acting like a [batch] square lower triangular matrix.
		* Properties :  See Documentation
		* Methods : See Documentation
	* tf.contrib.linalg.LinearOperatorLowRankUpdate // Perturb a LinearOperator with a rank K update.
		* Properties :  See Documentation
		* Methods : See Documentation
	Transformations and Combinations of operators
	* tf.contrib.linalg.LinearOperatorComposition // Composes one or more LinearOperators.
		* Properties :  See Documentation
		* Methods : See Documentation