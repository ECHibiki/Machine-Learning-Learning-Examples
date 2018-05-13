Arithmetic Operators
TensorFlow provides several operations that you can use to add basic arithmetic operators to your graph.

* tf.add // 
* tf.subtract // 
* tf.multiply // 
* tf.scalar_mul // 
* tf.div(x,y,name=None) // Divides x / y elementwise (using Python 2 division operator semantics). returns the quotient of x and y.
* tf.divide(x,y,name=None) // Computes Python style division of x by y.
* tf.truediv // x / y evaluated in floating point.
* tf.floordiv // 
* tf.realdiv // Returns x / y element-wise Tensor. Has the same type as x.
* tf.truncatediv // Returns x / y element-wise Tensor.Truncation designates that negative numbers will round fractional quantities toward zero. 
* tf.floor_div // 
* tf.truncatemod // 
* tf.floormod // 
* tf.mod // 
* tf.cross //  Pairwise crossproduct
Basic Math Functions
TensorFlow provides several operations that you can use to add basic mathematical functions to your graph.

* tf.add_n // Adds all inputs together returning A Tensor of same shape and type as the elements of inputs.
* tf.abs // 
* tf.negative // 
* tf.sign // 
* tf.reciprocal // 1/x tensor 
* tf.square // 
* tf.round // 
* tf.sqrt // 
* tf.rsqrt // 
* tf.pow // 
* tf.exp // exponent of x
* tf.expm1 // exponential of x - 1 
* tf.log // natural logarithm of x 
* tf.log1p // natural logarithm of (1 + x) 
* tf.ceil // 
* tf.floor // 
* tf.maximum // 
* tf.minimum // 
* tf.cos // 
* tf.sin // 
* tf.lbeta // 
* tf.tan // 
* tf.acos // 
* tf.asin // 
* tf.atan // 
* tf.cosh // 
* tf.sinh // 
* tf.asinh // 
* tf.acosh // 
* tf.atanh // 
* tf.lgamma // log of the absolute value of Gamma(x) element-wise.
* tf.digamma // derivative of Lgamma
* tf.erf // Gauss error function of x
* tf.erfc // complementary error function of x element-wise.
* tf.squared_difference // 
* tf.igamma // Compute the lower regularized incomplete Gamma function P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)
* tf.igammac //  incomplete Gamma function  Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)
* tf.zeta // Compute the Hurwitz zeta function
* tf.polygamma // Compute the polygamma function
* tf.betainc // Compute the regularized incomplete beta integral
* tf.rint //rint(-1.5) ==> -2.0
			rint(0.5000001) ==> 1.0
			rint([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
Matrix Math Functions
TensorFlow provides several operations that you can use to add linear algebra functions on matrices to your graph.

* tf.diag // 
* tf.diag_part // 
* tf.trace // 
* tf.transpose // 
* tf.eye // 
* tf.matrix_diag // 
* tf.matrix_diag_part // 
* tf.matrix_band_part // 
* tf.matrix_set_diag // 
* tf.matrix_transpose // 
* tf.matmul // 
* tf.norm // 
* tf.matrix_determinant // 
* tf.matrix_inverse // 
* tf.cholesky // 
* tf.cholesky_solve // 
* tf.matrix_solve // 
* tf.matrix_triangular_solve // 
* tf.matrix_solve_ls // 
* tf.qr // 
* tf.self_adjoint_eig // 
* tf.self_adjoint_eigvals // 
* tf.svd // Computes the singular value decompositions
* Tensor Math Function
TensorFlow provides operations that you can use to add tensor functions to your graph.
 
* tf.tensordot //
Complex Number Functions
TensorFlow provides several operations that you can use to add complex number functions to your graph.
 
* tf.complex // 
* tf.conj // 
* tf.imag // 
* tf.angle // 
* tf.real // 
Reduction
TensorFlow provides several operations that you can use to perform common math computations that reduce various dimensions of a tensor.

* tf.reduce_sum // 
* tf.reduce_prod // 
* tf.reduce_min // 
* tf.reduce_max // 
* tf.reduce_mean // 
* tf.reduce_all // 
* tf.reduce_any // 
* tf.reduce_logsumexp // 
* tf.count_nonzero // 
* tf.accumulate_n // 
* tf.einsum //
Scan
TensorFlow provides several operations that you can use to perform scans (running totals) across one axis of a tensor.

tf.cumsum
tf.cumprod
Segmentation
TensorFlow provides several operations that you can use to perform common math computations on tensor segments. Here a segmentation is a partitioning of a tensor along the first dimension, i.e. it defines a mapping from the first dimension onto segment_ids. The segment_ids tensor should be the size of the first dimension, d0, with consecutive IDs in the range 0 to k, where k<d0. In particular, a segmentation of a matrix tensor is a mapping of rows to segments.

For example:

c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
tf.segment_sum(c, tf.constant([0, 0, 1]))
  ==>  [[0 0 0 0]
        [5 6 7 8]]
tf.segment_sum // 
* tf.segment_prod // 
* tf.segment_min // 
* tf.segment_max // 
* tf.segment_mean // 
* tf.unsorted_segment_sum // 
* tf.sparse_segment_sum // 
* tf.sparse_segment_mean // 
* tf.sparse_segment_sqrt_n
Sequence Comparison and Indexing
TensorFlow provides several operations that you can use to add sequence comparison and index extraction to your graph. You can use these operations to determine sequence differences and determine the indexes of specific values in a tensor.

tf.argmin
tf.argmax
tf.setdiff1d
tf.where
tf.unique
tf.edit_distance
tf.invert_permutation