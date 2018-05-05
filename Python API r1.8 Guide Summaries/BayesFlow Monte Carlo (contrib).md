BayesFlow Monte Carlo (contrib)
Practice of estimation based on a sample mean. Bayesian statistics in otherwords.

tf.contrib.bayesflow.monte_carlo.expectation(f,samples, log_prob=None, use_reparametrization=True, axis=0, keep_dims=False, name=None)
// Finds monte-carlo approximation.
// P needs to be a reparametrized distributions
tf.contrib.bayesflow.monte_carlo.expectation_importance_sampler(f, log_p, sampling_dist_q, z=None, n=None, seed=None, name='expectation_importance_sampler')
// A montecarlo estimate returning an integral done in log-space
tf.contrib.bayesflow.monte_carlo.expectation_importance_sampler_logspace(log_f, log_p, sampling_dist_q, z=None, n=None, seed=None, name='expectation_importance_sampler_logspace')
//Importance sampling with a positive function, in log-space.