import numpy as np

class AdaBoost(object):
	"""An implementation of discreet AdaBoost algorithm for binary classification."""
	def __init__(self, base_classifier, X, y, classifier_opt_args = None):
		self.base_classifier = base_classifier
		self.X = X
		self.y = y

	def fit(self, n_iters):
		"""Iteratively add a given number of classifiers to the ensemble."""
		self.n_iters += n_iters

	n_iters 				= 0
	classifiers 			= []
	training_weights 		= []
	classifier_weights 		= []