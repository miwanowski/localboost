import numpy as np

from sklearn.tree import DecisionTreeClassifier
from math import log, exp

class AdaBoostEnsemble(object):
	"""An implementation of discreet AdaBoost algorithm for binary classification."""
	def __init__(self, X, y, base_classifier_args={}, ensemble_args={}, base_classifier=DecisionTreeClassifier):
		self.classifiers = []
		self.base_classifier = base_classifier
		self.X = X
		self.y = y
		self.n_iters = 0
		self.base_classifier_args = base_classifier_args
		self.ensemble_args = ensemble_args
		n_examples = y.shape[0]
		n_plus = sum(y == 1)
		ratio = float(n_plus)/(n_examples-n_plus)
		self.training_weights = np.ones(y.shape[0])
		self.training_weights[y == -1] = ratio
		self.training_weights /= sum(self.training_weights)
		self.classifier_weights = np.ndarray((0,))
		self.__dict__.update(ensemble_args)

	def fit(self, n_iters):
		"""Iteratively add a given number of classifiers to the ensemble."""
		for i in xrange(n_iters):
			# create a new base model:
			default_base_args = {'max_depth': 1}
			base_model = self.base_classifier(**(dict(default_base_args.items() + self.base_classifier_args.items())))
			base_model.fit(X=self.X, y=self.y, sample_weight=self.training_weights)

			# evaluate the new model's accuracy:
			base_prediction = base_model.predict(self.X)
			correctly_classified = base_prediction == self.y
			incorrectly_classified = ~correctly_classified
			base_error_rate = sum(self.training_weights[incorrectly_classified])
			if base_error_rate == 0:
				base_error_rate = 0.001
			alpha = 0.5*log((1.0-base_error_rate)/(base_error_rate))

			# add new model to the ensemble:
			self.classifiers.append(base_model)
			self.classifier_weights = np.append(self.classifier_weights, alpha)

			# update training weights:
			self.training_weights[correctly_classified] *= exp(-alpha)
			self.training_weights[incorrectly_classified] *= exp(alpha)

			# normalize weight distribution:
			self.training_weights /= sum(self.training_weights)

		self.n_iters += n_iters

	def predict(self, test_data):
		"""Use the model to predict class labels on a test data set."""
		prediction = np.zeros(test_data.shape[0])
		for i in xrange(self.n_iters):
			p = self.classifiers[i].predict(test_data)
			prediction += p * self.classifier_weights[i]
		return np.sign(prediction)