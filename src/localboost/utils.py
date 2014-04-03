import numpy as np
import matplotlib.pyplot as plt

from math import ceil

def trainTestSplit(dataset, train_ratio=0.6):
    """Spit a dataset into a training and test part according to a given ratio."""
    n_total = int(dataset.shape[0])
    n_train = int(ceil(train_ratio*n_total))
    train_mask = np.ndarray((n_total,), dtype=bool)
    train_mask.fill(False)
    train_mask[np.random.choice(n_total, n_train, replace=False)] = True
    test_mask = ~train_mask
    return dataset[train_mask], dataset[test_mask]

def benchmarkBoostedClassifier(classifier, trainX, trainY, testX, testY, iteration_vector,
                                 base_classifier_args = {}, ensemble_args={}):
    """Measure the error of a given boosted classifier as a function of the number of iterations."""
    boosted_classifier = classifier(trainX, trainY, base_classifier_args, ensemble_args)
    # a vector that stores differences between adjacent elements of iteration_vector; it is used to determine
    # the number of classifiers that need to be added in each loop in order to maintain the total number of
    # classifiers as given in the iteration_vector: 
    iteration_differences_vector = []
    for i in xrange(len(iteration_vector)):
        if i == 0:
            iteration_differences_vector.append(iteration_vector[i])
        else:
            iteration_differences_vector.append(iteration_vector[i] - iteration_vector[i-1])
    # run the benchmark:
    train_error_vector = []
    test_error_vector = []
    for n in iteration_differences_vector:
        boosted_classifier.fit(n)
        train_error = 1.0 - float(sum(boosted_classifier.predict(trainX) == trainY))/trainY.shape[0]
        test_error = 1.0 - float(sum(boosted_classifier.predict(testX) == testY))/testY.shape[0]
        train_error_vector.append(train_error)
        test_error_vector.append(test_error)
    return train_error_vector, test_error_vector

def crossValidatedBenchmark(classifier, datasetX, datasetY, k, iteration_vector,
                                                            base_classifier_args = {}, ensemble_args={}):
    """Perform k-fold cross-validation to evaluate boosted model's performance as a function of iteration count."""
    n_total = int(datasetX.shape[0])
    fold_size = n_total/k
    final_train_errors = np.ndarray((len(iteration_vector),), dtype=float)
    final_test_errors = np.ndarray((len(iteration_vector),), dtype=float)
    final_train_errors.fill(0)
    final_test_errors.fill(0)
    for i in xrange(k):
        print('Starting #' + str(i+1) + ' cross-validation fold..')
        if i == k-1:
            test_indices = range(i*fold_size, n_total)
        else:
            test_indices = range(i*fold_size, (i+1)*fold_size)
        test_mask = np.ndarray((n_total,), dtype=bool)
        test_mask.fill(False)
        test_mask[test_indices] = True
        train_mask = ~test_mask
        train_errors, test_errors = benchmarkBoostedClassifier(classifier,
                                                                datasetX[train_mask], 
                                                                datasetY[train_mask], 
                                                                datasetX[test_mask], 
                                                                datasetY[test_mask], 
                                                                iteration_vector,
                                                                base_classifier_args=base_classifier_args,
                                                                ensemble_args=ensemble_args)
        final_train_errors += train_errors
        final_test_errors += test_errors
    final_train_errors /= k
    final_test_errors /= k
    return final_train_errors, final_test_errors

def visualizeBenchmarkComparison(iterations, ref_train_errors, ref_test_erors, train_errors, test_errors, \
                                 title1, title2, main_title):
    """Plot the train and test errors of two compared classifiers against the number of iterations."""
    fig = plt.figure(figsize=(12,6))
    fig.suptitle(main_title)
    p1 = plt.subplot(121)
    p1.set_title(title1)
    p1.set_xlabel('iterations')
    ref_train_plot, = p1.plot(iterations, ref_train_errors)
    ref_test_plot, = p1.plot(iterations, ref_test_erors)
    p1.legend([ref_train_plot, ref_test_plot], ['train error', 'test error'])
    p2 = plt.subplot(122)
    p2.set_title(title2)
    p2.set_xlabel('iterations')
    p2.set_ylim(p1.get_ylim())
    train_plot, = p2.plot(iterations, train_errors)
    test_plot, = p2.plot(iterations, test_errors)
    p2.legend([train_plot, test_plot], ['train error', 'test error'])

# -------------------------------- batch benchmarking utils: --------------------------------

class BenchmarkScenario(object):
    """Class that defines a single comparison of regular AdaBoost with a given variant."""
    def __init__(self, ref_classifier, classifier, dataset, dataset_name, k, iteration_vector, ensemble_args):
        self.ref_classifier = ref_classifier
        self.classifier = classifier
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.k = k
        self.iteration_vector = iteration_vector
        self.ensemble_args = ensemble_args

    def run(self):
        datasetX = self.dataset[:,0:(self.dataset.shape[1]-1)]
        datasetY = self.dataset[:,self.dataset.shape[1]-1]
        ref_train_errors, ref_test_errors = crossValidatedBenchmark(self.ref_classifier,
                                                                   datasetX,
                                                                   datasetY,
                                                                   self.k,
                                                                   self.iteration_vector)

        train_errors, test_errors = crossValidatedBenchmark(self.classifier,
                                                            datasetX, 
                                                            datasetY, 
                                                            self.k, 
                                                            self.iteration_vector, 
                                                            ensemble_args=self.ensemble_args)
        self.last_train_errors, self.last_test_errors = train_errors, test_errors
        self.last_ref_train_errors, self.last_ref_test_errors = ref_train_errors, ref_test_errors
        title1 = self.ref_classifier.__name__
        title2 = self.classifier.__name__ + str(self.ensemble_args).replace(', ', ',\n')
        main_title = self.dataset_name
        visualizeBenchmarkComparison(self.iteration_vector,
                                     ref_train_errors, 
                                     ref_test_errors, 
                                     train_errors, 
                                     test_errors, 
                                     title1, 
                                     title2,
                                     main_title)
