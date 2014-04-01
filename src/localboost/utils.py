import numpy as np

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

def benchmarkBoostedClassifier(classifier, trainX, trainY, testX, testY, iteration_vector, **kwargs):
    """Measure the error of a given boosted classifier as a function of the number of iterations."""
    boosted_classifier = classifier(trainX, trainY, **kwargs)
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