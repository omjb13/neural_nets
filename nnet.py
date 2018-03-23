from __future__ import division

import logging
import warnings
from collections import defaultdict
from sys import argv

import numpy as np
from scipy.io.arff import loadarff

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

warnings.simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(suppress=True)

USAGE = """
1 hidden layer neural net : 
python nnet.py n LEARNING_RATE NUM_HIDDEN_NODES EPOCHS TRAIN_DATA TEST_DATA

Logistic regression : 
python nnet.py l LEARNING_RATE EPOCHS TRAIN_DATA TEST_DATA
"""


class MetaData(object):
    def __init__(self, meta):
        ranges = {}
        types = {}
        for _, (feature_name, details) in enumerate(meta._attributes.iteritems()):
            types[feature_name], ranges[feature_name] = details
        self.ranges = ranges
        self.types = types
        self.feature_names = [x for x in meta]


class DataSet(object):
    def __init__(self, raw_data, meta):
        self.metadata = MetaData(meta)
        # makes the assumption that class is always the last column
        self.data = raw_data[self.metadata.feature_names[0:-1]]
        self.labels = raw_data['class']

    def standardize_numeric(self, values_used=None):
        to_change = []
        for feature, type in self.metadata.types.iteritems():
            if type == "numeric":
                to_change.append(feature)
        new_values = {}
        for feature in to_change:
            current_fdata = self.data[feature]
            if values_used:
                mean = values_used[feature][0]
                stdev = values_used[feature][1]
            else:
                mean = np.mean(current_fdata)
                stdev = np.std(current_fdata)
                new_values[feature] = (mean, stdev)
            new_fdata = np.divide(np.subtract(current_fdata, mean), stdev)
            self.data[feature] = new_fdata
        if values_used:
            return values_used
        else:
            return new_values

    def shuffle(self):
        p = np.random.permutation(len(self.data))
        self.data = self.data[p]
        self.labels = self.labels[p]


class SingleLayerNN(object):
    def __init__(self):
        self.weights = 0

    def init_weights(self, dataset):
        sample = dataset.data[0]
        sample_encoded = encode_inputs(sample, dataset)
        num_weights = len(sample_encoded)
        self.weights = np.random.uniform(-0.01, 0.01, num_weights)

    def train(self, dataset):
        self.init_weights(dataset)
        for epoch in range(epochs):
            results = []
            dataset.shuffle()
            error = 0
            correct_count = 0
            for _x, _y in zip(dataset.data, dataset.labels):
                result = []
                y = convert_label_to_int(dataset, _y)
                x = encode_inputs(_x, dataset)
                # forward propagation
                z1 = x.dot(self.weights)
                a1 = sigmoid(z1)
                predicted = int(np.round(a1))
                if predicted == y:
                    correct_count += 1
                result.append(a1)
                result.append(predicted)
                result.append(y)
                results.append(result)
                # backward propagation
                error += - y * np.log(a1) - (1 - y) * np.log(1 - a1)
                error_gradients = np.multiply(x, (a1 - y))
                weight_delta = np.multiply(error_gradients, -learning_rate)
                self.weights = np.add(self.weights, weight_delta)
            print "%s\t%f\t%d\t%d" % (epoch + 1, error, correct_count, len(dataset.data) - correct_count)
            # print get_f1(results)

    def test(self, dataset):
        correct_count = 0
        results = []
        for _x, _y in zip(dataset.data, dataset.labels):
            to_print = []
            x = encode_inputs(_x, dataset)
            y = convert_label_to_int(dataset, _y)
            z1 = x.dot(self.weights)
            a1 = sigmoid(z1)
            predicted = int(np.round(a1))
            if predicted == y:
                correct_count += 1
            to_print.append(a1)
            to_print.append(predicted)
            to_print.append(y)
            results.append(to_print)
            print "\t".join(map(str, to_print))
        print "\t".join(map(str, (correct_count, len(dataset.data) - correct_count)))
        print get_f1(results)
        return results


class NeuralNet(object):
    def __init__(self, n):
        self.n_hidden = n
        # weights from input to hidden layer
        self.w1 = None
        # weights from hidden layer to output
        self.w2 = None

    def init_weights(self, dataset):
        sample = dataset.data[0]
        sample_encoded = encode_inputs(sample, dataset)
        n_w1 = len(sample_encoded)
        # w1 is (num_inputs + 1) x (num hidden nodes)
        self.w1 = np.random.uniform(-0.01, 0.01, (n_w1, self.n_hidden))
        # w2 is (num_hidden + 1) x (1)
        self.w2 = np.random.uniform(-0.01, 0.01, self.n_hidden + 1)

    def train(self, dataset):
        self.init_weights(dataset)
        for epoch in range(epochs):
            dataset.shuffle()
            correct_count = 0
            error = 0
            for _x, _y in zip(dataset.data, dataset.labels):
                # forward propagation
                y = convert_label_to_int(dataset, _y)
                x = encode_inputs(_x, dataset)
                a1 = x.dot(self.w1)
                z1 = sigmoid_np(a1)
                z1_with_bias = np.concatenate((np.ones(1), z1))
                a2 = z1_with_bias.dot(self.w2)
                z2 = sigmoid(a2)
                predicted = int(np.round(z2))
                if predicted == y:
                    correct_count += 1
                error += - y * np.log(z2) - (1 - y) * np.log(1 - z2)
                # backward propagation
                out_unit_error = y - z2
                hidden_unit_errors = np.zeros(self.n_hidden + 1)
                for i in range(self.n_hidden + 1):
                    oj = z1_with_bias[i]
                    hidden_unit_errors[i] = oj * (1 - oj) * self.w2[i] * out_unit_error
                w2_deltas = np.zeros(self.n_hidden + 1)
                for i in range(self.n_hidden + 1):
                    w2_deltas[i] = learning_rate * out_unit_error * z1_with_bias[i]
                w1_deltas = np.zeros((len(x), self.n_hidden))
                for j in range(self.n_hidden):
                    for i in range(len(x)):
                        w1_deltas[i][j] = learning_rate * hidden_unit_errors[j] * x[i]
                self.w1 += w1_deltas
                self.w2 += w2_deltas
            print "%s\t%f\t%d\t%d" % (epoch + 1, error, correct_count, len(dataset.data) - correct_count)

    def test(self, dataset):
        correct_count = 0
        results = []
        for _x, _y in zip(dataset.data, dataset.labels):
            to_print = []
            y = convert_label_to_int(dataset, _y)
            x = encode_inputs(_x, dataset)
            a1 = x.dot(self.w1)
            z1 = sigmoid_np(a1)
            z1_with_bias = np.concatenate((np.ones(1), z1))
            a2 = z1_with_bias.dot(self.w2)
            z2 = sigmoid(a2)
            predicted = int(np.round(z2))
            if predicted == y:
                correct_count += 1
            to_print.append(z2)
            to_print.append(predicted)
            to_print.append(y)
            results.append(to_print)
            print "\t".join(map(str, to_print))
        print "\t".join(map(str, (correct_count, len(dataset.data) - correct_count)))
        print get_f1(results)
        return results


## HELPERS

def convert_label_to_int(dataset, label):
    return dataset.metadata.ranges['class'].index(label)


def encode_inputs(x, dataset):
    # TODO - handle nominal features
    encoded = []
    for feature_name in dataset.metadata.feature_names:
        feature_range =  dataset.metadata.ranges[feature_name]
        if feature_name == "class":
            continue
        if feature_range is None :
            # numeric feature
            encoded.append(x[feature_name])
        else :
            # nominal feature
            encoded.extend(one_hot_encode(x[feature_name], feature_range))
    return np.concatenate([np.ones(1), np.array(encoded)])

def one_hot_encode(item, item_list):
    onehot_encoded = []
    for i in item_list:
        if i == item :
            onehot_encoded.append(1)
        else :
            onehot_encoded.append(0)
    return onehot_encoded

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_f1(results):
    counts = defaultdict(int)
    for _, predicted, actual in results:
        if predicted == 1 and actual == 1:
            counts["tp"] += 1
        elif predicted == 1 and actual == 0:
            counts["fp"] += 1
        elif predicted == 0 and actual == 1:
            counts["fn"] += 1
    try :
        precision = counts["tp"] / (counts["tp"] + counts["fp"])
        recall = counts["tp"] / (counts["tp"] + counts["fn"])
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        return 0
    return f1


sigmoid_np = np.vectorize(sigmoid)

if __name__ == '__main__':
    mode = argv[1]
    if mode == "n":
        learning_rate = float(argv[2])
        n_hidden = int(argv[3])
        epochs = int(argv[4])
        train_file = argv[5]
        test_file = argv[6]

    elif mode == "l":
        learning_rate = float(argv[2])
        epochs = int(argv[3])
        train_file = argv[4]
        test_file = argv[5]

    else:
        print USAGE
        exit(1)

    _train_data, train_meta = loadarff(file(train_file))
    train_data = DataSet(_train_data, train_meta)
    values_used = train_data.standardize_numeric()

    _test_data, test_meta = loadarff(file(test_file))
    test_data = DataSet(_test_data, test_meta)
    test_data.standardize_numeric(values_used)

    if mode == "l":
        n1 = SingleLayerNN()
        n1.train(train_data)
        n1.test(test_data)

    elif mode == "n":
        n1 = NeuralNet(n_hidden)
        n1.train(train_data)
        n1.test(test_data)
