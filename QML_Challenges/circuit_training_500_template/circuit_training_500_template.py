#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable
import torch.optim as optim

def classify_data(X_train, Y_train, X_test):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []
    # QHACK #
    np.random.seed(0)
    torch.manual_seed(0)
    
    num_classes = 3
    margin = 0.15
    feature_size = 3
    batch_size = 10
    lr_adam = 0.01
    train_split = 0.75
    # the number of the required qubits is calculated from the number of features
    num_qubits = int(np.ceil(np.log2(feature_size)))
    num_layers = 6
    total_iterations = 40
    dev = qml.device("default.qubit", wires=num_qubits)
    
    def layer(W):
        for i in range(num_qubits):
            qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        for j in range(num_qubits - 1):
            qml.CNOT(wires=[j, j + 1])
        if num_qubits >= 2:
            # Apply additional CNOT to entangle the last with the first qubit
            qml.CNOT(wires=[num_qubits - 1, 0])
            
    def circuit(weights, feat=None):
        qml.templates.embeddings.AmplitudeEmbedding(feat, range(num_qubits), pad=0.0, normalize=True)
        for W in weights:
            layer(W)
        
        return qml.expval(qml.PauliZ(0))

    qnodes = []
    for iq in range(num_classes):
        qnode = qml.QNode(circuit, dev, interface="torch")
        qnodes.append(qnode)
    
    def variational_classifier(q_circuit, params, feat):
        weights = params[0]
        bias = params[1]
        return q_circuit(weights, feat=feat) + bias
    
    def multiclass_svm_loss(q_circuits, all_params, feature_vecs, true_labels):
        loss = 0
        num_samples = len(true_labels)
        for i, feature_vec in enumerate(feature_vecs):
            # Compute the score given to this sample by the classifier corresponding to the
            # true label. So for a true label of 1, get the score computed by classifer 1,
            # which distinguishes between "class 1" or "not class 1".
            s_true = variational_classifier(
                q_circuits[int(true_labels[i])],
                (all_params[0][int(true_labels[i])], all_params[1][int(true_labels[i])]),
                feature_vec,
            )
            s_true = s_true.float()
            li = 0
    
            # Get the scores computed for this sample by the other classifiers
            for j in range(num_classes):
                if j != int(true_labels[i]):
                    s_j = variational_classifier(
                        q_circuits[j], (all_params[0][j], all_params[1][j]), feature_vec
                    )
                    s_j = s_j.float()
                    li += torch.max(torch.zeros(1).float(), s_j - s_true + margin)
            loss += li
    
        return loss / num_samples

    def classify(q_circuits, all_params, feature_vecs, labels):
        predicted_labels = []
        for i, feature_vec in enumerate(feature_vecs):
            scores = np.zeros(num_classes)
            for c in range(num_classes):
                score = variational_classifier(
                    q_circuits[c], (all_params[0][c], all_params[1][c]), feature_vec
                )
                scores[c] = float(score)
            pred_class = np.argmax(scores)
            predicted_labels.append(pred_class)
        return predicted_labels
    
    
    def accuracy(labels, hard_predictions):
        loss = 0
        for l, p in zip(labels, hard_predictions):
            if torch.abs(l - p) < 1e-5:
                loss = loss + 1
        loss = loss / labels.shape[0]
        return loss
    def split_data(feature_vecs, Y):
        num_data = len(Y)
        num_train = int(train_split * num_data)
        index = np.random.permutation(range(num_data))
        feat_vecs_train = feature_vecs[index[:num_train]]
        Y_train = Y[index[:num_train]]
        feat_vecs_test = feature_vecs[index[num_train:]]
        Y_test = Y[index[num_train:]]
        return feat_vecs_train, feat_vecs_test, Y_train, Y_test
    def training(features, Y):
        feat_vecs_train, feat_vecs_test, Y_train, Y_test = split_data(features, Y)
        num_train = Y_train.shape[0]
        q_circuits = qnodes
    
        # Initialize the parameters
        all_weights = [
            Variable(0.1 * torch.randn(num_layers, num_qubits, 3), requires_grad=True)
            for i in range(num_classes)
        ]
        all_bias = [Variable(0.1 * torch.ones(1), requires_grad=True) for i in range(num_classes)]
        optimizer = optim.Adam(all_weights + all_bias, lr=lr_adam)
        params = (all_weights, all_bias)
        print("Num params: ", 3 * num_layers * num_qubits * 3 + 3)
    
        costs, train_acc, test_acc = [], [], []
    
        # train the variational classifier
        for it in range(total_iterations):
            batch_index = np.random.randint(0, num_train, (batch_size,))
            feat_vecs_train_batch = feat_vecs_train[batch_index]
            Y_train_batch = Y_train[batch_index]
    
            optimizer.zero_grad()
            curr_cost = multiclass_svm_loss(q_circuits, params, feat_vecs_train_batch, Y_train_batch)
            curr_cost.backward()
            optimizer.step()
    
            # Compute predictions on train and validation set
            predictions_train = classify(q_circuits, params, feat_vecs_train, Y_train)
            predictions_test = classify(q_circuits, params, feat_vecs_test, Y_test)
                        
            acc_train = accuracy(Y_train, predictions_train)
            acc_test = accuracy(Y_test, predictions_test)
    
            print(
                "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc test: {:0.7f} "
                "".format(it + 1, curr_cost.item(), acc_train, acc_test)
            )
    
            costs.append(curr_cost.item())
            train_acc.append(acc_train)
            test_acc.append(acc_test)
    
        return costs, train_acc, test_acc
    
    X = torch.tensor(X_train)
    print("First X sample, original  :", X[0])

    # normalize each input
    normalization = torch.sqrt(torch.sum(X ** 2, dim=1))
    X_norm = X / normalization.reshape(len(X), 1)
    print("First X sample, normalized:", X_norm[0])

    Y = torch.tensor(Y_train)
    
    costs, train_acc, test_acc = training(X, Y)
    XT = torch.tensor(X_test)
    print("First X sample, original  :", X[0])

    # normalize each input
    normalization = torch.sqrt(torch.sum(XT ** 2, dim=1))
    X_normT = XT / normalization.reshape(len(XT), 1)
    all_weights = [
    Variable(0.1 * torch.randn(num_layers, num_qubits, 3), requires_grad=True)
    for i in range(num_classes)
    ]
    all_bias = [Variable(0.1 * torch.ones(1), requires_grad=True) for i in range(num_classes)]
    params = (all_weights, all_bias)
    print("First X sample, normalized:", X_normT[0])
    predictions = classify(qnodes, params, X_normT, [])
    # QHACK #

    return array_to_concatenated_string(predictions)


def array_to_concatenated_string(array):
    """DO NOT MODIFY THIS FUNCTION.

    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY THIS FUNCTION.

    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY THIS FUNCTION.

    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    with open('1.in') as f:
        X_train, Y_train, X_test = parse_input(f.read())
    #X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")
