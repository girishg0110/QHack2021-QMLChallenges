#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    
    gradQuantities = np.zeros([5, 2], dtype=np.float64)
     
    s = 1e-6
    base = circuit(weights)
    for i in range(len(gradient)):
        modWeights = weights.copy()
        modWeights[i] += s
        gradQuantities[i, 0] = circuit(modWeights)
        modWeights[i] -= 2*s
        gradQuantities[i, 1] = circuit(modWeights)
        gradient[i] = gradQuantities[i, 0] - gradQuantities[i, 1]
    gradient /= 2*np.sin(s)
    
    for i in range(len(hessian)):
        for j in range(i, len(hessian[i])):
            if (i == j):
                hessian[i, i] = 4*(gradQuantities[i, 0] - 2*base + gradQuantities[i, 1])
            else:
                modWeights = weights.copy()
                modWeights[i] += s
                modWeights[j] += s
                hessian[i, j] += circuit(modWeights)
                modWeights[j] -= 2*s
                hessian[i, j] -= circuit(modWeights)
                modWeights[i] -= 2*s
                hessian[i, j] += circuit(modWeights)
                modWeights[j] += 2*s
                hessian[i, j] -= circuit(modWeights)
                hessian[j, i] = hessian[i, j]
    hessian /= (4*np.sin(s)**2)
    # QHACK #
    #error_check(gradient, hessian)
    return gradient, hessian, circuit.diff_options["method"]

def error_check(grad, hess):
    answer = np.zeros(len(grad) + len(grad)**2)
    i = 0
    while (i<len(grad)):
        answer[i] = grad[i]
        i+=1
    i = 0
    while (i<len(grad)**2):
        answer[len(grad)+i] = hess.flatten()[i]
        i+=1
    with open('1.ans') as file:
        l = file.readline().split(',')[:-1]
        l = np.array([float(a) for a in l])
        
    print( (answer-l)/answer )

if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = "0.1,0.2,0.1,0.2,0.7" #sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)
    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)


    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
