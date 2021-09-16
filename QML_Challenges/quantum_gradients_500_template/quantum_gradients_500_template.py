#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #
    @qml.qnode(dev)
    def Q(params):
        variational_circuit(params)
        return qml.probs(wires=[0,1,2])
    
    s=np.pi/2
    num_params = len(params)
    base = Q(params)[0]
    fubini = np.zeros((num_params, num_params))
    for i in range(num_params):
        for j in range(num_params):
            if (i == j):
                mod = params.copy()
                mod[i] += 2*s
                plusPi = Q(mod)[0]
                mod[i] -= 4*s
                minusPi = Q(mod)[0]
                fubini[i, i] = (2*base - plusPi - minusPi)
            else:
                mod = params.copy()
                mod[i] += s
                mod[j] += s
                #print("pop", Q(mod))
                eval1 = Q(mod)[0]
                mod[i] -= 2*s
                eval2 = Q(mod)[0]
                mod[j] -= 2*s
                eval3 = Q(mod)[0]
                mod[i] += 2*s
                eval4 = Q(mod)[0]
                #print(eval1, eval2, eval3, eval4)
                fubini[i,j] = (eval2 + eval4 - eval1 - eval3)
    fubini /= 8
    
    
    gradient = np.zeros([num_params], dtype=np.float64)
    gradQuantities = np.zeros([num_params, 2], dtype=np.float64)
     
    s = 1e-6
    for i in range(len(gradient)):
        modWeights = params.copy()
        modWeights[i] += s
        gradQuantities[i, 0] = qnode(modWeights)
        modWeights[i] -= 2*s
        gradQuantities[i, 1] = qnode(modWeights)
        gradient[i] = gradQuantities[i, 0] - gradQuantities[i, 1]
    gradient /= 2*np.sin(s)
    
    print("fub", np.round(fubini, 5), sep="\n")
    print("approx", qml.metric_tensor(qnode)(params), sep="\n")
    
    print(fubini.shape, gradient.shape)
    natural_grad = np.linalg.inv(fubini) @ gradient
    print("nat_grad", natural_grad, sep="\n")

    # QHACK #

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = "1,1,1,2,2,2" #sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
