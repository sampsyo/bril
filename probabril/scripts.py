import numpy as np;

A = np.array([[0,1,0,1,0,0,0,0,0], [0,0,1, 0,0,0, 0,0,1], [2,0,0, 0,0,0, 0,0,0], [0,0,0, 0,2,0, 0,0,0], [0,0,0, 0,0,1, 1,0,0], [0,0,0,2,0,0, 0,0,0], [0,0,1, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,2,0], [0,0,0, 0,0,0, 0,0,2]]) /2


def sample(A):
    state = 0
    for i in range(1000):
        state = np.random.choice(9, p = A[state])
    return state

np.around(np.linalg.matrix_power(A, 100).T @ np.array([1,0,0, 0,0,0, 0,0,0]), 3)
S = [sample(A) for i in range(1000)  ]
np.unique(S, return_counts=True)
