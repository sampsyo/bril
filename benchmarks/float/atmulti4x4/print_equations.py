# Use this script in a deepmind alphatensor checkout to print numpy formulas for alphatensor 4,4,4 general matmul
# https://github.com/deepmind/alphatensor/issues/3
import numpy as np
from ast import literal_eval as make_tuple

np.random.seed(0)

"""
The *.npz files contain a dict with keys like "(2,3,4)" and values containing
a list of matrices U, V and W. For example, for the 2-by-2 times 2-by-2 case,
we have the following matrices:

U =
[[ 0  1  1  0  1  1  0]
 [ 0  0 -1  1  0  0  0]
 [ 1  1  1  0  1  0  0]
 [-1 -1 -1  0  0  0  1]]

V =
[[0 0 0 0 1 1 0]
 [1 1 0 0 1 0 1]
 [0 1 1 1 1 0 0]
 [0 1 1 0 1 0 1]]

W =
[[ 0  0  0  1  0  1  0]
 [ 0 -1  0  0  1 -1 -1]
 [-1  1 -1 -1  0  0  0]
 [ 1  0  0  0  0  0  1]]

Each column of U is multiplied with the vectorized matrix A.
Likewise, Each column of V is multiplied with the vectorized matrix B.
The resulting vectors are multiplied pointwise and their product is
multiplied with W, which forms the entries of the product matrix C = A times B.
Also see the function `multiply` below.
"""

# There are two factorizations, one for useful numbers ###and one for mod 2 math.
filename = "factorizations_r.npz"
mod = None
# Load the factorizations. Note that allow_pickle=True allows arbitrary
# code execution. A JSON file would have been a better format choice
# since nothing here is stored in NumPy format anyway.
factorizations = dict(np.load(filename, allow_pickle=True))

#(print(item) for item in factorizations.items())
### # Test each factorization
for key, UVW in factorizations.items():
#for key, UVW in factorizations[(4,4,4)]:
#for key, UVW in factorizations.items():
    #print(key)
    if key == "4,4,4":
#        U, V, W = map(np.array, UVW)
#if (4,4,4) not in factorizations:
#  print("error no key (4,4,4)")
#  exit

#fact_tuple = factorizations[(4,4,4)]
#key = fact_tuple[0]
#UVW = fact_tuple[1]
        U, V, W = map(np.array, UVW)
        m, k, n = make_tuple(key)
        print(f"\nMultiply {m}-by-{k} matrix A with {k}-by-{n} matrix B")
        if mod is not None:
            print(f"using mod {mod} arithmetic")
        print()
        # Check that shapes are correct
        assert m * k == U.shape[0]
        assert k * n == V.shape[0]
        assert m * n == W.shape[0]
        assert U.shape[1] == V.shape[1]
        assert U.shape[1] == W.shape[1]
        # Generate two random matrices for testing
        A = np.random.randint(10, size=(m, k))
        B = np.random.randint(10, size=(k, n))
        def multiply(A, B, U, V, W):
            # Multiply two matrices A and B using index matrices U, V and W
            a = A.ravel()
            b = B.ravel()
        
            tmp = (U.T @ a) * (V.T @ b)
            c = W @ tmp
            C = c.reshape(n, m).T
        
            return C
        
        # Multiply matrices
        C = multiply(A, B, U, V, W)
        
        # Check that result is correct, taking potential mod 2 into account
        if mod is None:
            assert np.allclose(C, A @ B)
        else:
            assert np.allclose(C % mod, (A @ B) % mod)
        
        def make_code(variables, factors):
            # Generate code like "(a11 + a21 - a22)"
            parts = []
        
            for variable, factor in zip(variables, factors):
                # Simplify +1 and -1 factors
                if factor == 1:
                    factor = " + "
                elif factor == -1:
                    factor = " - "
                elif factor < 0:
                    factor = f" {factor} * "
                elif factor > 0:
                    factor = f" + {factor} * "
                else:
                    continue
        
                parts.append(factor + variable)
        
            code = "".join(parts).lstrip(" +")
        
            if len(parts) > 1:
                code = "(" + code + ")"
        
            return code
        
        def make_variables(var, m, n):
            # Generate variables like a11, a12, a21, a22
            # or maybe a_1_1, a_1_2, a_2_1, a_2_2.
            # For larger matrices, we need a separator to avoid
            # confusing e.g. a_1_11 with a_11_1.
            separator = "_" if max(m, n, k) > 9 else ""
            return [f"{var}{separator}{i + 1}{separator}{j + 1}"
                for i in range(m) for j in range(n)]
        
        A_variables = make_variables("a", m, k)
        B_variables = make_variables("b", k, n)
        C_variables = make_variables("c", m, n)
        h_variables = [f"h{i + 1}" for i in range(U.shape[1])]
        
        lines = [
            ", ".join(A_variables) + " = A.ravel()",
            ", ".join(B_variables) + " = B.ravel()",
        ]
        
        # Generate code for computation of temporary vector
        for h, u, v in zip(h_variables, U.T, V.T):
            sa = make_code(A_variables, u)
            sb = make_code(B_variables, v)
        
            lines.append(f"{h} = {sa} * {sb}")
        
        # Generate code for computation
        for c, w in zip(C_variables, W):
            lines.append(f"{c} = " + make_code(h_variables, w).strip("()"))
        
        lines.append("C = np.array([" + ", ".join(C_variables) +
            f"]).reshape({n}, {m}).T")
        
        code = "\n".join(lines)
        
        print(code)
        
        # Verify that code generates the correct result
        exec(code)
        
        if mod is None:
            assert np.allclose(C, A @ B)
        else:
            assert np.allclose(C % mod, (A @ B) % mod)
        
