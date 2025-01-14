## looking at the convergence of the statistical terms

def next_term_matrix(prevterm,mul):
    return np.matmul(prevterm,mul)
import numpy as np
import matplotlib.pyplot as plt
import math
from  scipy import linalg

phasehift = [[1, 0, 0, 0, 0], [0, np.e**(2*np.pi*1j/5), 0, 0, 0], [0, 0, 
   np.e**(4*np.pi*1j/5), 0, 0], [0, 0, 0, np.e**(6*np.pi*1j/5), 0], [0, 0, 0, 0, 
   np.e**(8*np.pi*1j/5)]]
spinshift = [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 
   0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0]]
H1 = np.zeros((25,25),dtype=complex)

parameter = -(0.079 + 0.060j)
H1 = np.zeros((25,25),dtype=complex)
for k in range(1,5):
    for j in range(1,5):
        H1 += (np.matmul(np.kron(np.linalg.matrix_power(spinshift, k), np.identity(5)) +  np.kron(np.identity(5), np.linalg.matrix_power(spinshift, k)),np.kron(np.linalg.matrix_power(np.conjugate(phasehift), j), np.linalg.matrix_power(phasehift, j))) +  np.matmul(np.kron(np.linalg.matrix_power(np.conjugate(phasehift), j), np.linalg.matrix_power(phasehift, j)), (np.kron(np.linalg.matrix_power(spinshift, k), np.identity(5)) + np.kron(np.identity(5), np.linalg.matrix_power(spinshift, k)) )))
H1 = parameter*H1



### taylor expansion method
terms = [np.identity(25,dtype=complex)]
prev = H1
for m in range(1,150):
    terms.append(terms[-1] +prev/math.factorial(m))
    prev = next_term_matrix(prev,H1)



# plt.xlabel("real part of terms")
# plt.ylabel("imaginary part of terms")
# plt.savefig('Convergence_test_wtau.png')
# plt.show()
# print(terms[-1])


### exponentiate method (they agree)
expH1 = linalg.expm(H1)
for (i,rows) in enumerate(expH1):
 for (j,elements) in enumerate(rows):
    k = i%5
    g = i//5
    l = j%5
    p = j//5
    print(str(g)+str(k)+"  ->  "+str(p)+str(l)+":  "+str(elements))
