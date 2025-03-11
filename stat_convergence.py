## looking at the convergence of the statistical terms

def next_term_matrix(prevterm,mul):
    return np.matmul(prevterm,mul)
import numpy as np
import matplotlib.pyplot as plt
import math
from  scipy import linalg
from scipy import optimize
import SymPy

phasehift = [[1, 0, 0, 0, 0], [0, np.e**(2*np.pi*1j/5), 0, 0, 0], [0, 0, 
   np.e**(4*np.pi*1j/5), 0, 0], [0, 0, 0, np.e**(6*np.pi*1j/5), 0], [0, 0, 0, 0, 
   np.e**(8*np.pi*1j/5)]]
spinshift = [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 
   0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0]]
H1 = np.zeros((25,25),dtype=complex)

parameter = 1
H1 = np.zeros((25,25),dtype=complex)
Halt =  np.zeros((25,25),dtype=complex)
for k in range(1,5):
    for j in range(1,5):
        Halt+=(np.matmul(np.kron(np.linalg.matrix_power(spinshift, k), np.identity(5)),np.kron(np.linalg.matrix_power(np.conjugate(phasehift), j), np.linalg.matrix_power(phasehift, j))) +  np.matmul(np.kron(np.linalg.matrix_power(np.conjugate(phasehift), j), np.linalg.matrix_power(phasehift, j)), (np.kron(np.linalg.matrix_power(spinshift, k), np.identity(5))  )))

        H1 += (np.matmul(np.kron(np.linalg.matrix_power(spinshift, k), np.identity(5)) +  np.kron(np.identity(5), np.linalg.matrix_power(spinshift, k)),np.kron(np.linalg.matrix_power(np.conjugate(phasehift), j), np.linalg.matrix_power(phasehift, j))) +  np.matmul(np.kron(np.linalg.matrix_power(np.conjugate(phasehift), j), np.linalg.matrix_power(phasehift, j)), (np.kron(np.linalg.matrix_power(spinshift, k), np.identity(5)) + np.kron(np.identity(5), np.linalg.matrix_power(spinshift, k)) )))
H = parameter*H1+ sum([np.kron(np.linalg.matrix_power(spinshift, k), np.identity(5)) for k in range(1,5)])

Halt = Halt*parameter

# ### taylor expansion method
# terms = [np.identity(25,dtype=complex)]
# prev = H1
# for m in range(1,150):
#     terms.append(terms[-1] +prev/math.factorial(m))
#     prev = next_term_matrix(prev,H1)


eigenvalues = linalg.eig(Halt)
print(eigenvalues[0])
eigenvalues = linalg.eig(H)
print(eigenvalues[0])
# print(terms[0])
# plt.plot(range(0,150),terms[:][0][0])
# plt.xlabel("m expansion of terms")
# plt.ylabel("imaginary part of terms")
# plt.savefig('Convergence_test_wtau.png')
# plt.show()


# ### exponentiate method (they agree)
# expH1 = linalg.expm(H1)
# for (i,rows) in enumerate(expH1):
#  for (j,elements) in enumerate(rows):
#     k = i%5
#     g = i//5
#     l = j%5
#     p = j//5
#     print(str(g)+str(k)+"  ->  "+str(p)+str(l)+":  "+str(elements))



x = SymPy.Symbol('x')
H1 = SymPy.Matrix(H1)
M = H1 * x
print(SymPy.exp(M))














## manual cft graining

def a(x):
    return -3/2 + 5*x/2 + (np.sqrt(5)*np.sqrt(5 - 6*x + 5*x**2))/(2)


def apaper(x):
    return -x/8 + 3/8 + np.sqrt(x**2/64 - 3 *x/32  + 25/64)
# def f(x):
#     return x*(a(x)**2 - 2* a(x) + 1)/(2 *a(x) + 5 - 2)
def f(x):
    return x*(-a(x)**2 + 1)/(4/a(x) + 1)
def fpaper(x):
    return x*(1 - apaper(x)**2)/(2*apaper(x) + 3*apaper(x)**2)
print(optimize.fixed_point(fpaper,5))
print(optimize.fixed_point(f,6))