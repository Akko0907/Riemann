import numpy as np
import sympy as sp

# constante Rs e coordenadas 'q'
Rs,q0,q1,q2,q3  = sp.symbols("Rs,q0,q1,q2,q3")

# Metrica de Schwarzschild em coordenadas polares
g = np.zeros((4,4),dtype='object')
g[0][0] = (q1-Rs)/q1
g[1][1] = -q1/(q1-Rs)
g[2][2] = -q1**2
g[3][3] = -q1**2 * sp.sin(q2)**2

# Funcao para o calculo da inversa de g
def inverse(g):
    g = sp.Matrix(g)
    g_inv = g.inv()
    g_inv = np.array(g_inv)
    return g_inv

# Executa a derivacao com respeito a cada coordenada
# para cada elemento de g e guarda em um array d = (4,4,4)
def del_k(g):
    # Derivative k, Line i, Column j
    D = np.zeros((4,4,4),dtype='object')
    del_ = [q0,q1,q2,q3]
    for k in range(4):
        for i in range(4):
            for j in range(4):
                if type(g[i][j]) is int:
                    D[k][i][j] = 0
                else:
                    D[k][i][j] = g[i][j].diff(del_[k])
    return D


# Calcula os elementos do simbolo de Christoffel
def Chrisff_Symbol(g,D):
    S = np.zeros((4,4,4),dtype='object')
    for k in range(4):
        for i in range(4):
            for j in range(4):
                for n in range(4):
                    S[k][i][j] += g[k][n]*(D[i][n][j] + D[j][n][i] - D[n][i][j])/2
    return S

# calcula o tensor de Riemann
def Riemann(S):
    del_ = [q0,q1,q2,q3]
    R = np.zeros((4,4,4,4),dtype='object')
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):

                    if type(S[i][l][j]) is int:
                        diff_a = 0
                    else:
                        diff_a = S[i][j][l].diff(del_[k])
                    
                    if type(S[i][k][j]) is int:
                        diff_b = 0
                    else:
                        diff_b = S[i][j][k].diff(del_[l])
                    
                    R[i][j][k][l] += diff_a - diff_b
                    for n in range(4):
                        R[i][j][k][l] += - S[n][j][k]*S[i][n][l] + S[n][j][l]*S[i][n][k]
    return R
    

# Calcula o tensor de Ricci
def Ricci(R,g,g_inv):
    Ri = np.zeros((4,4),dtype='object')
    for i in range(4):
        for j in range(4):
            for u in range(4):
                for a in range(4):
                    for b in range(4):
                        Ri[i][j] += g_inv[b][u]*g[u][a]*R[a][i][b][j]
    return Ri

print("\nCoordenadas Esfericas:")
g_inv = inverse(g)
D = del_k(g)
S = Chrisff_Symbol(g_inv,D)
R = Riemann(S)
Ri = Ricci(R,g,g_inv)
Ri_ij = np.array([[sp.simplify(Ri[i][j]) for i in range(4)] for j in range(4)])

print("\nInput: Metrica de Schwarzschild ")
print(g)

print("\n\nOutput: Tensor de Ricci")
print(Ri_ij)
print("Dado que o tensor de Ricci se anula para a metrica de Schwarzschild,")
print("o e escalar de Ricci tbm deve valer 0.")
print("\n")


