import numpy as np

def Jacobi(matrix_A,matrix_b):
    mat = matrix_A.copy()
    b = matrix_b.copy()
    row = len(mat)
    col = len(mat[0])
    x_new = np.zeros(row)
    x_old = np.zeros(row)
    comp = np.ones(row)
    while(comp.sum()>=1e-4):
        #print(comp.any()>100)
        if comp.sum()<100.:
            x_old = x_new.copy()
        #print(comp)
            for i in range(row):
                sum = 0.
                for j in range(row):
                    if j!=i:
                        sum = sum + mat[i][j]*x_old[j]
                x_new[i] = (b[i] - sum)/mat[i][i]
            comp = np.abs(x_new - x_old)
            #print(x_new)
            #print(x_old)
            #print(comp)
        else:
            print("Solution not possible, method diverges")
            print("comp ",comp.sum())
            break
    print("comp ",comp.sum())
    return x_new


#swap the row with max leftmost value to top
def swap_max(row_i,row,matrix):
    large = row_i
    for i in range(row_i,row):
        if matrix[large][row_i] <= matrix[i][row_i]:
            large = i
            matrix[large][row_i] = matrix[i][row_i]
    temp = matrix[row_i].copy()
    matrix[row_i] = matrix[large]
    matrix[large] = temp
    return matrix

def Gauss_Jordan(matrix):
    row = len(matrix)
    col = len(matrix[0])
    d = 0
    scalar_prod = 1
    det =1
    #swap zero rows to last
    zeros = []
    for i in range(row):
        if matrix[i][1] == 0:
            zeros.append(i)
            for j in range(col):
                if matrix[i][j] != 0:
                    zeros.remove(i)
                    break
    matrix = np.delete(matrix,zeros,0)   # remove the rows with only zeros
    for i in zeros:
        matrix = np.append(matrix,np.zeros(col).reshape(1,col),0)        #add the removed zero rows to the last
    for i in range(row):
        matrix_ini = matrix.copy()
        matrix = swap_max(i,row,matrix)
        if (matrix != matrix_ini).any():
            d = d+1
        if abs(matrix[i][i]) <= 1e-12:
                matrix[i][i] = 0
        scalar = matrix[i][i]            #add a scalar prod multiplier to use in calculating determinant
        if scalar == 0:
            break
        matrix[i] = matrix[i]/scalar
        #print("matrix after div with scalar is ",matrix)
        scalar_prod = scalar_prod*scalar
        for j in range(row):
            if j != i:
                scalar = matrix[j][i]
                #print("scalar[",j,"][",i, "]is ",scalar)
                #if scalar == 0:
                 #   print("breaking")
                  #  break
                matrix[j] = matrix[j] - (scalar*matrix[i])
            if abs(matrix[i][j]) <= 1e-12:
                matrix[i][j] = 0
            #print("matrix after subt is ",matrix)
    for i in range(row):
        det = det*matrix[i][i]
    det = pow(-1,d)*scalar_prod*det
    for i in range(row):
        for j in range(col):
            if abs(matrix[i][j]) <= 1e-12:
                matrix[i][j] = 0
    np.set_printoptions(precision=12,suppress=True)        
    #print(matrix)
    return matrix, det

def make_aug(matrix_A, matrix_b):
    row_A = len(matrix_A)
    col_A = len(matrix_A[0])
    row_b = len(matrix_b)
    col_b = len(matrix_b[0])
    if row_A == row_b:
        aug_matrix = np.append(matrix_A,matrix_b,axis=1)
    elif row_A == col_b:
        aug_matrix = np.append(matrix_A,np.reshape(matrix_b,(col_b,1)))
    else:
        raise Exception("unique soln probably not possible")
    return aug_matrix

def extract_inv(matrix):
    row = len(matrix)
    col = len(matrix[0])
    col_e = math.floor(col/2)      # column next to halfway
    matrix_e = np.reshape(matrix[:,col_e],(math.floor(col/2),1))     #reshape the extracted array from [1,n] to [n,1]
    col_e = col_e + 1
    for i in range(math.floor(col/2)-1):
        matrix_col = np.reshape(matrix[:,col_e],(math.floor(col/2),1))
        matrix_e = np.append(matrix_e,matrix_col,axis=1)             #append along axis=1, i.e, column wise
        #print("matrix_e is ",matrix_e)
        col_e = col_e + 1
    return matrix_e

def Gauss_Seidel(matrix_A,matrix_b):
    mat = matrix_A.copy()
    b = matrix_b.copy()
    row = len(mat)
    col = len(mat[0])
    x_new = np.zeros(row)
    x_old = np.zeros(row)
    comp = np.ones(row)
    while(comp.sum()>=1e-4):
        #print(comp.any()>100)
        if comp.sum()<100.:
            x_old = x_new.copy()
        #print(comp)
            for i in range(row):
                sum_1 = 0.
                for j in range(i):
                    sum_1 = sum_1 + mat[i][j]*x_new[j]
                sum_2 = 0.
                for j in range(i+1,row):
                    sum_2 = sum_2 + mat[i][j]*x_old[j]
                x_new[i] = (b[i] - sum_1 - sum_2)/mat[i][i]
            comp = np.abs(x_new - x_old)
            #print(x_new)
            #print(x_old)
            print("comp",comp.sum())
        else:
            print("Solution not possible, method diverges.")
            break
    print("comp ",comp.sum())
    return x_new

def LU_decomp(mat):
    matrix = mat.copy()
    row = len(matrix)
    col = len(matrix[0])
    A = []
    for i in range(row):
        for j in range(i,row):
            sum = 0.
            #print("j is ",j,"sum is",sum)
            if j>=1:    
                for k in range(i):
                    if i!=0:
                        sum = sum + matrix[i][k]*matrix[k][j]
            matrix[i][j] = matrix[i][j] - sum
            #print("matrix[",i+1,"][",j+1,"] is ",matrix[i][j],"sum was",sum)
            sum = 0.
            #print("j is ",j,"sum is",sum)
            for k in range(i):
                sum = sum + matrix[j][k]*matrix[k][i]
            if j!=i:
                matrix[j][i] = (matrix[j][i]- sum)/matrix[i][i]
                #print("matrix[",j+1,"][",i+1,"] is ",matrix[j][i],"sum was",sum)
    return matrix

def extract_LU(matrix):
    row = len(matrix)
    col = len(matrix[0])
    L = np.zeros((row,col))
    U = np.zeros((row,col))
    for i in range(row):
        L[i][i] = 1
        U[i][i] = matrix[i][i]
        for j in range(i+1,col):
            L[j][i] = matrix[j][i]
        for j in range(i):
            U[j][i] = matrix[j][i]
            
    return L,U

def inv_LU(mat):
    matrix = mat.copy()
    row = len(matrix)
    col = len(matrix[0])
    y = np.zeros((row,col))
    x = np.zeros((row,col))
    b = np.identity(row)

def solve_LU(mat,b_mat):
    matrix = mat.copy()
    b = b_mat.copy()
    row = len(matrix)
    col = len(matrix[0])
    y = np.zeros((row,1))
    x = np.zeros((row,1))
    
    for i in range(row):
        sum = 0.
        for j in range(i):
            if i>j:
                sum = sum + matrix[i][j]*y[j]
            if i==j:
                sum = sum + y[j]
        y[i] = b[i] - sum
    for i in range(row):
        sum = 0.
        for j in range(row-i,row):
            if row-1-i<j:
                #print("i ",row-1-i,"j ",j)
                sum = sum + matrix[row-1-i][j]*x[j]
        #print(sum)
        x[row-1-i] = (y[row-1-i]-sum)/matrix[row-1-i][row-1-i]
        
    return x

from itertools import product

def Conjugate_Gradient(mat_A,mat_b,x_old,tol=1e-4):
    x_new = x_old.copy()
    r_new = mat_b - np.dot(mat_A,x_new)
    d_new = r_new
    r_new_norm = np.linalg.norm(r_new)
    
    num_iter = 0
    x = [x_new]
    while r_new_norm > tol:
        A_d_new = np.dot(mat_A,d_new)
        r_new_r_new = np.dot(r_new,r_new)
        
        alpha = r_new_r_new/np.dot(d_new,A_d_new)
        x_new = x_new + (alpha*d_new)
        r_new = r_new - (alpha*A_d_new)
        beta = np.dot(r_new, r_new) / r_new_r_new
        d_new = r_new + (beta * d_new)
        
        num_iter += 1
        #if(np.linalg.norm(r_new)>=r_new_norm):
        #    break
        x.append(x_new)
        r_new_norm = np.linalg.norm(r_new)
        #print("comp",r_new_norm)
        if(num_iter>=30000):
            break
    print('Iteration: {} \t x = {} \t residual = {:.4f}'.format(num_iter, x_new, r_new_norm))    
    #print('\nSolution: \t x = {}'.format(x_new))
        
    return np.array(x_new)

def Cholesky_decomp(mat):
    A = mat.copy()
    row = A.shape[0]
    col = A.shape[1]
    L = np.matrix(np.zeros((row,col)))
    #print(L)
    for i in range(row):
        for j in range(i):
            #print("L[i,:j-1] =",i,j,L[i,:j-1])
            #print("L[j,:j-1]",i,j,L[j,:j-1])
            L[i,j] = (A[i,j]-np.dot(L[i,:j],L[j,:j].getH()))/L[j,j]
            #print("L[",i,j,"]=",L[i,j])
        #print("A[",i,i,"]=",A[i,i])
        #print("L[",i,",:",i,"]=",L[i,:i])
        L[i,i] = np.power(A[i,i]-np.dot(L[i,:i],L[i,:i].getH()),0.5)
        #print("L[",i,i,"]=",L[i,i])
    return L

import matplotlib.pyplot as plt

def polyfit(x,y,order):
    row = len(x)
    n = order  #for nth order poly fit
    mat_A = np.zeros((row,n+1))
    for i in range(row):
        for j in range(n+1):
            mat_A[i][j]= pow(x[i],j)
    mat_At = mat_A.T
    mat_A1 = np.dot(mat_At,mat_A)
    mat_y1 = np.dot(mat_At,y)
    x_ini = np.ones(n+1)
    mat_c = Conjugate_Gradient(mat_A1,mat_y1,x_ini)     #coefficient matrix
    coeffs = np.flip(mat_c)        # since mat_c has the order w0,w1,w2,w3
    
    return coeffs


def showfit_poly(x,y,coeffs):
    poly = np.poly1d(coeffs)

    x_new = np.linspace(x[0],x[-1])

    y_new = np.polyval(poly, x_new)

    plt.plot(x, y, "o", x_new, y_new)

    plt.xlim([x[0]-1, x[-1] + 1 ])
    
    return None

    
def phi(x):
    return 1, 2*x - 1, 8*pow(x,2) - 8*x + 1, 32*pow(x,3) - 48*pow(x,2) + 18*x - 1
    
    
def cheby_fit(x,y):
    row = len(x)
    mat_A = np.zeros((row,4))

    for i in range(row):
        for j in range(4):
            mat_A[i][j]= phi(x[i])[j]
    mat_At = mat_A.T
    mat_A1 = np.dot(mat_At,mat_A)
    mat_y1 = np.dot(mat_At,y)
    x_ini = np.ones(4)
    coeffs = Conjugate_Gradient(mat_A1,mat_y1,x_ini)     #coefficient matrix
    
    return coeffs

def cheby_y(x,coeffs):
    n = len(x) 
    y = np.zeros(n)
    y = coeffs[0]*phi(x)[0] + coeffs[1]*phi(x)[1] + coeffs[2]*phi(x)[2] + coeffs[3]*phi(x)[3]
    return y


def showfit_cheby(x,y,coeffs):
    x_new = np.linspace(x[0],x[-1])

    y_new = cheby_y(x_new,coeffs)

    plt.plot(x, y, "o", x_new, y_new)

    plt.xlim([x[0]-1, x[-1] + 1 ])
    
    return None

import time

seed = time.time()

def mlcg(am):
    global seed
    #print("seed_o",seed)
    seed = (am[0]*seed) % am[1]
    #print("seed_n",seed)
    return seed/am[1]

def monte_carlo(func,imp,a,b,n=1000):
    am = [572,16381]
    u = np.zeros(n)
    for i in range(n):
        u[i] = np.abs(mlcg(am))
    x = np.zeros(n)
    x = a + (b-a)*u
    if imp:
        g = func(x)*monte_carlo(imp,None,a,b,n)/imp(x)
    else:
        g = func(x)*(b-a)
    I = np.sum(g)/n
    
    return I
    