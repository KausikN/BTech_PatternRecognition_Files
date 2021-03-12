'''
Utils Functions
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, plot_implicit
import sympy

# Main Functions
def CovarianceMatrix(Pts):
    # Z = np.array(Pts)
    # Z = Z - np.mean(Z)

    # Z_T = np.transpose(Z)

    # CM = (1/(Z.shape[0]-1))*(np.matmul(Z_T, Z))
    CM = np.cov(np.transpose(Pts))

    return CM

def DecisionBoundary_2Class(Ps_1, Ps_2, P_w_1, P_w_2, display=False):
    Ps_1 = np.array(Ps_1)
    Ps_2 = np.array(Ps_2)

    # Find Cov Matrix
    CM_1 = CovarianceMatrix(Ps_1)
    CM_2 = CovarianceMatrix(Ps_2)

    if display:
        print("Cov Matrix 1")
        print(CM_1)
        print()
        print("Cov Matrix 2")
        print(CM_2)

    mean_1 = np.array([np.mean(Ps_1[:, 0]), np.mean(Ps_1[:, 1])])
    mean_2 = np.array([np.mean(Ps_2[:, 0]), np.mean(Ps_2[:, 1])])

    mean_diff = mean_1 - mean_2
    mean_sum = mean_1 + mean_2

    if display:
        print("Mean 1")
        print(mean_1)
        print("Mean 2")
        print(mean_2)
        print("Mean Diff")
        print(mean_diff)
        print("Mean Sum")
        print(mean_sum)

    # Case 1 and 2
    m = 0.0
    c = 0.0
    mean_diffdash = [mean_diff[0], mean_diff[1]]
    if np.equal(CM_1, CM_2).all():
        I = np.array([[1, 0], [0, 1]])
        # Case 1
        if np.equal(CM_1, I*CM_1[0, 0]).all():
            print("Case 1")
            sigmasq = CM_1[0, 0]
            Coeff = sigmasq / (mean_diff[0]**2 + mean_diff[1]**2)
            mean_diffdash = mean_diff
        # Case 2
        else:
            print("Case 2")
            invCM_1 = np.linalg.inv(CM_1)
            Coeff = 1/(np.matmul(np.array([mean_diff]), np.matmul(invCM_1, np.array([mean_diff]).T))[0, 0])
            mean_diffdash = np.matmul(invCM_1, mean_diff)
            
        m = -(mean_diffdash[0] / mean_diffdash[1])
        c = (0.5)*(mean_diffdash[0]*mean_sum[0] + mean_diffdash[1]*mean_sum[1]) - (Coeff)*(np.log(P_w_1/P_w_2))*(mean_diff[0]*mean_diffdash[0] + mean_diff[1]*mean_diffdash[1])
        c = c / mean_diffdash[1]
    return m, c

def GetBoundary(Pts):
    Pts= np.array(Pts)
    X = Pts[:, 0]
    Y = Pts[:, 1]
    boundary = np.array([[np.min(X), np.max(X)], [np.min(Y), np.max(Y)]])
    return boundary

def GeneratePoints(m, c, rangeValues):
    x = np.linspace(rangeValues[0], rangeValues[1], rangeValues[2])
    y = (m*x + c)
    Pts = np.stack((x, y), axis=1)
    return Pts

def PlotPoints(PsList, colors):
    for Ps, color in zip(PsList, colors):
        Ps = np.array(Ps)
        x, y = Ps[:, 0], Ps[:, 1]
        plt.scatter(x, y, c=color)
    # plt.show()

def PlotLines(PsList, colors):
    for Ps, color in zip(PsList, colors):
        Ps = np.array(Ps)
        x, y = Ps[:, 0], Ps[:, 1]
        plt.plot(x, y, c=color)
    # plt.show()

def DiscriminantFunctionEquation(Ps, P_w, display=False):
    Ps = np.array(Ps)
    Xsstrlist = []
    for i in range(Ps.shape[1]):
        Xsstrlist.append('x' + str(i+1))
    Xsstr = ' '.join(Xsstrlist)
    Xs = symbols(Xsstr)
    X = np.array([list(Xs)]).T
    eq = None

    Ps = np.array(Ps)

    # Find Cov Matrix
    CM = CovarianceMatrix(Ps)

    if display:
        print("Cov Matrix 1")
        print(CM)

    mean = []
    for i in range(Ps.shape[1]):
        mean.append(np.mean(Ps[:, i]))
    mean = np.array(mean)

    # Case 1 and 2
    m1 = np.array([mean]).T
    invCM = np.linalg.inv(CM)
    A1 = (-0.5)*invCM
    B1 = np.matmul(invCM, m1)
    C1 = (-0.5)*(np.matmul((m1).T, np.matmul(invCM, (m1)))[0, 0]) - (0.5)*(np.log(np.linalg.det(CM))) + np.log(P_w)
    G1 = np.matmul(X.T, np.matmul(A1, X)) + np.matmul(B1.T, X) + C1

    G1 = G1[0, 0]

    return G1

def DecisionBoundaryEquation_2Class(Ps_1, Ps_2, P_w_1, P_w_2, display=False):
    case = 3

    x, y = symbols('x y')
    X = np.array([[x, y]]).T
    eq = None

    Ps_1 = np.array(Ps_1)
    Ps_2 = np.array(Ps_2)

    # Find Cov Matrix
    CM_1 = CovarianceMatrix(Ps_1)
    CM_2 = CovarianceMatrix(Ps_2)

    if display:
        print("Cov Matrix 1")
        print(CM_1)
        print()
        print("Cov Matrix 2")
        print(CM_2)

    mean_1 = np.array([np.mean(Ps_1[:, 0]), np.mean(Ps_1[:, 1])])
    mean_2 = np.array([np.mean(Ps_2[:, 0]), np.mean(Ps_2[:, 1])])

    if display:
        print("Mean 1")
        print(mean_1)
        print("Mean 2")
        print(mean_2)

    # Case 1 and 2
    m1 = np.array([mean_1]).T
    m2 = np.array([mean_2]).T
    if np.equal(CM_1, CM_2).all():
        I = np.array([[1, 0], [0, 1]])
        # Case 1
        if np.equal(CM_1, I*CM_1[0, 0]).all():
            print("Case 1")
            case = 1
            sigmasq = CM_1[0, 0]
            Coeff = sigmasq / (np.matmul((m1 - m2).T, (m1 - m2))[0, 0])
            W = m1 - m2
            X0 = (0.5)*(m1 + m2) - (Coeff)*(np.log(P_w_1/P_w_2))*(m1 - m2)
            eq = np.dot(W.T, (X - X0))[0, 0]
        # Case 2
        else:
            print("Case 2")
            case = 2
            invCM_1 = np.linalg.inv(CM_1)
            Coeff = 1/(np.matmul((m1 - m2).T, np.matmul(invCM_1, (m1 - m2)))[0, 0])
            W = np.matmul(invCM_1, m1 - m2)
            X0 = (0.5)*(m1 + m2) - (Coeff)*(np.log(P_w_1/P_w_2))*(m1 - m2)
            eq = np.dot(W.T, (X - X0))[0, 0]
    # Case 3
    else:
        print("Case 3")
        case = 3
        # 1
        # invCM_1 = np.linalg.inv(CM_1)
        # A1 = (-0.5)*invCM_1
        # B1 = np.matmul(invCM_1, m1)
        # C1 = (-0.5)*(np.matmul((m1).T, np.matmul(invCM_1, (m1)))[0, 0]) - (0.5)*(np.log(np.linalg.det(CM_1))) + np.log(P_w_1)
        # G1 = np.matmul(X.T, np.matmul(A1, X)) + np.matmul(B1.T, X) + C1
        G1 = DiscriminantFunctionEquation(Ps_1, P_w_1)

        # 2
        # invCM_2 = np.linalg.inv(CM_2)
        # A2 = (-0.5)*invCM_2
        # B2 = np.matmul(invCM_2, m2)
        # C2 = (-0.5)*(np.matmul((m2).T, np.matmul(invCM_2, (m2)))[0, 0]) - (0.5)*(np.log(np.linalg.det(CM_2))) + np.log(P_w_2)
        # G2 = np.matmul(X.T, np.matmul(A2, X)) + np.matmul(B2.T, X) + C2
        G2 = DiscriminantFunctionEquation(Ps_2, P_w_2)

        eq = (G1 - G2)[0, 0]

    return eq, case

def PlotEquation(eq):
    plot_implicit(eq)

def Equation2Points(eq, rangeValues=[-1, 5, 100]):
    polyeq = sympy.poly(eq)
    terms = polyeq.terms()

    X = np.linspace(rangeValues[0], rangeValues[1], rangeValues[2])
    Pts = []

    # 1 var in terms (only y)
    if len(terms) > 0 and len(terms[0][0]) == 1:
        coeffs = [0.0, 0.0]
        for term in terms:
            coeffs[term[0][0]] = term[1]

        y = -coeffs[0]/coeffs[1]
        for x in X:
            Pts.append([x, y])
        Pts = np.array(Pts)
        LinesPts = [Pts]

    # 2 vars in terms
    elif len(terms) > 0 and len(terms[0][0]) == 2:
        coeffs = np.zeros((3, 3))
        for term in terms:
            coeffs[term[0][0], term[0][1]] = term[1]
        
        c, a_x, a_xx, a_xy, a_y, a_yy = tuple([coeffs[0, 0], coeffs[1, 0], coeffs[2, 0], coeffs[1, 1], coeffs[0, 1], coeffs[0, 2]])
        
        # Quadratic y
        if not a_yy == 0:
            Pts_Top = []
            Pts_Bottom = []
            for x in X:
                K = a_xx * (x**2) + a_x * x + c
                disc = (a_xy*x + a_y)**2 - 4*a_yy*K
                # Imaginary roots
                if disc < 0:
                    continue
                # Real roots
                else:
                    r_1 = (-(a_xy*x + a_y) + disc**(0.5)) / (2*a_yy)
                    r_2 = (-(a_xy*x + a_y) - disc**(0.5)) / (2*a_yy)
                    Pts_Top.append([x, r_1])
                    Pts_Bottom.append([x, r_2])
            LinesPts = [Pts_Top, Pts_Bottom]
            
        # Linear y
        else:
            for x in X:
                roots = []
                if not (a_xy*x + a_y) == 0:
                    roots.append((-a_xx*(x**2) - a_x*x - c) / (a_xy*x + a_y))
                for r in roots:
                    Pts.append([x, r])
            Pts = np.array(Pts)
            LinesPts = [Pts]
    
    return LinesPts

# Driver Code