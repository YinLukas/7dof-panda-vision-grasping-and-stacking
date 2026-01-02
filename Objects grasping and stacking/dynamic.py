import numpy as np

def F(x):
    x1, x2 = x
    F1 = 3 + x1 + 2*x2 - (x1**2 + 2*x1*x2)
    F2 = 4 + 3*x1 + 4*x2 - (x1*x2 + 2*x2**2)
    return np.array([F1, F2], dtype=float)

def J(x):
    x1, x2 = x
    J11 = 1 - 2*x1 - 2*x2
    J12 = 2 - 2*x1
    J21 = 3 - x2
    J22 = 4 - x1 - 4*x2
    return np.array([[J11, J12],
                     [J21, J22]], dtype=float)

def newton(F, J, x0, tol=1e-10, max_iter=50):
    x = np.array(x0, dtype=float)
    for k in range(max_iter):
        Fx = F(x)
        if np.linalg.norm(Fx) < tol:
            # converged
            # print(f"Converged in {k} steps")
            return x
        Jx = J(x)
        # solve Jx * dx = -F(x)
        dx = np.linalg.solve(Jx, -Fx)
        x = x + dx
    # print("Did not fully converge")
    return x

sol1 = newton(F, J, x0=[0, 0])
sol2 = newton(F, J, x0=[2, 2])
sol3 = newton(F, J, x0=[-2, 1])
sol4 = newton(F, J, x0=[-4, 2])
print(sol1, sol2, sol3, sol4)
