import torch
import matplotlib.pyplot as plt

# Logistic map: mu x (1 - x)

mu = torch.linspace(2.5, 4.0, 2000)
x  = torch.full_like(mu, 0.5)
X  = torch.empty(300, mu.shape[0])

for i in range(700):
    x = mu * x * (1 - x)
    # x[i] = mu[i] * x[i] * (1 - x[i])

for i in range(300):
    x = mu * x * (1 - x)
    # x[j] = mu[j] * x[j] * (1 - x[j])
    X[i] = x
    # X[i,j] = x[j]

MU = mu.expand_as(X)
# MU[i,j] = mu[j]

plt.scatter(
    MU, X,
    s=0.2, alpha=0.25,
    color='black', marker='.',
)

plt.show()
