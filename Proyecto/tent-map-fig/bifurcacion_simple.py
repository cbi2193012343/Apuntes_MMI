import torch
import matplotlib.pyplot as plt

# Tent-map: mu min(x, 1-x)

mu_vals = torch.linspace(1, 2, 2000)
x_vals  = torch.linspace(0, 1, 500)
x, mu = torch.meshgrid(
    x_vals, mu_vals, indexing='ij')
# x[i,j]  = x_vals[i]
# mu[i,j] = mu_vals[j]

for i in range(1000):
    x = mu * torch.min(x, 1 - x)
    # torch.min(x, 1 - x)[i,j] =
    #   min(x[i,j], 1 - x[i,j])
    # x[i,j] = mu[i,j] *
    #   min(x[i,j], 1 - x[i,j])

plt.scatter(
    mu, x,
    s=0.25, alpha=0.4,
    color='black', marker='.',
)
plt.show()
