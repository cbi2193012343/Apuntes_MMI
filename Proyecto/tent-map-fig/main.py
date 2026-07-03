import torch
import einops
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

# Tent-map: mu min(x, 1-x)

# We wish to the bifurcation diagram, i.e. the long term values of x_n for each
# mu in [0,2]

mu_vals = torch.linspace(1, 2, 4000)
x_vals = torch.linspace(0, 1, 1000)
x, mu = torch.meshgrid(x_vals, mu_vals, indexing='ij')

n_iter = 1000
plot_title = "Tent Map Bifurcation Diagram"

for i in range(n_iter):
    x = mu * torch.min(x, 1 - x)




plt.figure(figsize=(10, 7))
plt.xlim(1, 2)
plt.ylim(0, 1)
plt.title(plot_title)
plt.xlabel(r'$\mu$')
plt.ylabel(r'$x$')
plt.scatter(mu, x, s=0.25, alpha=0.4, color='black', marker='.', linewidths=0)
plt.tight_layout()
plt.savefig('tent_map.png', dpi=600, bbox_inches='tight')
