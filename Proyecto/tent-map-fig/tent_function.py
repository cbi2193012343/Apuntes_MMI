import torch
import matplotlib.pyplot as plt

# Tent-map: T_mu(x) = mu min(x, 1-x)

x = torch.linspace(0, 1, 1000)

plt.figure(figsize=(7, 5))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r'$x$')
plt.ylabel(r'$T_\mu(x)$')

for mu in [1.0, 1.5, 2.0]:
    T = mu * torch.min(x, 1 - x)
    plt.plot(x, T, label=fr'$\mu = {mu}$')

plt.legend()
file_name = 'tent_function.png'
plt.savefig(file_name, dpi=300, bbox_inches='tight')
print(f"Saved to {file_name}")
