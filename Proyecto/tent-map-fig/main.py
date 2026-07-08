import torch
import matplotlib.pyplot as plt


mu    = torch.linspace(1, 2, 6000)
x     = torch.linspace(0, 1, 2000)
X, MU = torch.meshgrid(x, mu, indexing = 'ij')

n_iter = 1000

print("Computing long-term behavior of tent map...")
for i in range(n_iter):
    X = MU * torch.min(X, 1 - X)




print("Plotting...")
plt.figure(figsize=(6.5, 6))
plt.xlim(1, 2)
plt.ylim(0, 1)
plt.xlabel(r'$\mu$', fontsize=20)
plt.ylabel(r'$x$', fontsize=20)
plt.title('Tent Map Bifurcation Diagram', fontsize=18)
plt.tick_params(labelsize=15)

# Set the color of each point to black
color = torch.zeros(X.shape + (4,))

# Modify the alpha channel based on the value of mu,
# since the density of points is higher for higher mu

# eyeballing this: mu = 1.2 has significant details and is quite dense
# so let's have the minimum alpha at it and increase linearly
# So that details on the right are still visible
color[:, :, 3] = torch.clamp(1.5 * (MU - 1.2), 0.06, 1)




plt.scatter(
    MU.flatten().numpy(),
    X.flatten().numpy(),
    color=color.reshape(-1, 4).numpy(),
    s=0.065,
    marker='.',
    linewidths=0,
)
file_name = 'tent_map.png'
plt.savefig(file_name, dpi=600, bbox_inches='tight')
print(f"Saved to {file_name}")
