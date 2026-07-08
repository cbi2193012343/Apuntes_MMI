import torch
import matplotlib.pyplot as plt


mu = torch.linspace(2.5, 4.0, 6000)
x  = torch.full_like(mu, 0.5)

n_transient = 1000
n_keep      = 900

print("Discarding transients of logistic map...")
for i in range(n_transient):
    x = mu * x * (1 - x)

# Keep the asymptotic orbit: one column of samples per iteration
print("Collecting asymptotic orbit...")
MU = mu.expand(n_keep, -1).clone()
X  = torch.empty(n_keep, mu.shape[0])
for i in range(n_keep):
    x = mu * x * (1 - x)
    X[i] = x


print("Plotting...")
plt.figure(figsize=(6.5, 6))
plt.xlim(2.5, 4.0)
plt.ylim(0, 1)
plt.xlabel(r'$\mu$', fontsize=20)
plt.ylabel(r'$x$', fontsize=20)
plt.title('Logistic Map Bifurcation Diagram', fontsize=18)
plt.tick_params(labelsize=15)

color = torch.zeros(X.shape + (4,))

# The pre-chaotic cascade lives on a handful of thin branches, so it needs a
# strong alpha to be seen; the chaotic band fills the plane densely and would
# smear to solid black at the same alpha. Ramp alpha down as mu grows so both
# the delicate forks on the left and the fine structure on the right survive.
color[:, :, 3] = torch.clamp(0.9 - 0.36 * (MU - 2.5), 0.045, 1)


plt.scatter(
    MU.flatten().numpy(),
    X.flatten().numpy(),
    color=color.reshape(-1, 4).numpy(),
    s=0.055,
    marker='.',
    linewidths=0,
)
file_name = 'logistic_map.png'
plt.savefig(file_name, dpi=600, bbox_inches='tight')
print(f"Saved to {file_name}")
