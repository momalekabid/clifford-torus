"""visualize samples from the clifford torus distributions."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt

from clifford_torus import (
    CliffordPowerSphericalDistribution,
    CliffordTorusUniform,
)

torch.manual_seed(0)
out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)

fig, axes = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={"aspect": "equal"})
circle = torch.linspace(0, 2 * 3.14159265, 200)
for ax, kappa in zip(axes, [0.0, 1.0, 4.0, 16.0]):
    ax.plot(circle.cos(), circle.sin(), color="lightgray", lw=1)
    loc = torch.zeros(2)
    if kappa == 0.0:
        q = CliffordTorusUniform(dim=2)
    else:
        q = CliffordPowerSphericalDistribution(loc, torch.full((2,), kappa))
    z = q.rsample(torch.Size([500]))  # (500, 4)
    phase0 = torch.angle(torch.fft.fft(z, dim=-1)[..., 1])
    ax.scatter(phase0.cos(), phase0.sin(), s=6, alpha=0.5)
    ax.set_title(f"kappa={kappa}")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(
    "Samples on a single Clifford-torus factor S^1 at increasing concentration"
)
fig.savefig(
    os.path.join(out_dir, "circle_concentration.png"), dpi=150, bbox_inches="tight"
)
plt.close(fig)

fig = plt.figure(figsize=(14, 5))
d = 3
loc = torch.zeros(d)

ax1 = fig.add_subplot(1, 3, 1, projection="3d")
p_unif = CliffordTorusUniform(dim=d)
x_unif = p_unif.rsample(torch.Size([1500]))
ax1.scatter(x_unif[:, 0], x_unif[:, 1], x_unif[:, 2], s=4, alpha=0.4)
ax1.set_title("CliffordTorusUniform")

ax2 = fig.add_subplot(1, 3, 2, projection="3d")
q_low = CliffordPowerSphericalDistribution(loc, torch.full((d,), 1.0))
x_low = q_low.rsample(torch.Size([1500]))
ax2.scatter(x_low[:, 0], x_low[:, 1], x_low[:, 2], s=4, alpha=0.4, color="C1")
ax2.set_title("CliffordPowerSpherical, kappa=1")

ax3 = fig.add_subplot(1, 3, 3, projection="3d")
q_high = CliffordPowerSphericalDistribution(loc, torch.full((d,), 8.0))
x_high = q_high.rsample(torch.Size([1500]))
ax3.scatter(x_high[:, 0], x_high[:, 1], x_high[:, 2], s=4, alpha=0.4, color="C2")
ax3.set_title("CliffordPowerSpherical, kappa=8")

fig.suptitle("Real-valued samples (1st 3 of 2d=6), d=3")
fig.savefig(os.path.join(out_dir, "torus_3d_scatter.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Saved figures to {out_dir}")
