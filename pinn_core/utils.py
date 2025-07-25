import torch 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd


def plot_Schrodinger_wavefunction(model, t_fixed=0.5, resolution=100, xlim=(-1, 1), ylim=(-1, 1), save_path=None):
    model.eval()
    x = torch.linspace(xlim[0], xlim[1], resolution)
    y = torch.linspace(ylim[0], ylim[1], resolution)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    x_flat = xx.reshape(-1, 1)
    y_flat = yy.reshape(-1, 1)
    t = torch.full_like(x_flat, t_fixed)
    X = torch.cat([x_flat, y_flat, t], dim=1)

    with torch.no_grad():
        u, v = model(X)
        prob_density = (u**2 + v**2).reshape(resolution, resolution)

    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.pcolormesh(xx, yy, prob_density.cpu(), shading='gouraud', cmap='inferno')
    plt.colorbar(c, ax=ax)
    ax.set_title(f"Probability Density at t={t_fixed}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def animate_wavefunction(model, frames=60, resolution=100, save_path="wave_animation.gif"):
    fig, ax = plt.subplots(figsize=(6, 5))
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    mesh = None

    def update(frame):
        nonlocal mesh
        t_val = frame / (frames - 1)
        x_flat = xx.reshape(-1, 1)
        y_flat = yy.reshape(-1, 1)
        t_flat = torch.full_like(x_flat, t_val)
        X = torch.cat([x_flat, y_flat, t_flat], dim=1)

        with torch.no_grad():
            u, v = model(X)
            prob_density = (u**2 + v**2).reshape(resolution, resolution).cpu()

        ax.clear()
        mesh = ax.pcolormesh(xx, yy, prob_density, shading='gouraud', cmap='inferno')
        ax.set_title(f"t = {t_val:.2f}")
        ax.set_axis_off()
        return [mesh]

    ani = FuncAnimation(fig, update, frames=frames, blit=False)
    ani.save(save_path, writer='pillow', fps=10)
    plt.close()

def plot_loss_components(logs, save_path="plots/loss_components.png"):
    """
    Plot total loss and its components across training epochs.
    Parameters:
        logs: dictionary or pandas DataFrame with keys
              ["epoch", "total", "initial", "pde", "boundary"]
        save_path: where to save the plot
    """
    if isinstance(logs, dict):
        import pandas as pd
        logs = pd.DataFrame(logs)

    plt.figure(figsize=(10, 6))
    plt.plot(logs["epoch"], logs["total"], label="Total Loss", linewidth=2.0)
    plt.plot(logs["epoch"], logs["initial"], label="Initial Condition Loss")
    plt.plot(logs["epoch"], logs["pde"], label="PDE Residual Loss")
    plt.plot(logs["epoch"], logs["boundary"], label="Boundary Loss")

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Loss Components over Training")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📉 Saved: {save_path}")


def plot_norm_drift(csv_file="logs/wave.csv", save_path="plots/norm_drift.png"):
    """
    Plot norm drift at various time snapshots (t=0, 0.5, 1).
    Parameters:
        csv_file: CSV file saved from training loop
        save_path: where to save the plot
    """
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["norm_t0"], label="Norm at t=0.0", linestyle='-')
    plt.plot(df["epoch"], df["norm_t05"], label="Norm at t=0.5", linestyle='--')
    plt.plot(df["epoch"], df["norm_t1"], label="Norm at t=1.0", linestyle='-.')

    plt.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5, label="Ideal Norm")
    plt.xlabel("Epoch")
    plt.ylabel("Integrated |ψ|²")
    plt.title("Norm Conservation over Time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"🧮 Norm drift plot saved: {save_path}")
