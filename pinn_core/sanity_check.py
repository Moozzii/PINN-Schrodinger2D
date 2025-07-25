import torch
import matplotlib.pyplot as plt

def sanity_check(model, physics, t=0.0, title='Sanity Check at t=0'):
    model.eval()
    with torch.no_grad():
        # Grid
        x_vals = torch.linspace(0, 1, 200)
        y_vals = torch.linspace(0, 1, 200)
        X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")
        x_flat = X.reshape(-1, 1)
        y_flat = Y.reshape(-1, 1)

        # Normalize to [-1, 1]
        x_norm = 2 * x_flat - 1
        y_norm = 2 * y_flat - 1

        t_input = 2 * torch.full_like(x_flat, t) - 1
        input_grid = torch.cat([x_norm, y_norm, t_input], dim=1)

# Continue as is...


        # Model prediction
        u, v = model(input_grid)
        u = u.reshape(200, 200)
        v = v.reshape(200, 200)
        prob = u**2 + v**2

        # Min/max
        print(f"--- {title} ---")
        print(f"Real part: min={u.min().item():.5f}, max={u.max().item():.5f}")
        print(f"Imag part: min={v.min().item():.5f}, max={v.max().item():.5f}")
        print(f"Prob dens: min={prob.min().item():.5f}, max={prob.max().item():.5f}")
        print(f"Norm ≈ {prob.sum().item() * (1/200)**2:.5f}")

        # Plots
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        im1 = axs[0].imshow(u.cpu(), cmap='RdBu', origin='lower', extent=[0, 1, 0, 1])
        axs[0].set_title('Real Part')
        plt.colorbar(im1, ax=axs[0])

        im2 = axs[1].imshow(v.cpu(), cmap='RdBu', origin='lower', extent=[0, 1, 0, 1])
        axs[1].set_title('Imaginary Part')
        plt.colorbar(im2, ax=axs[1])

        im3 = axs[2].imshow(prob.cpu(), cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
        axs[2].set_title('Probability Density')
        plt.colorbar(im3, ax=axs[2])

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
