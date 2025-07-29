import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import os

def Training_Loop(model, physics, epochs=6000, log_file="loss_logs.csv", plot_every_n_epochs=500):
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def lr_schedule(epoch):
        if epoch < 2000:
            return 1.0  
        else:
            return max(1e-5 / lr, 1.0 - (epoch - 2000) / (epochs - 2000))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    loss_logs = {
        "epoch": [],
        "total": [],
        "initial": [],
        "pde": [],
        "boundary": [],
        "norm_t0": [],
        "norm_t025": [],
        "norm_t05": [],
        "norm_t075": [],
        "norm_t1": [],
    }

    print("🚀 Starting Training Loop")
    for epoch in tqdm(range(epochs)):
        model.train()

        # Plot debug heatmaps
        if epoch % plot_every_n_epochs == 0:
            N = 100
            x = torch.linspace(-1, 1, N, device=device)
            y = torch.linspace(-1, 1, N, device=device)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            t_vals = [0.0, 0.25, 0.5, 0.75, 1.0]

            for t in t_vals:
                t_tensor = torch.full_like(xx, t)
                grid = torch.stack([xx, yy, t_tensor], dim=-1).reshape(-1, 3)

                with torch.no_grad():
                    output = model(grid)
                    u, v = output if isinstance(output, tuple) else (output, torch.zeros_like(output))
                    prob_density = (u**2 + v**2).reshape(N, N).cpu()

                plt.figure(figsize=(6, 5))
                plt.pcolormesh(xx.cpu(), yy.cpu(), prob_density, shading='gouraud', cmap='inferno')
                plt.axis('off')
                plt.tight_layout()
                os.makedirs("plots", exist_ok=True)
                save_path = f'plots/debug_heat_t{str(t).replace(".", "_")}_epoch{epoch}.png'
                plt.savefig(save_path, dpi=150)
                plt.close()
                print(f"[✓] Debug heatmap saved: {save_path}")

        # Dynamic loss weighting
        if epoch < 2500:
            weights = (10, 1, 10)
        elif epoch < 5000:
            weights = (15, 3, 15)
        else:
            weights = (20, 8, 40)

        loss, (L_init, L_phys, L_bound) = physics.compute_loss(model, *weights)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if epoch >= 1000:
            scheduler.step()

        # Logging losses
        loss_logs["epoch"].append(epoch)
        loss_logs["total"].append(loss.item())
        loss_logs["initial"].append(L_init.item())
        loss_logs["pde"].append(L_phys.item())
        loss_logs["boundary"].append(L_bound.item())

        with torch.no_grad():
            norm_t0 = physics.compute_norm(model, t_fixed=0.0).item()
            norm_t025 = physics.compute_norm(model, t_fixed=0.25).item()
            norm_t05 = physics.compute_norm(model, t_fixed=0.5).item()
            norm_t075 = physics.compute_norm(model, t_fixed=0.75).item()
            norm_t1 = physics.compute_norm(model, t_fixed=1.0).item()

        loss_logs["norm_t0"].append(norm_t0)
        loss_logs["norm_t025"].append(norm_t025)
        loss_logs["norm_t05"].append(norm_t05)
        loss_logs["norm_t075"].append(norm_t075)
        loss_logs["norm_t1"].append(norm_t1)

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:04d} | Total: {loss.item():.6f} | Init: {L_init.item():.4e} | PDE: {L_phys.item():.4e} | Bound: {L_bound.item():.4e}")
            print(f"Norms ➤ t=0: {norm_t0:.5f}, t=0.25: {norm_t025:.5f}, t=0.5: {norm_t05:.5f}, t=0.75: {norm_t075:.5f}, t=1: {norm_t1:.5f}")

        if epoch % 1000 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pt")

    df = pd.DataFrame(loss_logs)
    df.to_csv(log_file, index=False)
    print(f"📈 Loss log saved to {log_file}")

    torch.save(model.state_dict(), "checkpoints/final_checkpoint.pth")
    print("💾 Final model saved to checkpoints/final_checkpoint.pth")

    return model, loss_logs
