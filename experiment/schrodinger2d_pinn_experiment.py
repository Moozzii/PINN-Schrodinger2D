import os
import json
import torch

from equation.schrodinger2d_pinn import Schrodinger_2D
from pinn_core.model import PINN_ID
from pinn_core.training import Training_Loop
from pinn_core.utils import (
    plot_Schrodinger_wavefunction,
    plot_loss_components,
    plot_norm_drift,
    animate_wavefunction,
)

# ==== Setup ====
os.makedirs("plots", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("animations", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ==== Define Model & Physics ====
model = PINN_ID(input_dim=3, hidden_dim=128, num_layers=6, omega_0=30.0)
physics = Schrodinger_2D(x0=0.5, y0=0.5, kx=4, ky=4)

# ==== Train Model ====
model.train()
epochs = 8000
log_csv_path = "logs/wave.csv"
model_trained, loss_logs = Training_Loop(model, physics, epochs=epochs, log_file=log_csv_path)

# ==== Save Final Model ====
final_model_path = "checkpoints/wave_final.pth"
torch.save(model_trained.state_dict(), final_model_path)

# ==== Save Metadata for Reuse ====
metadata = {
    "epochs": epochs,
    "final_model": final_model_path,
    "log_csv": log_csv_path,
    "kx": physics.kx,
    "ky": physics.ky,
    "x0": physics.x0,
    "y0": physics.y0
}
with open("logs/train_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
print("📝 Saved metadata to logs/train_metadata.json")

# ==== Reload Model for Evaluation ====
model.load_state_dict(torch.load(final_model_path))
model.eval()

# ==== Plot Time Snapshots ====
for t in [0, 0.25, 0.5, 0.75, 1.0]:
    plot_Schrodinger_wavefunction(model, t_fixed=t, save_path=f"plots/wave_at_t{t}.png")

# ==== Loss & Norm Drift Plots ====
plot_loss_components(loss_logs, save_path="plots/loss_components.png")
plot_norm_drift(csv_file=log_csv_path, save_path="plots/wave_norm_drift.png")

# ==== Animate Evolution ====
animate_wavefunction(model, frames=60, save_path="animations/wave_animation.gif")

print("✅ All outputs saved and training completed.")
