import sympy
from glassbox import GlassBox

import torch
import torch.nn as nn

#load the class definition to load weights from generate_model.py
class KineticEnergyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

print("--- LOADING SAVED MODEL ---")

loaded_model = KineticEnergyNet()
try:
    loaded_model.load_state_dict(torch.load("kinetic_energy.pth"))
    loaded_model.eval() # Set to evaluation mode (crucial!)
    print(" Model weights loaded successfully.")
except FileNotFoundError:
    print(" Error: 'kinetic_energy.pth' not found. Run generate_model.py first.")
    exit()

print("\n--- STARTING GLASSBOX ---")
print("Target: We hope to find '0.5 * x^2'")

# Initialize GlassBox
# Note: We MUST provide input_shape because the loaded model has no data attached.
gb = GlassBox(loaded_model, input_shape=(1,))

# Run Extraction (Math Mode)
# We use a strict penalty to force it to find the clean '0.5' and 'square'
equation = gb.extract_formula(complexity_penalty=0.02)

print("\n" + "="*40)
print(f"EXTRACTED EQUATION:  {equation}")
print("="*40)

# Optional: Verification
# Let's check if the equation string actually contains what we expect
if "x0**2" in equation or "square(x0)" in equation:
    print("\n SUCCESS: Quadratic relationship detected.")
else:
    print("\n WARNING: Quadratic relationship not clearly found.")