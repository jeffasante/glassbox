import torch
import torch.nn as nn
from glassbox import GlassBox

# 1. SETUP: Create a "Mystery Model"
# Let's say this model calculates: y = 3 * x0 + 5 * x1
print("Creating a Mystery Model (y = 3*x0 + 5*x1)...")
class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
        # Manually setting weights to simulate a trained model
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[3.0, 5.0]]))
            self.fc.bias.copy_(torch.tensor([0.0]))
            
    def forward(self, x):
        return self.fc(x)

mystery_model = LinearNet()

# 2. THE TEST: Can GlassBox figure it out WITHOUT data?
print("\n--- STARTING BLIND INTERROGATION ---")

# Notice: We define input_shape=(2,) because the model expects 2 inputs
gb = GlassBox(mystery_model, input_shape=(2,))

# NOTICE: We pass X_data=None! 
# GlassBox must generate its own data to figure out the math.
equation = gb.extract_formula(X_data=None)

gb.explain()