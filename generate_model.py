import torch
import torch.nn as nn
import os

# DEFINE THE ARCHITECTURE
# Let's make a model that learns a specific Physics formula:
# Energy = 0.5 * Mass * Velocity^2  (E = 0.5 * v^2) assuming mass=1
class KineticEnergyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # A slightly deeper network to make it "Black Box"
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    print("Training Kinetic Energy Model (E = 0.5 * v^2)...")
    
    #  GENERATE TRAINING DATA
    # Velocity from -10 to 10
    X = torch.linspace(-5, 5, 500).reshape(-1, 1)
    # Target: 0.5 * v^2
    y = 0.5 * X**2

    # TRAIN
    model = KineticEnergyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(500):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.5f}")

    filename = "kinetic_energy.pth"
    torch.save(model.state_dict(), filename)
    print(f"\nModel saved to '{filename}'")
    print("Test phase: Run the next script to reverse engineer this file.")