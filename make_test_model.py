import torch
import torch.nn as nn

# 1. Define the Architecture (We will copy-paste this later)
class SecretModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple network that learns y = x^2 + 5
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# 2. Train it
print("Training 'SecretModel' (Target: y = x^2 + 5)...")
model = SecretModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
loss_fn = nn.MSELoss()

# Data: x from -5 to 5
X = torch.linspace(-5, 5, 500).reshape(-1, 1)
y = X**2 + 5  # The secret formula

for epoch in range(200):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

# 3. Save weights
torch.save(model.state_dict(), "my_secret_model.pth")
print("✅ Saved 'my_secret_model.pth'")