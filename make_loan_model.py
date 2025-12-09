import torch
import torch.nn as nn

class LoanModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 5 Inputs: [Age, Income, Debt, YearsEmployed, CreditScore]
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    print("Training LoanModel...")
    
    # GENERATE SYNTHETIC DATA
    # We create 1000 samples with 5 features
    X = torch.rand(1000, 5) 
    
    # THE SECRET FORMULA:
    # Risk Score = 2.0 * Income (x1) - 1.5 * Debt (x2)
    # We ignore Age (x0), Years (x3), and CreditScore (x4) to see if GlassBox ignores them too.
    y = (2.0 * X[:, 1] - 1.5 * X[:, 2]).reshape(-1, 1)

    # 3. TRAIN
    model = LoanModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for i in range(500):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        
    # 4. SAVE
    torch.save(model.state_dict(), "loan_model.pth")
    print("Saved 'loan_model.pth'")


'''
To use it in GlassBox

Code: Paste the class below.
Class Name: LoanModel
File: Upload the new loan_model.pth.


import torch.nn as nn

class LoanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

'''