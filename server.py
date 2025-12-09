import os
# Disable Julia's signal handling and multithreading to prevent crashes/hangs
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "no"
os.environ["PYTHON_JULIACALL_THREADS"] = "1"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
import shutil
from typing import Optional
from glassbox import GlassBox

app = FastAPI()

# MODEL DEFINITIONS
class PhysicsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x)

class KineticEnergyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), 
            nn.Linear(64, 64), nn.ReLU(), 
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

class LogicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 20), nn.ReLU(), nn.Linear(20, 1))
    def forward(self, x): return self.net(x)

# GLOBAL STATE (Single User Demo)
CURRENT_MODEL = None
INPUT_SHAPE = None

# API ENDPOINTS

@app.post("/api/load_model")
async def load_model(
    source: str = Form(...),
    demo_type: Optional[str] = Form(None),
    arch_type: Optional[str] = Form(None),
    model_code: Optional[str] = Form(None),
    class_name: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    global CURRENT_MODEL, INPUT_SHAPE
    
    try:
        if source == "demo":
            if demo_type == "physics":
                model = PhysicsNet()
                X = torch.linspace(-3, 3, 100).reshape(-1, 1)
                y = 2.5 * X**2 + torch.cos(X)
                opt = torch.optim.Adam(model.parameters(), lr=0.05)
                for _ in range(100):
                    loss = nn.MSELoss()(model(X), y)
                    opt.zero_grad(); loss.backward(); opt.step()
                CURRENT_MODEL = model
                INPUT_SHAPE = (1,)
                
            elif demo_type == "logic":
                model = LogicNet()
                X = torch.rand(500, 2) * 10
                y = ((X[:, 0] > 6) & (X[:, 1] < 3)).float().reshape(-1, 1)
                opt = torch.optim.Adam(model.parameters(), lr=0.02)
                for _ in range(100):
                    loss = nn.MSELoss()(model(X), y)
                    opt.zero_grad(); loss.backward(); opt.step()
                CURRENT_MODEL = model
                INPUT_SHAPE = (2,)
                
        elif source == "upload":
            if not file:
                return {"success": False, "error": "No file uploaded"}
            
            # Save temp file
            temp_filename = f"temp_{file.filename}"
            with open(temp_filename, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            # Initialize Architecture
            if arch_type and arch_type.startswith("kinetic"):
                model = KineticEnergyNet()
                INPUT_SHAPE = (1,)
            else:
                # Default fallback
                model = KineticEnergyNet()
                INPUT_SHAPE = (1,)
            
            # Load Weights
            state_dict = torch.load(temp_filename)
            model.load_state_dict(state_dict)
            model.eval()
            CURRENT_MODEL = model
            
            return {"success": True, "message": "Model uploaded successfully"}

        elif source == "custom_code":
            if not file or not model_code or not class_name:
                return {"success": False, "error": "Missing file, code, or class name"}

            # Save temp file
            temp_filename = f"temp_{file.filename}"
            with open(temp_filename, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Dynamic Code Execution
            local_scope = {}
            global_scope = {'torch': torch, 'nn': nn}
            
            try:
                exec(model_code, global_scope, local_scope)
            except Exception as e:
                return {"success": False, "error": f"Code execution failed: {str(e)}"}
            
            ModelClass = local_scope.get(class_name)
            if ModelClass is None:
                return {"success": False, "error": f"Class '{class_name}' not found in code"}
            
            try:
                model = ModelClass()
                state_dict = torch.load(temp_filename)
                model.load_state_dict(state_dict)
                model.eval()
                
                # Guess input shape
                first_layer_weight = list(model.parameters())[0]
                in_features = first_layer_weight.shape[1]
                INPUT_SHAPE = (in_features,)
                CURRENT_MODEL = model
                
                return {"success": True, "message": f"Custom model {class_name} loaded successfully"}
            except Exception as e:
                return {"success": False, "error": f"Model instantiation failed: {str(e)}"}

    except Exception as e:
        return {"success": False, "error": str(e)}
    
    return {"success": True, "message": "Model loaded successfully"}

class ExtractRequest(BaseModel):
    mode: str
    complexity: float

@app.post("/api/extract")
async def extract_model(req: ExtractRequest):
    global CURRENT_MODEL, INPUT_SHAPE
    
    if CURRENT_MODEL is None:
        return {"success": False, "error": "No model loaded"}
    
    try:
        gb = GlassBox(CURRENT_MODEL, input_shape=INPUT_SHAPE)
        
        if req.mode == "math":
            result = gb.extract_formula(complexity_penalty=req.complexity)
            note = None
            if "x0**2" in result:
                note = "Quadratic Law Detected!"
            return {"success": True, "result": f"y = {result}", "note": note}
            
        else: # logic
            result = gb.extract_logic(feature_names=["Income", "Debt"])
            return {"success": True, "result": result}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

# Serve Static Files
app.mount("/", StaticFiles(directory="web", html=True), name="static")
