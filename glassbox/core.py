import os

# Disable Julia's signal handling to prevent crashes on macOS
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "no"
# Disable Julia's multithreading to prevent segfaults when signal handling is disabled
os.environ["PYTHON_JULIACALL_THREADS"] = "1"
import sympy

from pysr import PySRRegressor

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text

from .interrogator import Interrogator  # Import your new tool

class GlassBox:
    def __init__(self, model, input_shape=None):
        self.teacher = model
        self.student_model = None
        self.student_equation = None
        self.mode = None
        self.input_shape = input_shape # Needed for generating data

    def _get_data(self, X_data):
        """Internal helper: Use provided data OR generate synthetic data."""
        if X_data is not None:
            return self._to_numpy_and_tensor(X_data)
        
        # If no data provided, we must interrogate!
        if self.input_shape is None:
            raise ValueError("If no X_data is provided, you must provide input_shape=(n_features,) in __init__")
            
        print("No training data provided. Generating synthetic probe data...")
        X_synthetic = Interrogator.generate_probe_data(self.input_shape)
        return self._to_numpy_and_tensor(X_synthetic)

    def _to_numpy_and_tensor(self, X_data):
        if isinstance(X_data, torch.Tensor):
            X_tensor = X_data
            X_numpy = X_data.detach().numpy()
        else:
            X_numpy = X_data
            X_tensor = torch.tensor(X_data, dtype=torch.float32)
        return X_numpy, X_tensor

    def predict_teacher(self, X_tensor):
        with torch.no_grad():
            preds = self.teacher(X_tensor).numpy()
        return preds

    # METHODS
    # def extract_formula(self, X_data=None, complexity_penalty=0.01):
    #     self.mode = 'math'
    #     X_numpy, X_tensor = self._get_data(X_data)
    #     y_teacher = self.predict_teacher(X_tensor)
        
    #     print("GlassBox [Math]: Fitting symbolic regression...")
    #     self.student_model = PySRRegressor(
    #         niterations=40,
    #         binary_operators=["+", "*", "-", "/"],
    #         unary_operators=["cos", "sin", "square"],
    #         model_selection="best",
    #         parsimony=complexity_penalty,
    #         verbosity=0
    #     )
    #     self.student_model.fit(X_numpy, y_teacher)
    #     self.student_equation = str(self.student_model.sympy())
    #     return self.student_equation

    def extract_logic(self, X_data=None, feature_names=None, max_depth=3):
        self.mode = 'logic'
        X_numpy, X_tensor = self._get_data(X_data)
        y_teacher = self.predict_teacher(X_tensor).ravel()

        print("GlassBox [Logic]: Distilling decision tree...")
        self.student_model = DecisionTreeRegressor(max_depth=max_depth)
        self.student_model.fit(X_numpy, y_teacher)
        
        self.student_equation = "Logic Tree"
        return export_text(self.student_model, feature_names=feature_names)

    def explain(self):
        print(f"\n[GlassBox Explanation]: {self.student_equation}")


    def extract_formula(self, X_data=None, complexity_penalty=0.01):
        self.mode = 'math'
        X_numpy, X_tensor = self._get_data(X_data)
        y_teacher = self.predict_teacher(X_tensor)
        
        print("GlassBox [Math]: Fitting symbolic regression...")
        self.student_model = PySRRegressor(
            niterations=15,    # Reduced from 40 for speed
            populations=10,    # Reduced population size for speed
            binary_operators=["+", "*", "-", "/"],
            unary_operators=["cos", "sin", "square"],
            model_selection="best",
            parsimony=complexity_penalty,
            verbosity=0,
            # Safety settings for macOS/Streamlit
            parallelism='serial', # Updated from deprecated multithreading=False
            timeout_in_seconds=60
        )
        self.student_model.fit(X_numpy, y_teacher)
        
        raw_eq = self.student_model.sympy() 
        clean_eq = sympy.simplify(raw_eq)  
        self.student_equation = str(clean_eq)
        # --------------------------------
        
        return self.student_equation