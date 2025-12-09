import numpy as np
import torch

'''This file generates synthetic data to quiz the Black Box.'''
class Interrogator:
    @staticmethod
    def generate_probe_data(input_shape, n_samples=2000, bounds=(-5, 5)):
        """
        Generates random data to 'quiz' the neural network.
        """
        # input_shape should be a tuple, e.g., (2,) for 2 features
        n_features = input_shape[0]
        
        print(f"GlassBox Interrogator: Generating {n_samples} synthetic inputs...")
        
        # 1. Uniform Noise (Exploring the whole space)
        # Random numbers between bounds[0] and bounds[1]
        low, high = bounds
        X = np.random.uniform(low, high, size=(n_samples, n_features))
        
        return X