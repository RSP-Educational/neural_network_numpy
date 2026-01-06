import numpy as np

def generate_linear_data(m:float = 0., n:float = 0., num_samples:int = 100, noise:float = 0.) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1, 1, num_samples)
    y = m * x + n + noise * np.random.randn(num_samples)
    return x.reshape(-1, 1), y.reshape(-1, 1)

def generate_sinusoidal_data(num_samples:int = 100, noise:float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1, 1, num_samples)
    y = np.sin(np.pi * x) + noise * np.random.randn(num_samples) * np.cos(np.pi * x)
    return x.reshape(-1, 1), y.reshape(-1, 1)