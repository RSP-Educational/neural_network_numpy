import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class Module:
    """Base class for all modules (layers, activations, etc.)"""
    def forward(self, input:np.ndarray):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

def task_1_1_solution():
    img = cv.imread('images/task_1_1_solution.png')
    plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

def task_1_2_solution():
    img = cv.imread('images/task_1_2_solution.png')
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

def task_1_3_solution():
    img = cv.imread('images/task_1_3_solution.png')
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

def test_ReLU(relu):
    class ReLU(Module):
        """ReLU activation function"""
        def forward(self, input):
            self.input = input.copy()
            return np.maximum(0., input)
        
        def backward(self, grad_output):
            grad_input = grad_output.copy()
            grad_input[self.input <= 0] = 0
            return grad_input
        
    relu_target = ReLU()
    
    x = np.linspace(-1, 1, 20)
    expected_forward = relu_target.forward(x)
    forward_output = relu.forward(x)
    if not np.array_equal(forward_output, expected_forward):
        print(f"❌ ReLU Vorwärtspropagation fehlgeschlagen.\ninput:\n{x},\nerwartete Ausgabe:\n{expected_forward},\nerhaltene Ausgabe:\n{forward_output}")
    else:
        print("✅ ReLU Vorwärtspropagation erfolgreich")

    grad_output = x.copy()
    expected_backward = relu_target.backward(grad_output)
    backward_output = relu.backward(grad_output)
    if not np.array_equal(backward_output, expected_backward):
        print(f"❌ ReLU Rückwärtspropagation fehlgeschlagen.\ngrad_output:\n{grad_output},\nerwartete Ausgabe:\n{expected_backward},\nerhaltene Ausgabe:\n{backward_output}")
    else:
        print("✅ ReLU Rückwärtspropagation erfolgreich")

    z = np.linspace(-1, 1, 500)
    g_z = relu.forward(z)
    grad_g_z = relu.backward(np.ones_like(z))
    grad_g_z[grad_g_z != 0] = 1
    plt.plot(z, g_z, label="$g(z)$", color='blue')
    plt.plot(z, grad_g_z, label="$g'(z)$", color='red')
    plt.xlabel("$z$")
    plt.hlines([0], xmin=-1, xmax=1, colors='black', linestyles='-', linewidth=0.5)
    plt.vlines([0], ymin=-1, ymax=1, colors='black', linestyles='-', linewidth=0.5)
    plt.xlim(-1, 1)
    plt.ylim(-1.05, 1.05)
    plt.legend()
    plt.show()

def test_sigmoid(sigmoid):
    class Sigmoid(Module):
        """Sigmoid activation function"""
        def forward(self, input):
            self.output = 1 / (1 + np.exp(-input))
            return self.output
        
        def backward(self, grad_output):
            return grad_output * (self.output * (1 - self.output))
        
    sigmoid_target = Sigmoid()
    
    x = np.linspace(-1, 1, 20)
    expected_forward = sigmoid_target.forward(x)
    forward_output = sigmoid.forward(x)
    if not np.allclose(forward_output, expected_forward):
        print(f"❌ Sigmoid Vorwärtspropagation fehlgeschlagen.\ninput:\n{x},\nerwartete Ausgabe:\n{expected_forward},\nerhaltene Ausgabe:\n{forward_output}")
    else:
        print("✅ Sigmoid Vorwärtspropagation erfolgreich")

    grad_output = np.linspace(-1, 1, 20)
    expected_backward = sigmoid_target.backward(grad_output)
    backward_output = sigmoid.backward(grad_output)
    if not np.allclose(backward_output, expected_backward):
        print(f"❌ Sigmoid Rückwärtspropagation fehlgeschlagen.\ngrad_output:\n{grad_output},\nerwartete Ausgabe:\n{expected_backward},\nerhaltene Ausgabe:\n{backward_output}")
    else:
        print("✅ Sigmoid Rückwärtspropagation erfolgreich")

    xmin, xmax = -5, 5
    z = np.linspace(xmin, xmax, 500)
    g_z_hat = sigmoid.forward(z)
    g_z = 1 / (1 + np.exp(-z))
    grad_g_z_hat = sigmoid.backward(np.ones_like(z))
    grad_g_z = (g_z * (1 - g_z))
    plt.plot(z, g_z, color='gray', linewidth=0.7)
    plt.plot(z, grad_g_z, color='gray', linewidth=0.7)
    plt.plot(z, g_z_hat, label="$g(z)$", color='blue')
    plt.plot(z, grad_g_z_hat, label="$g'(z)$", color='red')
    plt.xlabel("$z$")
    plt.hlines([0], xmin=xmin, xmax=xmax, colors='black', linestyles='-', linewidth=0.5)
    plt.vlines([0], ymin=-1, ymax=1, colors='black', linestyles='-', linewidth=0.5)
    plt.xlim(xmin, xmax)
    plt.ylim(-1.0, 1.0)
    plt.legend()
    plt.show()

def test_tanh(tanh):
    class Tanh(Module):
        """Tanh activation function"""
        def forward(self, input):
            self.output = np.tanh(input)
            return self.output
        
        def backward(self, grad_output):
            return grad_output * (1 - self.output ** 2)
        
    tanh_target = Tanh()

    x = np.linspace(-1, 1, 20)
    expected_forward = tanh_target.forward(x)
    forward_output = tanh.forward(x)
    if not np.allclose(forward_output, expected_forward):
        print(f"❌ Tanh Vorwärtspropagation fehlgeschlagen.\ninput:\n{x},\nerwartete Ausgabe:\n{expected_forward},\nerhaltene Ausgabe:\n{forward_output}")
    else:
        print("✅ Tanh Vorwärtspropagation erfolgreich")

    grad_output = np.linspace(-1, 1, 20)
    expected_backward = tanh_target.backward(grad_output)
    backward_output = tanh.backward(grad_output)
    if not np.allclose(backward_output, expected_backward):
        print(f"❌ Tanh Rückwärtspropagation fehlgeschlagen.\ngrad_output:\n{grad_output},\nerwartete Ausgabe:\n{expected_backward},\nerhaltene Ausgabe:\n{backward_output}")
    else:
        print("✅ Tanh Rückwärtspropagation erfolgreich")

    xmin, xmax = -5, 5
    z = np.linspace(xmin, xmax, 500)
    g_z_hat = tanh.forward(z)
    g_z = np.tanh(z)
    grad_g_z_hat = tanh.backward(np.ones_like(z))
    grad_g_z = 1 - g_z ** 2
    plt.plot(z, g_z, color='gray', linewidth=0.7)
    plt.plot(z, grad_g_z, color='gray', linewidth=0.7)
    plt.plot(z, g_z_hat, label="$g(z)=tanh(z)$", color='blue')
    plt.plot(z, grad_g_z_hat, label="$g'(z)=1-tanh^2(z)$", color='red')
    plt.xlabel("$z$")
    plt.hlines([0], xmin=xmin, xmax=xmax, colors='black', linestyles='-', linewidth=0.5)
    plt.vlines([0], ymin=-1, ymax=1, colors='black', linestyles='-', linewidth=0.5)
    plt.xlim(xmin, xmax)
    plt.ylim(-1.02, 1.02)
    plt.legend(loc='lower right')
    plt.show()