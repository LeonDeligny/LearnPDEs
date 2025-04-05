# Physics-Informed Neural Networks (PINNs)

## **Project Goal**

The goal of this project is to develop a framework using a deep learning model to approximate solutions to Partial Differential Equations (PDEs) without requiring any input data. The Physics-Informed Neural Network (PINN) is trained solely on:
- **Physics Loss**: Derived from the governing equations (e.g., PDE = 0).
- **Boundary Conditions (BC) Loss**: Ensuring the solution satisfies the boundary constraints.

## **Objectives**

The project progresses through increasingly complex problems:
1. **Simple ODEs**:
   - Exponential: \( f' = f, \, f(0) = 1 \) :heavy_check_mark:
  
  ![Training Process](./assets/exponential.gif)

1. **Higher-Order ODEs**:
   - Example: \( f'' = -f, \, f(0) = 1, \, f'(0) = 0 \)
2. **PDEs**:
   - Example: \( \nabla f = 0 \)
3. **Navier-Stokes Equations**:
   - Solve fluid dynamics problems governed by the Navier-Stokes equations.

## **Features**

- Train neural networks to approximate solutions to ODEs and PDEs.
- No labeled data required; training is based on physics and boundary conditions.
- Scalable to more complex equations like Navier-Stokes.

## **Getting Started**

### **Prerequisites**

- Python 3.8+
- PyTorch (with MPS support for macOS)
- Additional dependencies:
  ```bash
  pip install -r requirements.txt