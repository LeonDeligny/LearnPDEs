**Project Goal**:
  Define a framework with a deep learning model that learns how to approximate a PDE without any input data.
  The Physics-Informed Neural Network has to train only on the Physics (PDE = 0) and Boundary conditions (BC = 0) losses.

**Objectives**:
  1. Start with a simple ODE (e.g. f' = f, f(0) = 1)
  2. Raise the difficulty (e.g. f'' = -f, f(0) = 1)
  3. Train on a PDE (e.g. Nabla(f) = 0)
  4. Finish with Navier-Stockes
