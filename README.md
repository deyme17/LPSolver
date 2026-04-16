# 📊 Linear Programming & Integer Suite

A comprehensive optimization suite for solving **Linear Programming (LP)** and **Integer Programming (IP)** problems. Featuring a modular architecture, the application provides a GUI-driven approach to the **Simplex Method** and the **Branch and Bound** algorithm.

---

## 🚀 Key Features

* **Integer Programming**: Solve Mixed-Integer Linear Programming (MILP) problems using a robust **Branch and Bound** search tree.
* **Multiple Initialization Strategies**: Support for various methods to find the initial Basic Feasible Solution (BFS):
    * **Standard Slack**: For problems where slack variables form an immediate basis.
    * **Big-M Method**: Using large penalty coefficients ($M = 10^6$) for $\ge$ and $=$ constraints.
    * **Two-Phase Method**: A stable, two-stage optimization process to eliminate artificial variables before the main optimization.
* **Intuitive UI**: Built with **PyQt6**, featuring a dynamic input form, real-time constraint generation, and a dedicated results dashboard.
* **Extensible Architecture**: Clean interface-based design (`ISolver`, `IBFSFinder`) allows for easy integration of new algorithms.

---

## 🧠 Algorithmic Core

### 1. Simplex Solver
The engine transforms problems into **Standard Form** ($Ax = b, x \ge 0$) and iterates using a pivot-based Simplex Table. It handles:
* Minimization via objective negation.
* Slack and artificial variable injection.
* Unboundedness and Infeasibility detection.

### 2. Branch and Bound
For problems with integer constraints, the solver:
* Relaxes the problem to a continuous LP.
* Uses a **Priority Queue** (Heuristic: Best Relaxation Value) to explore the search tree.
* Prunes branches that are infeasible or worse than the current **Best Integer Solution**.

---

## 🛠️ Tech Stack

* **Python 3**: Core logic.
* **PyQt6**: Desktop interface and event handling.
* **NumPy**: Fast matrix operations and numerical stability.

---

## 🖥️ How to Run

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/deyme17/LPSolver.git
    cd LPSolver
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the application**:
    ```bash
    python main.py
    ```

---

## 📂 Project Structure

The project follows a modular directory structure to separate business logic, mathematical algorithms, and the user interface:

* **`core/`**: The heart of the application.
    * **`algorithms/`**: Implementation of the `SimplexAlgorithm`.
    * **`bfs/`**: Strategies for finding the initial Basic Feasible Solution (Basic, Big-M, Two-Phase).
    * **`solvers/`**: High-level solvers including `SimplexSolver` and `BranchAndBoundSolver`.
* **`view/`**: PyQt6 widgets and window definitions for the graphical interface.
* **`utils/`**: Shared components including data `containers`, `interfaces`, `validators`, and `constants`.
