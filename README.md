# Schrödinger 2D PINN

A Physics-Informed Neural Network (PINN) to solve the **2D time-dependent Schrödinger equation** using the SIREN architecture (sinusoidal activation functions). This project simulates quantum wavefunction dynamics with high-fidelity visualizations, norm conservation tracking, and physically grounded design — built entirely from scratch in PyTorch.

> ⚛️ “Where physics meets machine learning — not as a black box, but as a white box with boundary conditions.”

---

## 🔍 Motivation

Quantum systems are notoriously hard to simulate, especially in higher dimensions. This project explores **PINNs as an alternative to traditional numerical solvers**, using neural networks trained directly on the governing physics — in this case, the 2D Schrödinger equation.

Rather than predicting outputs from data, the model **learns to respect physical laws**.

---

## 🧠 Core Features

- ✅ Solves the 2D time-dependent Schrödinger equation in bounded domains
- 🌀 Supports initial wave packets with user-defined center and momentum
- 🧱 Built using [SIREN (sinusoidal representation networks)](https://arxiv.org/abs/2006.09661) for smooth representation of high-frequency wavefunctions
- 🎥 Generates clean, arXiv-ready plots and animations of wavefunction evolution
- 📉 Tracks **norm conservation** and expectation values to verify physical consistency
- 🧪 Modular training and visualization pipeline, easy to extend to 3D or potentials

---

## 🧾 Equation Overview

The Schrödinger equation solved is:

\[
i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m} \nabla^2 \psi + V(x, y)\psi
\]

with customizable potential \( V(x, y) \). For this project, we simulate a **potential well** (zero inside, infinite at boundaries) with user-controlled initial Gaussian wave packets:

\[
\psi(x, y, 0) = \exp\left(-\frac{(x - x_0)^2 + (y - y_0)^2}{2\sigma^2} \right) \cdot e^{i(k_x x + k_y y)}
\]

---

## 🚀 Quick Start

Clone the repo and run the experiment:

```bash
git clone https://github.com/Moozzii/PINN-Schrodinger2D
cd schrodinger-2d-pinn
pip install -r requirements.txt
python experiment/schrodinger2d_pinn.experiment
