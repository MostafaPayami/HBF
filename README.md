# HBF
Hybrid Beamforming Methods for mmWave MIMO Systems
The Python and MATLAB codes have been used for the experiments and results presented in the following paper [1]. 

[1] M. Payami, and S. D. Blostein, “Hybrid Beamforming for Massive MIMO Systems: A Challenging Problem with A Simple Solution” (2025).

The Python codes are TensorFlow/Keras models for AE-HBFnet.

The following software and libraries are required:
1) MATLAB: Version R2023b or later
2) TensorFlow: Version 2.19.0 or later
3) Keras: Version 3.8.0 or later

***************************************************************************************************

Paper's Abstract: 

Hybrid analog and digital beamforming is a key technique that can address the high cost and power consumption of fully-digital precoding, and thereby facilitates practical implementation(s) of millimeter wave (mmWave) massive multiple-input multiple-output (MIMO) wireless systems. A major challenge is however that the optimal design of hybrid beamforming architectures involves solving a high-dimensional and nonconvex optimization problem, being recast as a constrained matrix factorization problem for point-to-point mmWave massive MIMO communications. To accomplish the hybrid precoder design, the nonconvex optimization problem is first transformed (in)to an equivalent unconstrained system of nonlinear equations, and then an algorithm called SNQ-NDJ is developed to solve the derived system of equations using the Newton method with diagonal Jacobian. Consisting of only elementary function(s) evaluations and matrix multiplication, the proposed SNQ-NDJ method is simple, effective and fast, obviating any need for computing the inverse or pseudo-inverse of a matrix, prevalent in conventional methods. Furthermore, a model-driven neural network AE-HBFnet is designed based on self-supervised learning paradigm to perform the hybrid beamforming task. The AE-HBFnet is a lightweight generative model whose architecture is inspired by autoencoders. Next, the proposed frameworks are extended to broadband mmWave massive MIMO systems with orthogonal frequency division multiplexing (OFDM) scheme, offering the same advantages. Simulation and computational comparisons demonstrate that the proposed methods outperform state-of-the-art algorithms in terms of achievable spectral efficiency while reducing the computational complexity significantly. 

***************************************************************************************************
