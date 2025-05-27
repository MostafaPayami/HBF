# Hybrid Beamforming Methods for mmWave Massive MIMO Systems 
The MATLAB and Python codes used for the experiments and results presented in the following paper [1]. 

[1] M. Payami, and S. D. Blostein, “Hybrid Beamforming for mmWave Massive MIMO Systems Based on Analytical Phase Optimization and Self-Supervised Learning”, submitted to IEEE Transactions on Signal Processing, 2025.

The MATLAB codes implement the IFPAD method, and the Python codes are TensorFlow/Keras models for the AE-HBFnet DNN.

The following software and libraries are required:
1) MATLAB: Version R2023b or later
2) TensorFlow: Version 2.19.0 or later
3) Keras: Version 3.8.0 or later

***************************************************************************************************

Paper's Abstract: 

Hybrid analog and digital beamforming is a key technique that can address the high cost and power consumption of fully digital precoding, and thereby facilitates practical implementation of millimeter wave (mmWave) massive multiple-input multiple-output (MIMO) wireless systems. A major challenge is however that the optimal design of hybrid beamforming architectures involves solving a high-dimensional and nonconvex optimization problem, which is recast as one of constrained matrix factorization for point-to-point mmWave massive MIMO communications. To accomplish hybrid precoder design, the nonconvex optimization problem is transformed into an unconstrained system of nonlinear equations in terms of the analog beamformer phases to acquire local minima of the constrained optimization problem. Then, the Newton method with diagonal Jacobian is adopted to solve this system of equations, based on which an inverse-free phase-optimized analog/digital algorithm termed IFPAD is developed that obtains jointly optimal hybrid beamformers. Consisting of only elementary function evaluations and matrix multiplication, the proposed IFPAD features simple implementation and low complexity and avoids iterative matrix inversions prevalent in conventional methods. Additionally, a model-driven neural network, AE-HBFnet, is designed based on a self-supervised learning paradigm to perform the hybrid beamforming more efficiently. The AE-HBFnet is a lightweight generative model whose architecture is inspired by autoencoders. The proposed schemes are next extended to broadband mmWave massive MIMO systems with orthogonal frequency division multiplexing (OFDM). Simulation and computational comparisons demonstrate improved spectral efficiency and significantly reduced computational complexity compared to existing algorithms.

***************************************************************************************************
