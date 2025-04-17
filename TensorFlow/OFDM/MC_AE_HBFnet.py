##### Multicarrier AE-HBFnet: Hybrid Beamforming Neural Network for Broadband mmWave Massive MIMO-OFDM Systems #####

import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

### Massive mmWave MIMO-OFDM System Parameters ###

Nt  = 256    # Number of transmit antennas
Nr  = 64     # Number of receive antennas
K   = 128    # Number of OFDM subcarriers
Ns  = 5      # Number of data streams
NRF = 7      # Number of RF chains
NL  = 12     # Number of Learned Multicarrier SNQ-NDJ layers
Nf  = 16     # Number of feature maps

### Load Data ###

Fopt_train = sio.loadmat('Fopt_OFDM_Nt256_Ns5.mat', mat_dtype=True)['Fopt']   # Optimal Fully-Digital Precoder

# Wopt_train = sio.loadmat('Fopt_OFDM_Nr64_Ns5.mat',  mat_dtype=True)['Wopt']   # Optimal Fully-Digital Combiner

Fopt_test = sio.loadmat('Fopt_OFDM_Nt256_Ns5_Test.mat', mat_dtype=True)['Fopt']

### TensorFlow Custom Functions ###

def Angle(A):
    T = tf.math.angle(tf.complex(A[..., 0], A[..., 1]))
    return T

def Abs(A):
    R = tf.math.abs(tf.complex(A[..., 0], A[..., 1]))
    return R

def Phase2Analog(T):
    F  = tf.stack([tf.math.cos(T), tf.math.sin(T)], axis=-1)
    return F

def Matrix_Multiplication(A, B):
    C0 = tf.linalg.matmul(tf.complex(A[..., 0], A[..., 1]), tf.complex(B[..., 0], B[..., 1]))
    C  = tf.stack([tf.math.real(C0), tf.math.imag(C0)], axis=-1)
    return C

def Matrix_Multiplication_AH_B(A, B):
    C0 = tf.linalg.matmul(tf.complex(A[..., 0], A[..., 1]), tf.complex(B[..., 0], B[..., 1]), adjoint_a=True)
    C  = tf.stack([tf.math.real(C0), tf.math.imag(C0)], axis=-1)
    return C

def Matrix_Multiplication_A_BH(A, B):
    C0 = tf.linalg.matmul(tf.complex(A[..., 0], A[..., 1]), tf.complex(B[..., 0], B[..., 1]), adjoint_b=True)
    C  = tf.stack([tf.math.real(C0), tf.math.imag(C0)], axis=-1)
    return C

def Subtract_Identity(A):
    In = tf.expand_dims(tf.expand_dims(tf.eye(Ns, dtype=A.dtype), axis=0), axis=1)
    In = tf.tile(In, [tf.shape(A)[0], K, 1, 1])
    A2 = A - In
    return A2

def Subtract_One(A):
    Ones = tf.expand_dims(tf.expand_dims(tf.ones([Ns, Ns], dtype=A.dtype), axis=0), axis=0)
    Ones = tf.tile(Ones, [tf.shape(A)[0], K, 1, 1])
    A2   = Ones - A
    return A2

def Add_Identity(A):
    In = tf.eye(NRF, num_columns=Ns, dtype=A.dtype)
    In = tf.expand_dims(tf.expand_dims(tf.expand_dims(In, axis=0), axis=1), axis=-1)
    In = tf.tile(In, [tf.shape(A)[0], K, 1, 1, 1])
    A2 = A + In
    return A2

def Random_Phi0(x):
  phi0 = tf.random.uniform(shape=(tf.shape(x)[0], K, NRF, Ns), minval=-np.pi, maxval=np.pi)
  return phi0

def Z_arg(t, phi, rho, Fop):
    Frf = tf.complex(tf.math.cos(t) / tf.math.sqrt(1.0 * Nt), tf.math.sin(t) / tf.math.sqrt(1.0 * Nt))
    Fbb = tf.complex(rho * tf.math.cos(phi), rho * tf.math.sin(phi))
    I   = tf.eye(tf.shape(Fbb)[2], batch_shape=[tf.shape(Fbb)[0]], dtype=Fbb.dtype)
    Lbb = tf.math.reduce_sum(tf.linalg.matmul(Fbb, Fbb, adjoint_b=True), axis=1) * (1 - I)
    Z   = tf.math.reduce_sum(tf.linalg.matmul(tf.complex(Fop[..., 0], Fop[..., 1]), Fbb, adjoint_b=True), axis=1) - tf.linalg.matmul(Frf, Lbb)
    return tf.math.angle(Z)

def W_arg(t, phi, rho, Fop):
    Frf = tf.complex(tf.math.cos(t) / tf.math.sqrt(1.0 * Nt), tf.math.sin(t) / tf.math.sqrt(1.0 * Nt))
    Fbb = tf.complex(rho * tf.math.cos(phi), rho * tf.math.sin(phi))
    I   = tf.eye(tf.shape(Frf)[2], batch_shape=[tf.shape(Frf)[0]], dtype=Frf.dtype)
    Lrf = tf.expand_dims(tf.linalg.matmul(Frf, Frf, adjoint_a=True) * (1 - I), axis=1)
    W   = tf.linalg.matmul(tf.expand_dims(Frf, axis=1), tf.complex(Fop[..., 0], Fop[..., 1]), adjoint_a=True) - tf.linalg.matmul(Lrf, Fbb)
    return tf.math.angle(W)

def W_abs(t, phi, rho, Fop):
    Frf = tf.complex(tf.math.cos(t) / tf.math.sqrt(1.0 * Nt), tf.math.sin(t) / tf.math.sqrt(1.0 * Nt))
    Fbb = tf.complex(rho * tf.math.cos(phi), rho * tf.math.sin(phi))
    I   = tf.eye(tf.shape(Frf)[2], batch_shape=[tf.shape(Frf)[0]], dtype=Frf.dtype)
    Lrf = tf.expand_dims(tf.linalg.matmul(Frf, Frf, adjoint_a=True) * (1 - I), axis=1)
    W   = tf.linalg.matmul(tf.expand_dims(Frf, axis=1), tf.complex(Fop[..., 0], Fop[..., 1]), adjoint_a=True) - tf.linalg.matmul(Lrf, Fbb)
    return tf.math.abs(W)

def Power_Normalization(Frf, Fbb):
    F     = tf.linalg.matmul(tf.expand_dims(tf.complex(Frf[..., 0], Frf[..., 1]), axis=1), tf.complex(Fbb[..., 0], Fbb[..., 1]))
    Fnorm = tf.expand_dims(tf.norm(F, ord='fro', axis=[-2, -1], keepdims=True), axis=-1)  # Frobenius norm
    Fbb   = tf.math.sqrt(1.0*Ns) / tf.math.real(Fnorm)  * Fbb
    return Fbb

Custom_Functions = {
    'Angle': Angle,
    'Abs': Abs,
    'Phase2Analog': Phase2Analog,
    'Matrix_Multiplication': Matrix_Multiplication,
    'Matrix_Multiplication_AH_B': Matrix_Multiplication_AH_B,
    'Matrix_Multiplication_A_BH': Matrix_Multiplication_A_BH,
    'Subtract_Identity': Subtract_Identity,
    'Subtract_One': Subtract_One,
    'Add_Identity': Add_Identity,
    'Random_Phi0': Random_Phi0,
    'Z_arg': Z_arg,
    'W_arg': W_arg,
    'W_abs': W_abs,
    'Power_Normalization': Power_Normalization}

def learning_rate_schedule(epoch):
    if epoch < 15:
        return 1e-3
    elif epoch < 20:
        return 5e-4
    elif epoch < 25:
        return 2e-4
    elif epoch < 28:
        return 1e-4
    else:
        return 1e-5
LR_scheduler = tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule)

### Layer Design of Learned SNQ-NDJ for MIMO-OFDM Systems ###

def Learned_SNQ_NDJ_Layer(T, Phi, Rho, Fopt):
    Psi = tf.keras.layers.Lambda(lambda x: Z_arg(x[0], x[1], x[2], x[3]), output_shape=(Nt, NRF))([T, Phi, Rho, Fopt])
    dT  = tf.keras.layers.Subtract()([Psi, T])
    sT  = tf.keras.ops.sin(dT)
    sT  = tf.keras.layers.Reshape((Nt, NRF, 1))(sT)
    sT  = tf.keras.layers.Dense(Nf, use_bias=False)(sT)
    sT  = tf.keras.activations.swish(sT)
    sT  = tf.keras.layers.Dense(1,  use_bias=False)(sT)
    sT  = tf.keras.activations.tanh(sT)
    sT  = tf.keras.layers.Reshape((Nt, NRF))(sT)
    T   = tf.keras.layers.Add()([T, sT])

    Beta  = tf.keras.layers.Lambda(lambda x: W_abs(x[0], x[1], x[2], x[3]), output_shape=(NRF, Ns))([T, Phi, Rho, Fopt])
    Omega = tf.keras.layers.Lambda(lambda x: W_arg(x[0], x[1], x[2], x[3]), output_shape=(NRF, Ns))([T, Phi, Rho, Fopt])
    dPhi  = tf.keras.layers.Subtract()([Omega, Phi])
    sPhi  = tf.keras.ops.sin(dPhi)
    sPhi  = tf.keras.layers.Reshape((K, NRF, Ns, 1))(sPhi)
    sPhi  = tf.keras.layers.Dense(Nf, use_bias=False)(sPhi)
    sPhi  = tf.keras.activations.swish(sPhi)
    sPhi  = tf.keras.layers.Dense(1,  use_bias=False)(sPhi)
    sPhi  = tf.keras.layers.Reshape((K, NRF, Ns))(sPhi)
    sPhi  = tf.keras.activations.tanh(sPhi)
    Phi   = tf.keras.layers.Add()([Phi, sPhi])

    dRho = tf.keras.layers.Subtract()([Beta, Rho])
    sRho = tf.keras.layers.Rescaling(scale=0.5, offset=0.0)(dRho)
    sRho = tf.keras.layers.Reshape((K, NRF, Ns, 1))(sRho)
    sRho = tf.keras.layers.Dense(Nf, use_bias=False)(sRho)
    sRho = tf.keras.activations.swish(sRho)
    sRho = tf.keras.layers.Dense(1, use_bias=False)(sRho)
    sRho = tf.keras.layers.Reshape((K, NRF, Ns))(sRho)
    sRho = tf.keras.layers.LeakyReLU(negative_slope=0.1)(sRho)
    Rho  = tf.keras.layers.Add()([Rho, sRho])
    return T, Phi, Rho

### Multicarrier AE-HBFnet ###

# Input Fopt[k] (k = 1, 2, ..., K)
Fopt = tf.keras.Input(shape=(K, Nt, Ns, 2), name='Optimal_Precoder')   # Last dimension represents Real and Imaginary parts

# Encoder
Ts = tf.keras.layers.Lambda(lambda x: Angle(x), output_shape=(K, Nt, Ns))(Fopt)
Rs = tf.keras.layers.Lambda(lambda x: Abs(x), output_shape=(K, Nt, Ns))(Fopt)
Rs2 = tf.keras.layers.Permute((1, 3, 2))(Rs)
RR = tf.keras.ops.matmul(Rs2, Rs)
RR = tf.keras.layers.Lambda(lambda x: Subtract_One(x), output_shape=(K, Ns, Ns))(RR)
Fs = tf.keras.layers.Lambda(lambda x: Phase2Analog(x), output_shape=(Nt, Ns, 2))(Ts)
FF = tf.keras.layers.Lambda(lambda x: Matrix_Multiplication_AH_B(x[0], x[1]), output_shape=(K, Ns, Ns, 2))([Fs, Fs])
FF = tf.keras.layers.Lambda(lambda x: Abs(x), output_shape=(K, Ns, Ns))(FF)
FF = tf.keras.layers.Rescaling(scale=1.0/Nt, offset=0.0)(FF)
FF = tf.keras.layers.Lambda(lambda x: Subtract_Identity(x), output_shape=(K, Ns, Ns))(FF)
RR = tf.keras.layers.Reshape((K, Ns, Ns, 1))(RR)
FF = tf.keras.layers.Reshape((K, Ns, Ns, 1))(FF)
Q  = tf.keras.layers.Concatenate(axis=-1)([RR, FF])
Q  = tf.keras.layers.Reshape((K, Ns * Ns * 2))(Q)

# Latent Space (Determisnistic and Probabilistic)
v   = tf.keras.layers.Dense(Nf, use_bias=False)(Q)
v   = tf.keras.activations.swish(v)
Phi = tf.keras.layers.Lambda(lambda x: Random_Phi0(x), output_shape=(K, NRF, Ns))(v)

# Decoder
Rho  = tf.keras.layers.Dense(NRF * Ns, use_bias=False)(v)
Rho  = tf.keras.layers.LeakyReLU(negative_slope=0.1)(Rho)
Rho  = tf.keras.layers.Reshape((K, NRF, Ns, 1))(Rho)
Rho  = tf.keras.layers.Lambda(lambda x: Add_Identity(x), output_shape=(K, NRF, Ns, 1))(Rho)
Fphi = tf.keras.layers.Lambda(lambda x: Phase2Analog(x), output_shape=(K, NRF, Ns, 2))(Phi)
FBB  = tf.keras.layers.Multiply()([Rho, Fphi])
Rho  = tf.keras.layers.Reshape((K, NRF, Ns))(Rho)
FopB = tf.keras.layers.Lambda(lambda x: Matrix_Multiplication_A_BH(x[0], x[1]), output_shape=(K, Nt, NRF, 2))([Fopt, FBB])
FopB = tf.keras.ops.sum(FopB, axis=1)
T    = tf.keras.layers.Lambda(lambda x: Angle(x), output_shape=(Nt, NRF))(FopB)
FRF  = tf.keras.layers.Lambda(lambda x: Phase2Analog(x), output_shape=(Nt, NRF, 2))(T)
FRF  = tf.keras.layers.Rescaling(scale=tf.math.sqrt(1.0/Nt), offset=0.0)(FRF)

# Decoder (Learned SNQ-NDJ)
for i in range(NL):
    T, Phi, Rho = Learned_SNQ_NDJ_Layer(T, Phi, Rho, Fopt)

# Final FRF and FBB[k] (k = 1, 2, ..., K)
Fphi = tf.keras.layers.Lambda(lambda x: Phase2Analog(x), output_shape=(K, NRF, Ns, 2))(Phi)
Rho  = tf.keras.layers.Reshape((K, NRF, Ns, 1))(Rho)
FBB  = tf.keras.layers.Multiply(name='Digital_Precoder_unnormalized')([Rho, Fphi])
Rho  = tf.keras.layers.Reshape((K, NRF, Ns))(Rho)
FRF  = tf.keras.layers.Lambda(lambda x: Phase2Analog(x), output_shape=(Nt, NRF, 2))(T)
FRF  = tf.keras.layers.Rescaling(scale=tf.math.sqrt(1.0/Nt), offset=0.0, name='Analog_Precoder')(FRF)
FRF2 = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(FRF)

# Reconstructed Input
Fopt_DNN = tf.keras.layers.Lambda(lambda x: Matrix_Multiplication(x[0], x[1]), output_shape=(K, Nt, Ns, 2))([FRF2, FBB])

# AE-HBFnet Model
HBFnet = tf.keras.models.Model(Fopt, Fopt_DNN)
HBFnet.summary()

# Training
HBFnet.compile(optimizer=tf.keras.optimizers.Lamb(), loss = ['MSE'])
HBFnet_hist = HBFnet.fit(Fopt_train, Fopt_train, validation_split=0.10, batch_size=16, epochs=30, callbacks=[LR_scheduler])

FRF_DNN  = HBFnet.get_layer('Analog_Precoder').output
# FRF_DNN2 = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(FRF_DNN)
FBB_DNN  = HBFnet.get_layer('Digital_Precoder_unnormalized').output
FBB_DNN  = tf.keras.layers.Lambda(lambda x: Power_Normalization(x[0], x[1]), name='Digital_Precoder', output_shape=(K, NRF, Ns, 2))([FRF_DNN, FBB_DNN])

AE_HBFnet = tf.keras.models.Model(inputs=HBFnet.input, outputs=[FRF_DNN, FBB_DNN])
AE_HBFnet.summary()

AE_HBFnet.save("MC_AE_HBFnet.keras")

### Prediction ###

# MC_AE_HBFnet = tf.keras.models.load_model("MC_AE_HBFnet.keras", custom_objects=Custom_Functions, compile=False, safe_mode=False)
[FRF_dnn, FBB_dnn] = AE_HBFnet.predict(Fopt_test)
print(FRF_dnn.shape)
print(FBB_dnn.shape)
sio.savemat("AE_HBFnet_Hybrid_Precoders_OFDM_Ns5_Test200_Result.mat", {"FRF_dnn": FRF_dnn, "FBB_dnn": FBB_dnn})

print('Done.')

### Plot Metrics ###

plt.plot(HBFnet_hist.history['loss'], label='Train Loss')
plt.plot(HBFnet_hist.history['val_loss'], label='Validation Loss')
plt.title('Training Result of HBFnet')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.grid(True, linestyle='--', alpha=0.75)
plt.tight_layout()
plt.show()
