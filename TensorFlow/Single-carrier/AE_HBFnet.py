##### AE-HBFnet: Hybrid Beamforming Neural Network for mmWave Massive MIMO Systems #####

import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

### Massive mmWave MIMO System Parameters ###

Nt  = 256    # Number of transmit antennas
Nr  = 64     # Number of receive antennas
Ns  = 6      # Number of data streams
NRF = 8      # Number of RF chains
NL  = 8      # Number of Learned IFPAD layers
Nf  = 16     # Number of feature maps

### Load Data ###

Fopt_train = sio.loadmat('Fopt_256x64_Ns6.mat', mat_dtype=True)['Fopt']   # Optimal Fully-Digital Precoder
Fopt_test  = sio.loadmat('Fopt_256x64_Ns6_Test.mat', mat_dtype=True)['Fopt_test']

# Wopt_train = sio.loadmat('Wopt_256x64_Ns6.mat',  mat_dtype=True)['Wopt']   # Optimal Fully-Digital Combiner
# Wopt_test  = sio.loadmat('Wopt_256x64_Ns6_Test.mat', mat_dtype=True)['Wopt_test']

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
    In = tf.expand_dims(tf.eye(Ns, dtype=A.dtype), axis=0)
    In = tf.tile(In, [tf.shape(A)[0], 1, 1])
    A2 = A - In
    return A2

def Subtract_One(A):
    Ones = tf.expand_dims(tf.ones([Ns, Ns], dtype=A.dtype), axis=0)
    Ones = tf.tile(Ones, [tf.shape(A)[0], 1, 1])
    A2 = Ones - A
    return A2

def Add_Identity(A):
    In = tf.eye(NRF, num_columns=Ns, batch_shape=[tf.shape(A)[0]], dtype=A.dtype)
    In = tf.expand_dims(In, axis=-1)
    A2 = A + In
    return A2

def Random_Phi0(x):
  phi0 = tf.random.uniform(shape=(tf.shape(x)[0], NRF, Ns), minval=-np.pi, maxval=np.pi)
  return phi0

def Z_arg(t, phi, rho, Fop):
    Frf = tf.complex(tf.math.cos(t) / tf.math.sqrt(1.0 * Nt), tf.math.sin(t) / tf.math.sqrt(1.0 * Nt))
    Fbb = tf.complex(rho * tf.math.cos(phi), rho * tf.math.sin(phi))
    I   = tf.eye(tf.shape(Fbb)[1], batch_shape=[tf.shape(Fbb)[0]], dtype=Fbb.dtype)
    Lbb = tf.linalg.matmul(Fbb, Fbb, adjoint_b=True) * (1 - I)
    Z   = tf.linalg.matmul(tf.complex(Fop[..., 0], Fop[..., 1]), Fbb, adjoint_b=True) - tf.linalg.matmul(Frf, Lbb)
    return tf.math.angle(Z)

def W_arg(t, phi, rho, Fop):
    Frf = tf.complex(tf.math.cos(t) / tf.math.sqrt(1.0 * Nt), tf.math.sin(t) / tf.math.sqrt(1.0 * Nt))
    Fbb = tf.complex(rho * tf.math.cos(phi), rho * tf.math.sin(phi))
    I   = tf.eye(tf.shape(Fbb)[1], batch_shape=[tf.shape(Fbb)[0]], dtype=Fbb.dtype)
    Lrf = tf.linalg.matmul(Frf, Frf, adjoint_a=True) * (1 - I)
    W   = tf.linalg.matmul(Frf, tf.complex(Fop[..., 0], Fop[..., 1]), adjoint_a=True) - tf.linalg.matmul(Lrf, Fbb)
    return tf.math.angle(W)

def W_abs(t, phi, rho, Fop):
    Frf = tf.complex(tf.math.cos(t) / tf.math.sqrt(1.0 * Nt), tf.math.sin(t) / tf.math.sqrt(1.0 * Nt))
    Fbb = tf.complex(rho * tf.math.cos(phi), rho * tf.math.sin(phi))
    I   = tf.eye(tf.shape(Fbb)[1], batch_shape=[tf.shape(Fbb)[0]], dtype=Fbb.dtype)
    Lrf = tf.linalg.matmul(Frf, Frf, adjoint_a=True) * (1 - I)
    W   = tf.linalg.matmul(Frf, tf.complex(Fop[..., 0], Fop[..., 1]), adjoint_a=True) - tf.linalg.matmul(Lrf, Fbb)
    return tf.math.abs(W)

def Power_Normalization(Frf, Fbb):
    F     = tf.linalg.matmul(tf.complex(Frf[..., 0], Frf[..., 1]), tf.complex(Fbb[..., 0], Fbb[..., 1]))
    Fnorm = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.norm(F, ord='fro', axis=[-2, -1]), axis=-1), axis=-1), axis=-1)  # Frobenius norm
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

### The Design of Learned IFPAD Subnetwork ###

def Learned_IFPAD_Layer(T, Phi, Rho, Fopt):
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
    sPhi  = tf.keras.layers.Reshape((NRF, Ns, 1))(sPhi)
    sPhi  = tf.keras.layers.Dense(Nf, use_bias=False)(sPhi)
    sPhi  = tf.keras.activations.swish(sPhi)
    sPhi  = tf.keras.layers.Dense(1,  use_bias=False)(sPhi)
    sPhi  = tf.keras.layers.Reshape((NRF, Ns))(sPhi)
    sPhi  = tf.keras.activations.tanh(sPhi)
    Phi   = tf.keras.layers.Add()([Phi, sPhi])

    dRho = tf.keras.layers.Subtract()([Beta, Rho])
    sRho = tf.keras.layers.Rescaling(scale=0.5, offset=0.0)(dRho)
    sRho = tf.keras.layers.Reshape((NRF, Ns, 1))(sRho)
    sRho = tf.keras.layers.Dense(Nf, use_bias=False)(sRho)
    sRho = tf.keras.activations.swish(sRho)
    sRho = tf.keras.layers.Dense(1, use_bias=False)(sRho)
    sRho = tf.keras.layers.Reshape((NRF, Ns))(sRho)
    sRho = tf.keras.layers.LeakyReLU(negative_slope=0.1)(sRho)
    Rho  = tf.keras.layers.Add()([Rho, sRho])
    return T, Phi, Rho

### AE-HBFnet ###

# Input
Fopt = tf.keras.Input(shape=(Nt, Ns, 2), name='Optimal_Precoder')   # Last dimension represents Real and Imaginary parts

# Encoder
Ts = tf.keras.layers.Lambda(lambda x: Angle(x), output_shape=(Nt, Ns))(Fopt)
Rs = tf.keras.layers.Lambda(lambda x: Abs(x), output_shape=(Nt, Ns))(Fopt)
RR = tf.keras.layers.Dot(axes=(1, 1))([Rs, Rs])
RR = tf.keras.layers.Lambda(lambda x: Subtract_One(x), output_shape=(Ns, Ns))(RR)
Fs = tf.keras.layers.Lambda(lambda x: Phase2Analog(x), output_shape=(Nt, Ns, 2))(Ts)
FF = tf.keras.layers.Lambda(lambda x: Matrix_Multiplication_AH_B(x[0], x[1]), output_shape=(Ns, Ns, 2))([Fs, Fs])
FF = tf.keras.layers.Lambda(lambda x: Abs(x), output_shape=(Ns, Ns))(FF)
FF = tf.keras.layers.Rescaling(scale=1.0/Nt, offset=0.0)(FF)
FF = tf.keras.layers.Lambda(lambda x: Subtract_Identity(x), output_shape=(Ns, Ns))(FF)
RR = tf.keras.layers.Reshape((Ns, Ns, 1))(RR)
FF = tf.keras.layers.Reshape((Ns, Ns, 1))(FF)
q  = tf.keras.layers.Concatenate(axis=-1)([RR, FF])
q  = tf.keras.layers.Flatten()(q)

# Latent Space (Determisnistic and Probabilistic)
v   = tf.keras.layers.Dense(Nf, use_bias=False)(q)
v   = tf.keras.activations.swish(v)
Phi = tf.keras.layers.Lambda(lambda x: Random_Phi0(x), output_shape=(NRF, Ns))(v)

# Decoder
Rho  = tf.keras.layers.Dense(NRF * Ns, use_bias=False)(v)
Rho  = tf.keras.layers.LeakyReLU(negative_slope=0.1)(Rho)
Rho  = tf.keras.layers.Reshape((NRF, Ns, 1))(Rho)
Rho  = tf.keras.layers.Lambda(lambda x: Add_Identity(x), output_shape=(NRF, Ns, 1))(Rho)
Fphi = tf.keras.layers.Lambda(lambda x: Phase2Analog(x), output_shape=(NRF, Ns, 2))(Phi)
FBB  = tf.keras.layers.Multiply()([Rho, Fphi])
Rho  = tf.keras.layers.Reshape((NRF, Ns))(Rho)
FopB = tf.keras.layers.Lambda(lambda x: Matrix_Multiplication_A_BH(x[0], x[1]), output_shape=(Nt, NRF, 2))([Fopt, FBB])
T    = tf.keras.layers.Lambda(lambda x: Angle(x), output_shape=(Nt, NRF))(FopB)
FRF  = tf.keras.layers.Lambda(lambda x: Phase2Analog(x), output_shape=(Nt, NRF, 2))(T)
FRF  = tf.keras.layers.Rescaling(scale=tf.math.sqrt(1.0/Nt), offset=0.0)(FRF)

# Decoder (Learned IFPAD)
for i in range(NL):
    T, Phi, Rho = Learned_IFPAD_Layer(T, Phi, Rho, Fopt)

# Final FRF and FBB
Fphi = tf.keras.layers.Lambda(lambda x: Phase2Analog(x), output_shape=(NRF, Ns, 2))(Phi)
Rho  = tf.keras.layers.Reshape((NRF, Ns, 1))(Rho)
FBB  = tf.keras.layers.Multiply(name='Digital_Precoder_unnormalized')([Rho, Fphi])
Rho  = tf.keras.layers.Reshape((NRF, Ns))(Rho)
FRF  = tf.keras.layers.Lambda(lambda x: Phase2Analog(x), output_shape=(Nt, NRF, 2))(T)
FRF  = tf.keras.layers.Rescaling(scale=tf.math.sqrt(1.0/Nt), offset=0.0, name='Analog_Precoder')(FRF)

# Reconstructed Input
Fopt_DNN = tf.keras.layers.Lambda(lambda x: Matrix_Multiplication(x[0], x[1]), output_shape=(Nt, Ns, 2))([FRF, FBB])

# AE-HBFnet Model
HBFnet = tf.keras.models.Model(Fopt, Fopt_DNN)
HBFnet.summary()

# Training
HBFnet.compile(optimizer=tf.keras.optimizers.Lamb(), loss = ['MSE'])
HBFnet_hist = HBFnet.fit(Fopt_train, Fopt_train, validation_split=0.10, batch_size=16, epochs=30, callbacks=[LR_scheduler])

FRF_DNN = HBFnet.get_layer('Analog_Precoder').output
FBB_DNN = HBFnet.get_layer('Digital_Precoder_unnormalized').output
FBB_DNN = tf.keras.layers.Lambda(lambda x: Power_Normalization(x[0], x[1]), name='Digital_Precoder', output_shape=(NRF, Ns, 2))([FRF_DNN, FBB_DNN])

AE_HBFnet = tf.keras.models.Model(inputs=HBFnet.input, outputs=[FRF_DNN, FBB_DNN])
AE_HBFnet.summary()

AE_HBFnet.save("AE_HBFnet.keras")

### Prediction ###

# AE_HBFnet = tf.keras.models.load_model("AE_HBFnet.keras", custom_objects=Custom_Functions, compile=False, safe_mode=False)
[FRF_dnn, FBB_dnn] = AE_HBFnet.predict(Fopt_test)
print(FRF_dnn.shape)
print(FBB_dnn.shape)
sio.savemat("AE_HBFnet_Hybrid_Precoders_Ns5_Test1000_Result.mat", {"FRF_dnn": FRF_dnn, "FBB_dnn": FBB_dnn})
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
