import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import GPyOpt 

# Function to plot spectrogram
def plot_spectrogram(signal, title, rate):
    fig, ax = plt.subplots(figsize=(20, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    img = librosa.display.specshow(D, sr=rate, x_axis='time', y_axis='log', ax=ax, cmap='seismic')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title=title)
    plt.show()

# Function to create sliding windows
def sliding_window(signal, frame_size, hop_length):
    slide = []
    for i in range(0, len(signal) - frame_size + 1, hop_length):
        a = signal[i:i + frame_size]
        slide.append(a)
    return np.array(slide)

# Directories for input and output
input_clean_audio_path = r"C:/Users/User/Downloads/audioClean2.wav"
input_noisy_audio_path = r"C:/Users/User/Downloads/audioNoise2.wav"
output_denoised_dir = r"C:/Users/User/Desktop/Shauryatrainee/practice"

# Ensure the paths are correct
assert os.path.exists(input_clean_audio_path), f"File not found: {input_clean_audio_path}"
assert os.path.exists(input_noisy_audio_path), f"File not found: {input_noisy_audio_path}"

# Load clean and noisy audio files
clean_data, rate = librosa.load(input_clean_audio_path, mono=True)
noisy_data, _ = librosa.load(input_noisy_audio_path, mono=True)

# Ensure the audio files are the same length
max_length = max(len(clean_data), len(noisy_data))
clean_data = np.pad(clean_data, (0, max_length - len(clean_data)), 'constant')
noisy_data = np.pad(noisy_data, (0, max_length - len(noisy_data)), 'constant')

# Normalize the data
clean_data = clean_data.astype('float32') / np.max(np.abs(clean_data))
noisy_data = noisy_data.astype('float32') / np.max(np.abs(noisy_data))

# Define the window size and hop length based on the sampling rate
window = int(rate * 0.04)  # 20 ms window
hop_length = int(window // 4)  # 25% overlap

# Detrend and create sliding windows for clean data
D3_env2 = sliding_window(clean_data, window, hop_length)

# Define the U-Net architecture for the VAE
l2_reg_default = 0.001  # Default value for L2 regularization

# Encoder model
def build_encoder(window, latent_dim):
    inputs = keras.Input(shape=(window, 1))
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_default))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)

    x = layers.GRU(64, return_sequences=True)(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_default))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)

    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_default))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)

    # Flatten for latent vector calculation
    x = layers.Flatten()(x)

    # Variational layers
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    
    return keras.Model(inputs, [z_mean, z_log_var])

# Decoder model
def build_decoder(latent_dim, window):
    decoder_input = layers.Input(shape=(latent_dim,))
    x = layers.Dense((window // 8) * 128, activation='relu')(decoder_input)
    x = layers.Reshape((window // 8, 128))(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1DTranspose(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1DTranspose(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.GRU(64, return_sequences=True)(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1DTranspose(32, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Conv1D(1, kernel_size=3, activation='sigmoid', padding='same')(x)
    
    return keras.Model(decoder_input, outputs)

# VAE model with custom layers and loss
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = Sampling()([z_mean, z_log_var])
        reconstructed = self.decoder(z)
        return reconstructed

# Sampling layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Define the VAE loss function
def vae_loss(encoder):
    def loss(inputs, outputs):
        min_len = min(inputs.shape[1], outputs.shape[1])
        reconstruction_loss = tf.reduce_mean(tf.square(outputs[:, :min_len, :] - inputs[:, :min_len, :]))
        
        z_mean, z_log_var = encoder(inputs)
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        
        return reconstruction_loss + kl_loss
    return loss

# Compile the VAE model with Adam optimizer
def compile_vae(encoder, decoder, learning_rate=1e-3):
    vae = VAE(encoder=encoder, decoder=decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=vae_loss(encoder))
    return vae

# Prepare input and output data for training
num_windows = len(D3_env2)
x_train = D3_env2.reshape(num_windows, window, 1)
x_train_noisy = sliding_window(noisy_data, window, hop_length)[:num_windows]  # Ensure shapes match

# Define the hyperparameter space
bounds = [
    {'name': 'latent_dim', 'type': 'discrete', 'domain': (16, 128)},
    {'name': 'window', 'type': 'discrete', 'domain': (100, 500)},
    {'name': 'hop_length', 'type': 'discrete', 'domain': (25, 100)},
    {'name': 'l2_reg', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
]

# Corrected objective function
def objective(hyperparameters):
    hyperparameters = hyperparameters[0]
    latent_dim = int(hyperparameters[0])
    window = int(hyperparameters[1])
    hop_length = int(hyperparameters[2])
    l2_reg = hyperparameters[3]
    learning_rate = hyperparameters[4]
    
    # Rebuild encoder and decoder with the current hyperparameters
    encoder = build_encoder(window, latent_dim)
    decoder = build_decoder(latent_dim, window)
    
    # Create a new instance of the VAE class
    vae = VAE(encoder, decoder)
    
    # Compile the VAE model
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=vae_loss(encoder))

    # Prepare input data with the current window and hop length
    x_train = sliding_window(clean_data, window, hop_length)
    x_train_noisy = sliding_window(noisy_data, window, hop_length)

    num_windows = len(x_train)

    x_train = x_train.reshape(num_windows, window, 1)
    x_train_noisy = x_train_noisy.reshape(num_windows, window, 1)

    # Train the VAE model and return validation loss
    history = vae.fit(x_train_noisy, x_train, epochs=5, batch_size=64, validation_split=0.1)
    val_loss = history.history['val_loss'][-1]

    return -val_loss  # Minimize validation loss

# Perform Bayesian optimization
bo = GPyOpt.methods.BayesianOptimization(f=objective, domain=bounds, acquisition_type='EI', num_cores=4, verbosity=True)

# Run the optimization
bo.run_optimization(max_iter=20)

# Print the optimized hyperparameters
print("Optimized hyperparameters:")
print(bo.x_opt)

# Use optimized hyperparameters to build and train the final model
best_hyperparameters = {
    'latent_dim': int(bo.x_opt[0]),
    'window': int(bo.x_opt[1]),
    'hop_length': int(bo.x_opt[2]),
    'l2_reg': float(bo.x_opt[3]),
    'learning_rate': float(bo.x_opt[4])
}

encoder = build_encoder(best_hyperparameters['window'], best_hyperparameters['latent_dim'])
decoder = build_decoder(best_hyperparameters['latent_dim'], best_hyperparameters['window'])
vae = compile_vae(encoder, decoder, learning_rate=best_hyperparameters['learning_rate'])

# Prepare input data with the optimized window and hop length
x_train = sliding_window(clean_data, best_hyperparameters['window'], best_hyperparameters['hop_length'])
x_train_noisy = sliding_window(noisy_data, best_hyperparameters['window'], best_hyperparameters['hop_length'])

num_windows = len(x_train)

x_train = x_train.reshape(num_windows, best_hyperparameters['window'], 1)
x_train_noisy = x_train_noisy.reshape(num_windows, best_hyperparameters['window'], 1)

# Train the VAE model with the optimized hyperparameters
history = vae.fit(x_train_noisy, x_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate the VAE model on the test set and perform post-processing as needed
decoded_windows = vae.predict(x_train_noisy)
decoded_data = np.hstack([decoded_windows[i].flatten() for i in range(num_windows)])

# Truncate the denoised data to match the original length
decoded_data = decoded_data[:len(clean_data)]

# Save the denoised audio
denoised_output_file = os.path.join(output_denoised_dir, 'audioNoisy_denoised_vae_gpbo.wav')
sf.write(denoised_output_file, decoded_data, rate)

print(f"Denoised file saved to: {denoised_output_file}")

# Calculate SNR
def calculate_snr(clean, denoised):
    noise = clean[:len(denoised)] - denoised
    signal_power = np.mean(clean[:len(denoised)] ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

snr_noisy = calculate_snr(clean_data, noisy_data)
snr_denoised = calculate_snr(clean_data, decoded_data)

print(f"SNR of Noisy Audio: {snr_noisy:.2f} dB")
print(f"SNR of Denoised Audio (VAE with GPBO): {snr_denoised:.2f} dB")

# Visualize the waveforms
plt.figure(figsize=(15, 10))
plt.subplot(3, 1, 1)
plt.plot(clean_data)
plt.title('Original Audio')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(range(len(noisy_data)), noisy_data, 'b.')
noisy_indices = np.where(noisy_data != clean_data)[0]
plt.plot(noisy_indices, noisy_data[noisy_indices], 'r.')
plt.title('Noisy Audio')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(decoded_data)
plt.title('Denoised Audio (VAE with GPBO)')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Play the original, noisy, and denoised audio
print('Original Audio')
display(Audio(clean_data, rate=rate))

print('Noisy Audio')
display(Audio(noisy_data, rate=rate))

print('Denoised Audio (VAE with GPBO)')
display(Audio(decoded_data, rate=rate))

# Plot spectrograms
plot_spectrogram(clean_data, 'Spectrogram of Original Audio', rate)
plot_spectrogram(noisy_data, 'Spectrogram of Noisy Audio', rate)
plot_spectrogram(decoded_data, 'Spectrogram of Denoised Audio (VAE with GPBO)', rate)
