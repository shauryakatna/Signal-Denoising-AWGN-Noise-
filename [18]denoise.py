import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from IPython.display import Audio, display

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
input_clean_audio_path = r"D:/CSIRCSIO/[02]audio/audioClean2.wav"
input_noisy_audio_path = r"D:/CSIRCSIO/[02]audio/audioNoise2.wav"
output_denoised_dir = r"D:/CSIRCSIO/[02]audio"

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
latent_dim = 64


# Define the L2 regularizer
l2_reg = regularizers.l2(0.0001)  # Reduced regularization term


# Encoder model
inputs = keras.Input(shape=(window, 1))
x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2_reg)(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
x = layers.Dropout(0.3)(x)

x = layers.GRU(128, return_sequences=True, kernel_regularizer=l2_reg)(x)
x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2_reg)(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
x = layers.Dropout(0.3)(x)

x = layers.GRU(256, return_sequences=True, kernel_regularizer=l2_reg)(x)
x = layers.Conv1D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2_reg)(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
x = layers.Dropout(0.3)(x)

# Flatten for latent vector calculation
x = layers.Flatten()(x)

# Variational layers
z_mean = layers.Dense(latent_dim, name='z_mean', kernel_regularizer=l2_reg)(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var', kernel_regularizer=l2_reg)(x)

# Sampling layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Decoder model
decoder_input = layers.Input(shape=(latent_dim,))
x = layers.Dense((window // 8) * 256, activation='relu', kernel_regularizer=l2_reg)(decoder_input)
x = layers.Reshape((window // 8, 256))(x)
x = layers.UpSampling1D(size=2)(x)
x = layers.Conv1DTranspose(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2_reg)(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.GRU(256, return_sequences=True, kernel_regularizer=l2_reg)(x)
x = layers.UpSampling1D(size=2)(x)
x = layers.Conv1DTranspose(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2_reg)(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.GRU(128, return_sequences=True, kernel_regularizer=l2_reg)(x)
x = layers.UpSampling1D(size=2)(x)
x = layers.Conv1DTranspose(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2_reg)(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Conv1D(1, kernel_size=3, activation='sigmoid', padding='same', kernel_regularizer=l2_reg)(x)

# Instantiate the decoder model
decoder = keras.Model(decoder_input, outputs)

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

# Instantiate the VAE model
encoder_model = keras.Model(inputs, [z_mean, z_log_var])
vae = VAE(encoder=encoder_model, decoder=decoder)

# Define the VAE loss function
def vae_loss(inputs, outputs):
    min_len = min(inputs.shape[1], outputs.shape[1])
    reconstruction_loss = tf.reduce_mean(tf.square(outputs[:, :min_len, :] - inputs[:, :min_len, :]))
    
    z_mean, z_log_var = vae.encoder(inputs)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    
    return reconstruction_loss + kl_loss

# Compile the VAE model with Adam optimizer and learning rate schedule
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,  # Reduced learning rate
    decay_steps=10000,
    decay_rate=0.9
)

vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=vae_loss)

# Build the VAE model by calling it on a batch of data
vae.build(input_shape=(None, window, 1))

# Print the VAE model summary
vae.summary()


# Prepare input and output data for training
num_windows = len(D3_env2)
x_train = D3_env2.reshape(num_windows, window, 1)
x_train_noisy = sliding_window(noisy_data, window, hop_length)[:num_windows]  # Ensure shapes match

# Data augmentation function
def augment_data(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    augmented_data = np.clip(augmented_data, -1.0, 1.0)
    return augmented_data

# Augment the noisy data
x_train_noisy_augmented = np.array([augment_data(x) for x in x_train_noisy])

# Training with early stopping and model checkpoint callbacks
# Training with early stopping and model checkpoint callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(filepath='vae_denoising_best_model', save_best_only=True, save_format='tf')
]


# Train the VAE model
history = vae.fit(x_train_noisy_augmented, x_train, epochs=100, batch_size=64, validation_split=0.1, callbacks=callbacks)

# Plot the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='train', linewidth=2)
plt.plot(history.history['val_loss'], label='validate', linewidth=2)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Validation Loss Over Epochs', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


# Encode and decode the data
decoded_windows = vae.predict(x_train_noisy)

# Reconstruct the denoised audio
decoded_data = np.hstack([decoded_windows[i].flatten() for i in range(num_windows)])

# Truncate the denoised data to match the original length
decoded_data = decoded_data[:len(clean_data)]

# Save the denoised audio
denoised_output_file = os.path.join(output_denoised_dir, 'audioNoisy_denoised_vae_complex.wav')
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
print(f"SNR of Denoised Audio (VAE with Skip Connections): {snr_denoised:.2f} dB")

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
plt.title('Denoised Audio (VAE with Skip Connections)')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Play the original, noisy, and denoised audio
print('Original Audio')
display(Audio(clean_data, rate=rate))

print('Noisy Audio')
display(Audio(noisy_data, rate=rate))

print('Denoised Audio (VAE)')
display(Audio(decoded_data, rate=rate))

# Plot spectrograms
plot_spectrogram(clean_data, 'Spectrogram of Original Audio', rate)
plot_spectrogram(noisy_data, 'Spectrogram of Noisy Audio', rate)
plot_spectrogram(decoded_data, 'Spectrogram of Denoised Audio (VAE)', rate)
