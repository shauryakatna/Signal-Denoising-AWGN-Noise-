GRU-Enhanced Variational Autoencoder (GE-VAE) for Audio Denoising
Insufficient noise elimination from audio signals—particularly in fields like medical imaging and seismic evaluation—can conceal critical information and impede precise interpretation. This repository introduces Model D, a GRU-Enhanced Variational Autoencoder (GE-VAE), as a resilient framework for audio denoising in Additive White Gaussian Noise (AWGN) scenarios.

Overview
We assess and contrast the denoising effectiveness of four deep generative models:

Baseline VAE

CNN-VAE

GRU-VAE

GE-VAE (our proposed model)

The GE-VAE framework uniquely combines Convolutional Neural Networks (CNNs) with Gated Recurrent Units (GRUs) to efficiently encapsulate both spatial and temporal attributes of audio signals, generating a concise latent representation.

Loss Function
To achieve a balance between reconstruction accuracy and regularization, the model’s loss function assigns:

75% weight to Mean Squared Error (MSE)

25% weight to KL Divergence

Experimental Results
Models A–C: Provide moderate enhancements in output Signal-to-Noise Ratio (SNR), with Model C achieving up to 18.8dB.

Proposed GE-VAE Model: Consistently exceeds previous models, attaining an output SNR of 21.4dB, and offers a 6–8dB increase across input noise levels from –5dB to +10dB.

These results validate the significant advantage of the proposed GE-VAE framework in reducing AWGN and improving audio quality over traditional VAE-based methods.<img width="585" height="321" alt="Screenshot 2025-08-21 195333" src="https://github.com/user-attachments/assets/4dae7f60-8e5a-494a-84b4-9f233aa458b0" />
<img width="548" height="335" alt="Screenshot 2025-07-21 142111" src="https://github.com/user-attachments/assets/d7f9121c-29eb-4e9c-a225-1358c1362dd1" />
