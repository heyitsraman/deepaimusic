import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
file = "guitar.wav"

# WAVEFORM
signal, sr = librosa.load(file, sr=44100)  # sr * T -> 22050 * 30
librosa.display.waveplot(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# IMPLEMENTING FFT SPECTRUM
fft =np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0,sr,len(magnitude))

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency)/2)]
plt.plot(left_frequency, left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

# IMPLEMENTING STFT SPECTROGRAM

n_fft = 2048
hop_length = 512
stft =librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrogram =np.abs(stft)

log_spectrogram =librosa.amplitude_to_db(spectrogram)
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)

plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()

librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length)

plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()

# MFCC
MFFC =librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)


librosa.display.specshow(MFFC, sr=sr, hop_length=hop_length)

plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()
