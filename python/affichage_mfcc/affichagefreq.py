import numpy as np
import matplotlib.pyplot as plt

FRAME_SIZE = 256
SAMPLE_RATE = 8000

data = np.fromfile(r"C:\Users\sacha\AppData\Local\Programs\Microsoft VS Code\elec.bin", dtype=np.float32)

frame0 = data[0:FRAME_SIZE]

# Axe des fréquences
freqs = np.linspace(0, SAMPLE_RATE/2, FRAME_SIZE//2)

fig, ax1 = plt.subplots(1, 1)
ax1.plot(freqs, frame0[:FRAME_SIZE//2])  # on prend seulement la moitié (symétrie FFT)
ax1.set_title("Spectre FFT")
ax1.set_xlabel("Fréquence (Hz)")
ax1.set_ylabel("Amplitude")
plt.tight_layout()
plt.show()