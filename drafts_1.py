# import torch
# print(torch.backends.mps.is_available())  # Should print True
# print(torch.backends.mps.is_built())  # Should print True


import librosa

array, sampling_rate = librosa.load(librosa.ex("trumpet"))

import matplotlib.pyplot as plt
import librosa.display

plt.figure().set_figwidth(12)
librosa.display.waveshow(array, sr=sampling_rate)
plt.show()