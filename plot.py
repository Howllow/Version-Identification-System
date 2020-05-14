import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

y1, sr1 = librosa.load(r"./bob.mp3")
y2, sr2 = librosa.load(r"./jimi.mp3")



cqb = librosa.feature.chroma_cqt(y=y1, sr=sr1)
cqj = librosa.feature.chroma_cqt(y=y2, sr=sr2)

plt.figure(figsize=(15, 15))
plt.subplot(3, 1, 1)
librosa.display.specshow(cqb, y_axis='chroma', x_axis='time')
plt.title('bob dylan')
plt.colorbar()
plt.subplot(3, 1, 2)
librosa.display.specshow(cqj, y_axis='chroma', x_axis='time')
plt.title('jimi hendrix')
plt.colorbar()
plt.tight_layout()
plt.savefig("./res.png")
