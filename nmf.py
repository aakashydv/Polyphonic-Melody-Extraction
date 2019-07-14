import librosa
import numpy as np
from sklearn.cluster import KMeans

y,sr=librosa.load('C:\\Users\\Rohin Gupta\\Desktop\\Audio\\Bach10_v1.1\\01-AchGottundHerr\\01-AchGottundHerr.wav',offset=2.0,sr=441000,duration=5.0)
inp = np.abs(librosa.stft(y))
inp=np.transpose(inp)
kmeans=KMeans(n_clusters=10)
kmeans.fit(inp)
centroid=kmeans.cluster_centers_
label=kmeans.labels_
centroid=centroid.transpose()
inp=inp.transpose()
print(centroid.shape[0],centroid.shape[1])
H=np.zeros((inp.shape[1],10))
count=np.zeros(10)
for i in range(inp.shape[1]):
	H[i][label[i]]=1
	count[label[i]]=count[label[i]]+1
D=np.zeros((10,10))
for i in range(D.shape[0]):
	D[i][i]=(1/count[i])
E=np.ones((inp.shape[1],10))
G=np.zeros((inp.shape[1],10))
for i in range(len(H)):    
    for j in range(len(H[0])): 
        G[i][j]=H[i][j]+0.2*E[i][j]
W=np.dot(G,D)
constant=np.dot(inp.transpose(),inp)
for i in range(100):
	print(i)
	XW=np.dot(constant,W)
	WXW=np.dot(W.transpose(),XW)
	deno=np.dot(G,WXW)
	final=np.divide(XW,deno)
	final=np.sqrt(final)
	G=np.multiply(G,final)

	XW=np.dot(constant,G)
	GG=np.dot(G.transpose(),G)
	WXW=np.dot(constant,W)
	deno=np.dot(WXW,GG)
	final=np.divide(XW,deno)
	final=np.sqrt(final)
	W=np.multiply(W,final)

plt.figure(figsize=(12, 8))
librosa.display.specshow(G,sr=44100, y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
