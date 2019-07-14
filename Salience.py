import librosa
import numpy as np
import math
pi=3.14159265358
import  scipy
import statistics
from scipy import signal
y,sr=librosa.load('C:\\Users\\aakas\\PycharmProjects\\Music\\Bach10\\Bach10_v1.1\\01-AchGottundHerr\\01-AchGottundHerr.wav',offset=2.0,sr=44100,duration=5.0)

Yule_Deno=[1.00000000000000,
  -3.47845948550071,
   6.36317777566148,
  -8.54751527471874,
   9.47693607801280,
  -8.81498681370155,
   6.85401540936998,
  -4.39470996079559,
   2.19611684890774,
  -0.75104302451432,
   0.13149317958808]

Yule_Num=[0.05418656406430,
  -0.02911007808948,
  -0.00848709379851,
  -0.00851165645469,
  -0.00834990904936,
   0.02245293253339,
  -0.02596338512915,
   0.01624864962975,
  -0.00240879051584,
   0.00674613682247,
  -0.00187763777362]

Butter_Num=[ 0.98500175787242,
  -1.97000351574484,
   0.98500175787242
]

Butter_Deno = [1.00000000000000,
               -1.96977855582618,
               0.97022847566350
               ]

y = signal.lfilter(Yule_Num, Yule_Deno, y)
y = signal.lfilter(Butter_Num, Butter_Deno, y)

inp1 = (librosa.stft(y, n_fft=8192, hop_length=128, win_length=2048, window='hann'))
inp = np.abs(inp1)
inp_ph = np.angle(inp1, deg=True)
freqbins = len(inp)
timeframes = len(inp[0])
peaks = []
magnitude = []
for i in range(timeframes):
    peaks_frame = []
    magnitudes_frame = []
    for j in range(1, freqbins - 1):
        if inp[j][i] >= inp[j - 1][i] and inp[j][i] >= inp[j + 1][i]:  # finding local maximum peaks
            if (i != 0):
                arg = inp_ph[j][i] - inp[j][i - 1] - 2 * pi * 128 / 8192 * j
                arg = arg % 360;
                if (arg > 180):
                    arg = arg - 360
                arg = arg / 180 * pi;
                bin_offset = (8192) / 2 / pi / 128 * (arg)
            else:
                bin_offset = 0
                arg = 0
            peaks_frame.append((j + bin_offset) * sr / 8192)
            magnitudes_frame.append(inp[j][i])  # Corresponding magnitudes
    peaks.append(peaks_frame)
    magnitude.append(magnitudes_frame)
    # print(peaks)
    # print(magnitude)
print(1)
for i in range(len(peaks)):
    j = 0
    while j != (len(peaks[i])):
        if (peaks[i][j] > 2000) or peaks[i][j] < 0:
            del peaks[i][j]
            del magnitude[i][j]
        else:
            j = j + 1

print(2)

harmonics = 21
alpha = 0.8
sailence = np.zeros((timeframes, 600))
tsigma=0.9

"""

O(timeframes*600*harmoncis) complexity : Expensive 

for i in range(timeframes):
    sailence_frame=[]
    maxmag=np.amax(magnitude[i])
    #print(maxmag)
    bin_current=1
    while(bin_current<=600):      #  checking each bin for fundamental frequency
        ans=0.0
        for h in range (1,harmonics):                  #  loop over harmonic of considered bin
            for j in range (len(peaks[i])):
              if(peaks[i][j]<=0):
                continue
                if np.log10(maxmag/magnitude[i][j]) > 2:
                    e=0
                    continue
                else:
                    e=1

                freq=peaks[i][j]/h
                bin1=112.55*np.log2(freq/43)+1
                delta=np.abs((bin1-bin_current)/10)
                if delta <=1:
                    g=np.power(np.cos(pi/2*delta),2)*np.power(alpha,h-1)
                else:
                    g=0
                    continue
                ans=ans+g*np.power(magnitude[i][j],1)*e
        sailence_frame.append(ans)
        bin_current=bin_current+1
    sailence.append(sailence_frame)
print(2)
"""

for i in range(timeframes):
    maxmag = np.amax(magnitude[i])
    for j in range(len(peaks[i])):
        for h in range(harmonics):
            bin_peak = int(112.55 * np.log2(peaks[i][j] / 43) + 1)       # Computatinally Efficient
            for z in range(bin_peak - 10, bin_peak + 11):
                if (z < 0) or (z >= 600):
                    continue
                if (maxmag / magnitude[i][j] < 100):
                    sailence[i][z] += magnitude[i][j] * pow(np.cos(pi * (abs(z - bin_peak) / 10) / 2), 2) * pow(0.8,
                                                                                                                h - 1)
print(3)

sailence_peaks = []
sailence_bin = []
for i in range(timeframes):
    peak_temp = []
    bin_temp = []
    for j in range(1, 599):
        if (sailence[i][j] > sailence[i][j + 1] and sailence[i][j] > sailence[i][j - 1]):
            peak_temp.append(sailence[i][j])
            bin_temp.append(j)
    sailence_peaks.append(peak_temp)
    sailence_bin.append(bin_temp)

sailence_peaks_temp = []
sailence_bin_temp = []
sailence_mean = []

for i in range(timeframes):
    peak_temp = []
    bin_temp = []
    maxmag = np.amax(sailence_peaks[i])
    for j in range(len(sailence_peaks[i])):
        if (sailence_peaks[i][j] > 0.9 * maxmag):
            sailence_mean.append(sailence_peaks[i][j])
            peak_temp.append(sailence_peaks[i][j])
            bin_temp.append(sailence_bin[i][j])
    sailence_peaks_temp.append(peak_temp)
    sailence_bin_temp.append(bin_temp)

sailence_peaks = sailence_peaks_temp
sailence_bin = sailence_bin_temp
mean = np.sum(sailence_mean) / len(sailence_mean)
std = statistics.stdev(sailence_mean)

sailence1 = []
bin1 = []
for i in range(len(sailence_peaks)):
    temp_peak = []
    temp_bin = []
    for j in range(len(sailence_peaks[i])):
        if sailence_peaks[i][j] >= mean - tsigma * std:
            temp_peak.append(sailence_peaks[i][j])
            temp_bin.append(sailence_bin[i][j])
            del sailence_peaks[i][j]
            del sailence_bin[i][j]
            j = j - 1
    sailence1.append(temp_peak)
    bin1.append(temp_bin)

sailence2 = sailence_peaks
bin2 = sailence_bin

contours_sailence = []
contours_bin = []
num = 0
for i in range(len(sailence1)):
    while (len(sailence1[i]) != 0):
        cs_temp = []
        cb_temp = []
        max_index = sailence1[i].index(max(sailence1[i]))
        cs_temp.append(sailence1[i][max_index])
        cs_bin.append(bin1[i][max_index])
        cur_bin = bin1[i][max_index]
        flag = True
        member = 1
        not_found = 0
        while (flag):
            for k in range(len(bin1[i + member])):
                if (abs(bin1[i + member][k] - cur_bin) < 8):
                    cs_temp.append(sailence1[i + member][k])
                    cb_temp.append(bin1[i + member][k])
                    del sailence1[i + member][k]
                    del bin1[i + member][k]
                    not_found = 0
                    member += 1
                if k == len(bin1[i + member]) - 1:
                    not_found = not_found + 1
                    if (not_found == 3):
                        flag = False
                        break
                    for z in range(len(bin2[i + member])):
                        if (abs(bin2[i + member][z] - cur_bin) < 8):
                            cs_temp.append(sailence2[i + member][z])
                            cb_temp.append(bin2[i + member][z])
                            member = member + 1
                            break
                        if (z == len(bin2[i + member]) - 1):
                            flag = False
                            break
        contours_sailence.append(cs_temp)
        contours_bin.append(cb_temp)
        del sailence1[i][max_index]
        del bin1[i][max_index]

#print(contours_bin)
#print(contours_sailence)
