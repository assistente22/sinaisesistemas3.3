import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.fft import rfft, rfftfreq, irfft

samplerate = 50*16 #50hz * 16segundos

elevct = pd.read_csv("./elevacao com tremor.csv")
elevpt = pd.read_csv("./elevacao pouco tremor.csv")

time_elev_ct = elevct["time"].values
x_elev_ct = elevct["x"].values

time_elev_pt = elevpt["time"].values
x_elev_pt = elevpt["x"].values

frequencias_pt = rfft(x_elev_pt)
frequencias_ct = rfft(x_elev_ct)

numero_capturas = min(len(time_elev_pt), len(time_elev_ct))

tempo = []

for i in range(numero_capturas):
    tempo.append(i)

plt.subplot(611)
plt.plot(tempo, x_elev_ct[0:numero_capturas])
plt.title("com tremor")

plt.subplot(612)
plt.plot(tempo, x_elev_pt[0:numero_capturas])
plt.title("pouco tremor")

numero_frequencias = rfftfreq(numero_capturas, 1/samplerate)

plt.subplot(613)
plt.plot(numero_frequencias, np.abs(frequencias_ct[:len(numero_frequencias)]))
plt.title("frequencias com tremor")

plt.subplot(614)
plt.plot(numero_frequencias, np.abs(frequencias_pt[:len(numero_frequencias)]))
plt.title('frequencias pouco tremor')

frequencias_pt[20:] = 0
resultado_filtrado = irfft(frequencias_pt)

plt.subplot(615)
plt.plot(numero_frequencias, np.abs(frequencias_pt[:len(numero_frequencias)]))
plt.title("frequencias com pouco tremor pós filtragem")


plt.subplot(616)
plt.plot(resultado_filtrado)
plt.title("movimento com tremores filtrados")

plt.show()