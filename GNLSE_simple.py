"""
Created on Mon May 22 15:43:21 2017
Fourier split step solver for the GNLSE 
@author: Mark Kamper Svendsen
"""

import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import math
from GNLSE_functions import linear_step, nonlinear_step, sech

# Parameters - export to external script?
pi = math.pi
c = 3E-4 # Speed of light in m/ps
beta2 = -10 # GVD in ps^2/m (Anormalous dispersion regime)
beta3 = 0
gamma = 0.1  # Nonlinearity in 1/(Wm)
# Pulse parameters
P0 = 100
T0 = 1
v  = 0
N2 = 1

spectogram = True



# Initialization of time- and frequency arrays
n = 14
N = 2**n
dt = 0.01
timelist = np.arange(-N/2,N/2-1)*dt
frequencies = timelist/dt*(2*pi/(N*dt))


# Step size and simulation domain length
dz = 1E-3;

# Characteristic length scales
L_NL = 1/(abs(gamma*P0))
L_DS = T0**2/(abs(beta2))
ratio = L_NL/L_DS
"""
# Testing if both nonlinear and dispersive effects are important
if ratio < 3:
    print("Nonlinear effects dominate!")
elif ratio > 0.2:
    print("Dispersive effects dominate!")
else:
    print("Both nonlinear and dispersive effects are important!")
"""
# Domain length
L_DS = T0**2/(abs(beta2))
L = L_DS
steps = round(L/dz)


# Pulse initialization
pulse = math.sqrt(N2*P0)*sech((timelist-2)/T0) + math.sqrt(N2*P0)*sech((timelist+2)/T0)
pulse_ini = np.copy(pulse)
pulse_F_ini = fftshift(fft(pulse))
pulse_F = fftshift(fft(pulse))

# Calculating initial pulse energy
energy_ini = dt*np.sum(np.abs(pulse_ini)**2)

# Initialization of data save arrays - predefine size for better speed!
energies = np.zeros(round(steps/10))
zs  = np.linspace(0, L, round(steps/10))
pulse_time_save = 1j*np.zeros((round(steps/10), np.size(pulse)))
pulse_freq_save = 1j*np.zeros((round(steps/10), np.size(pulse_F)))

# Solving via foor loop
for i in tqdm(range(steps)):  
    print(i) 
    time.sleep(0.03)
    # Perform half a linear step
    pulse_F = linear_step(pulse_F, beta2, beta3, frequencies, dz/2)
    
    # Calculate pulse in time domain
    pulse   = ifft(ifftshift(pulse_F))
    
    # Perform nonlinear step
    pulse   = nonlinear_step(pulse, gamma, dz)
    
    # Calculate pulse in frequency domain
    pulse_F = fftshift(fft(pulse))
    
    # Perform half a linear step
    pulse_F = linear_step(pulse_F, beta2, beta3, frequencies, dz/2)
    
    # Calculate pulse in time domain
    pulse = ifft(ifftshift(pulse_F))   
    
    # Calculating total energy
    if i % 10 == 0:
        energies[round(i/10)] = dt*np.sum(np.abs(pulse)**2) - energy_ini
        pulse_time_save[round(i/10):,] = pulse
        pulse_freq_save[round(i/10):,] = pulse_F
# %% 
## Visualization ## 

# Initial pulse
plt.figure(1)
plt.subplot(221)
plt.plot(timelist, np.abs(pulse_ini)**2)
plt.title('Initial temporal shape')
plt.xlabel('Time [ps]')
plt.ylabel('Amplitude')
plt.xlim(xmax = 5, xmin = -5)
plt.subplot(222)
plt.plot(frequencies, np.abs(pulse_F_ini)**2)
plt.title('Initial spectrum')
plt.xlabel('Frequency [THz]')
plt.ylabel('Amplitude')

# Final pulse
plt.subplot(223)
plt.plot(timelist, np.abs(pulse)**2)
plt.title('Final temporal shape')
plt.xlabel('Time [ps]')
plt.ylabel('Intensity [W]')
plt.xlim(xmax = 5, xmin = -5)
plt.subplot(224)
plt.plot(frequencies, np.abs(pulse_F)**2)
plt.title('Final spectrum')
plt.xlabel('Frequency [THz]')
plt.ylabel('Intensity [W]')

# Checking that total energy is conserved!
plt.figure(3)
plt.plot(zs, energies)
plt.title('Pulse energy variation vs propagation length')
plt.xlabel('Propagation length [m]')
plt.ylabel('Total pulse energy')


# Spectograms
if spectogram:
    xmax = 5
    xmin = -5
    # Pulse shape
    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Time [ps]')
    ax.set_ylabel('Propagation length [m]')
    ax.set_zlabel('Intensity [W]')
    ax.set_title('Pulse shape vs propagation length')    
    X, Y = np.meshgrid(timelist, zs)
    ax.plot_surface(X, Y, np.abs(pulse_time_save)**2)
    ax.set_autoscale_on(False)
    
    # Pulse spectrum
    fig = plt.figure(5)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Frequency [THz]')
    ax.set_ylabel('Propagation length [m]')
    ax.set_zlabel('Amplitude [A.U.]')
    ax.set_title('Spectrum vs propagation length')
    X, Y = np.meshgrid(frequencies, zs)
    ax.plot_surface(X, Y, np.abs(pulse_freq_save)**2)
    ax.set_autoscale_on(False)




