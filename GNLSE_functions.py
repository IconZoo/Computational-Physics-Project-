# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:46:15 2017

@author: kamper
"""

import numpy as np
from numpy.fft import fft
from numpy.fft import fftshift
from numpy.fft import ifft
from numpy.fft import ifftshift
"""
def linear_step(pulseF, beta2, wList, dz):
    return np.exp(1j/2*beta2*(wList**2)*dz)*pulseF
"""
def linear_step(pulseF, beta2, beta3, wList, dz):
    return np.exp(1j*(beta2*(wList**2)/2 + beta3/6)*dz)*pulseF
    
def nonlinear_step(pulse, gamma, dz):
    return np.multiply(np.exp(1j*gamma*(np.abs(pulse)**2)*dz), pulse)


def nonlinear_step_raman(pulse_F, gamma, R, w0, frequencies):
    pulse = ifft(fftshift(pulse_F))
    # AF = 1j*gamma*(frequencies + w0)/w0*fftshift(fft(pulse*ifft(fftshift(R*fftshift(fft(np.abs(pulse)**2))))))
    AF = 1j*gamma*(w0+frequencies)/w0*fftshift(fft(pulse*ifft(fftshift(R*fftshift(fft(np.abs(pulse)**2))))))
    # Split real and imaginary part to meet ode solver requirement
    # real = 1j*gamma*(frequencies+ w0)/w0*fftshift(fft(pulse*ifft(fftshift(R*AF2))))
    return AF

def nonlinear_step_RKtester(pulse, gamma, R, w0, frequencies):
    return 1j*gamma*np.abs(pulse)**2*pulse

def nonlinear_step_RK(pulse, gamma):
    return 1j*gamma*np.abs(pulse)**2
    # return 1j*gamma*np.abs(pulse)**2*pulse
    

# Combination of IP-method and scipy RK4-method
def NonlinOperator_IN(A, gamma, D1, D2, wlist, step):
    N = nonlinear_step_RK(A, gamma)


"""
def nonlinear_step_raman_imag(pulse, gamma, R, w0, frequencies):
    AF2 = fftshift(fft(pulse**2)) 
    # Split real and imaginary part to meet ode solver requirement
    # real = np.real(1j*gamma*(frequencies+ w0)/w0*fftshift(fft(pulse*ifft(fftshift(R*AF2)))))
    imag = np.imag(1j*gamma*(frequencies+ w0)/w0*fftshift(fft(pulse*ifft(fftshift(R*AF2))))).astype('float64')
    return imag
"""


# Define the sech function
def sech(x):
    return 1/(np.cosh(x))