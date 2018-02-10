#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 16:35:16 2017
RK(4-5)-IP method for solving the GNLSE
@author: kamper
"""

import numpy as np
from numpy.fft import fft, fftshift, ifft
# First simply RK4 method

def RK4_GNLSE(step, D, NonlinOperator, gamma, A):
    # Calculate interaction picture amplitude
    lin = np.exp(step/2*D)
    AI = ifft(fftshift(lin*fftshift(fft(A))))
    # Calculate RK4 coefficients
    k1 = step*ifft(fftshift(lin*fftshift(fft(NonlinOperator(A, gamma)*A))))
    k2 = step*NonlinOperator(AI + k1/2, gamma)*(AI + k1/2)
    k3 = step*NonlinOperator(AI + k2/2, gamma)*(AI + k2/2)
    A_temp = ifft(fftshift(lin*fftshift(fft(AI + k3))))
    k4 = step*NonlinOperator(A_temp, gamma)*A_temp
    # Calculate function 
    return ifft(fftshift(lin*fftshift(fft(AI + (k1 + 2*(k2 + k3) + k4)/6))))

