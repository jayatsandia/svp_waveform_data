#!/usr/bin/env python

# Old code that likely doesn't work

from __future__ import division
from numpy import logical_and, average, diff
from matplotlib.mlab import find
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import math
import pandas as pd
import os, os.path
import datetime
import time
from scipy.signal import argrelextrema
from scipy.signal import kaiserord, lfilter, firwin, freqz
from numpy import argmax, sqrt, mean, diff, log
from scipy.signal import blackmanharris, fftconvolve
from numpy.fft import rfft, irfft
from sklearn import svm
from sklearn.cluster import KMeans

def tp(v):
    v = np.transpose(np.array([v]))
    return v

def file_extract(file,vname,iname,tname):

    d = pd.read_csv(file, sep='\t',header=0)

    time = np.array([d['Time'].as_matrix()])
    Vac = np.array([d[vname].as_matrix()])
    Iac = np.array([d[iname].as_matrix()])
    A = np.array([d[tname].as_matrix()])

    max_Vac, min_Vac = np.max(Vac), np.min(Vac)
    max_Iac, min_Iac = np.max(Iac), np.min(Iac)

    return time,Vac,Iac,max_Vac,min_Vac,max_Iac,min_Iac,A

def filter(sig,fs,filt):

    # Kaiser Filter
    if filt == 0:
        nyq_rate = fs / 2.0
        # The desired width of the transition from pass to stop, relative to the Nyquist rate.
        width = 120.0/nyq_rate

        # The desired attenuation in the stop band, in dB.
        ripple_db = 65	#16#60.0

        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = kaiserord(ripple_db, width)

        # The cutoff frequency of the filter.
        cutoff_hz = 55

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

        print(taps.shape)

        # Use lfilter to filter x with the FIR filter.
        #sig_ff = signal.filtfilt(taps, 0.84, sig) #lfilter(taps, 2.0, x)
        #sig_ff = signal.filtfilt(taps, 0.835, sig)

        #sig_ff = signal.lfilter(taps,0.2,sig)
        sig_ff = signal.filtfilt(taps, 0.835, sig)

    # Butterworth Filter
    if filt == 1:
        wn = (2.*math.pi*60.)/fs
        b, a = signal.butter(4, wn, analog=False)
        sig_ff = signal.filtfilt(b, a, sig)

    return sig_ff

def zero_cross(time,sig_actual,sig_filter):
    ''' Freq. from Zero Crossing Estimate '''
    t,sig = np.transpose(time),np.transpose(sig_filter)

    indices = find((sig[1:] >= 0) & (sig[:-1] < 0))
    tf = t[indices]
    f = 1/np.diff(tf, axis=0)
    tz = np.zeros((len(tf),1))

    return tz,tf,f

def amplitude(ftime,sig_actual,sig_filter):

    sig,t = np.transpose(sig_filter),ftime.T

    maxloc,minloc = argrelextrema(sig, np.greater),argrelextrema(sig, np.less)
    xmax,xmin = sig[maxloc],sig[minloc]
    maxloc,minloc = np.transpose(np.array([maxloc[0]])),np.transpose(np.array([minloc[0]]))
    tmax,tmin = t[maxloc],t[minloc]

    return xmax,xmin,maxloc,minloc,tmax,tmin


def label_average(labels,label_running, rm):
    for i in range(len(label_running)):
        if labels[rm+1] == 0:
            if i < rm:
                label_running[i] = 0
            if label_running[i] < 0.5:
                label_running[i] = 0
            else:
                label_running[i] = 1
        else:
            if i < rm:
                label_running[i] = 1
            if label_running[i] >= 0.5:
                label_running[i] = 0
            else:
                label_running[i] = 1

    return label_running

def trip_times(ftime,tvz,dvz,thresh,type):

    time_trip, time_return = 'NAN','NAN'

    if type == 'freq':
        thresh = 60 - 60 * thresh
    if type == 'volt':
        thresh = np.mean(dvz) - np.mean(dvz) * thresh

    dvz = dvz[10:]
    tvz = tvz[10:]

    tvz = np.transpose(np.array([tvz.ravel()]))
    mat = pd.rolling_mean(tvz,10)
    ma = pd.rolling_mean(dvz,10)

    if type == 'freq':
        for i in range(len(dvz)):
            d = np.mean(dvz[i+6:i+10]) - np.mean(dvz[i:i+5])
            if abs(d) > thresh:
                i_middle = i + 6
                time_trip = float(tvz[i+6])
                break

        for j in range(i_middle + 10,len(dvz)):
            d = np.mean(dvz[j+6:j+10]) - np.mean(dvz[j:j+5])
            if abs(d) > thresh:
                j_stop = j + 6
                time_return = float(tvz[j+6])
                break

    if type == 'volt':
        rm = 12 # 8
        mat = pd.rolling_mean(tvz,rm)
        ma = pd.rolling_mean(dvz,rm)

        km = KMeans(n_clusters=2)
        km.fit(ma[rm:].reshape(-1,1))
        labels = km.labels_

        label_running = pd.rolling_mean(labels,rm)
        label_running = label_average(labels,label_running, rm)


        index1 = np.where(label_running[:-1] != label_running[1:])[0]

        if len(index1) > 1:
            index_start = int(index1[-2:-1]) + 1
            index_end = int(index1[-1:]) + 1
            time_trip = tvz[index_start]
            time_return = tvz[index_end]
        if len(index1) == 1:
            index_start = int(index1[0]) + 1
            index_end = 0
            time_trip = tvz[index_start]
            time_return = 0

    return time_trip, time_return, ma, mat


def ride_through_time(ftime,tmax,tmin,dmax,dmin,imax,thresh):

    rm = 4
    mat = pd.rolling_mean(tmax.ravel(),rm)
    dmax = pd.rolling_mean(dmax,rm)

    mat = mat[rm-1:]
    dmax = dmax[rm-1:]

    km = KMeans(n_clusters=2)
    km.fit(dmax.reshape(-1,1))
    labels = km.labels_

    label_running = pd.rolling_mean(labels,rm)
    print label_running
    label_running = label_average(labels,label_running, rm)

    index1 = np.where(label_running[:-1] != label_running[1:])[0]
    index = int(index1[-1:]) + 1

    if len(index1) > 12 or min(dmax) > 5:
        time_trip = tmax[-1:]
    else:
        time_trip = tmax[index]

    #plt.scatter(mat,dmax)
    #plt.show()

    print label_running
    print index1

    return time_trip


''' 
Frequency Trip, Voltage Trip, and Ride Through
'''
def freq_trip(file,fs,filter_type,name,thresh):
    vname,iname,tname = name[0],name[1],name[2]

    # EXTRACT DATA ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ftime,Vac,Iac,max_Vac,min_Vac,max_Iac,min_Iac,A = file_extract(file,vname,iname,tname)

    # FILTER DATA +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    sig_Iac,sig_Vac = filter(Iac,fs,filter_type),filter(Vac,fs,filter_type)

    # PERIOD LENGTHS (zero crossing) ++++++++++++++++++++++++++++++++++++++++++
    tz,tf,freq = zero_cross(ftime,Vac,sig_Vac)		#Vtz-xlocation,Vz-zero value,Vdz-distance between zero values
    #Itz,Iz,Vdz = zero_cross(ftime,sig_Vac)

    #Vz = freq_from_fft(ftime,sig_Vac)

    # EVALUATE AMPLITUDE SAGS +++++++++++++++++++++++++++++++++++++++++++++++++
    Vmax,Vmin,Vmaxloc,Vminloc,Vtmax,Vtmin = amplitude(ftime,Vac,sig_Vac)
    #Imax,Imin,Imaxloc,Iminloc,Itmax,Itmin = amplitude(ftime,Iac,sig_Iac)

    # CHECK FREQ TRIP +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    freq_trip, freq_return, ma, mat = trip_times(ftime,tf,freq,thresh[0],'freq')

    return freq_trip, freq_return, ma, mat, sig_Vac, Vac, freq, tz, tf, ftime #,Vtz,1/Vdz

def volt_trip(file,fs,filter_type,name,thresh):
    vname,iname,tname = name[0],name[1],name[2]

    # EXTRACT DATA ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ftime,Vac,Iac,max_Vac,min_Vac,max_Iac,min_Iac,A = file_extract(file,vname,iname,tname)

    # FILTER DATA +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    sig_Iac,sig_Vac = filter(Iac,fs,filter_type),filter(Vac,fs,filter_type)

    # PERIOD LENGTHS (zero crossing) ++++++++++++++++++++++++++++++++++++++++++
    tz,tf,freq = zero_cross(ftime,Vac,sig_Vac)		#Vtz-xlocation,Vz-zero value,Vdz-distance between zero values
    #Itz,Iz,Vdz = zero_cross(ftime,Iac,sig_Vac)

    # EVALUATE AMPLITUDE SAGS +++++++++++++++++++++++++++++++++++++++++++++++++
    Vmax,Vmin,Vmaxloc,Vminloc,Vtmax,Vtmin = amplitude(ftime,Vac,sig_Vac)
    Imax,Imin,Imaxloc,Iminloc,Itmax,Itmin = amplitude(ftime,Iac,sig_Iac)

    #fit_function(ftime,Vtmax,Vmax,thresh[1],'volt')

    volt_trip, volt_return, ma, mat = trip_times(ftime,Vtmax,Vmax,thresh[1],'volt')

    return volt_trip, volt_return, ma, mat, sig_Vac, Vac, freq, tz, tf, ftime, Vtmax, Vtmin, Vmax, Vmin

def ride_trip(file,fs,filter_type,name,thresh):
    vname,iname,tname = name[0],name[1],name[2]

    # EXTRACT DATA ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ftime,Vac,Iac,max_Vac,min_Vac,max_Iac,min_Iac,A = file_extract(file,vname,iname,tname)

    # FILTER DATA +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Kaiser Filter ------------------------------------------
    sig_Iac,sig_Vac = filter(Iac,fs,filter_type),filter(Vac,fs,filter_type)
    # PERIOD LENGTHS (zero crossing) ++++++++++++++++++++++++++++++++++++++++++
    Vtz,Vz,Vdz = zero_cross(ftime,Vac,sig_Vac)		#Vtz-xlocation,Vz-zero value,Vdz-distance between zero values
    Itz,Iz,Vdz = zero_cross(ftime,Iac,sig_Vac)

    # EVALUATE AMPLITUDE SAGS +++++++++++++++++++++++++++++++++++++++++++++++++
    Vmax,Vmin,Vmaxloc,Vminloc,Vtmax,Vtmin = amplitude(ftime,Vac,sig_Vac)
    Imax,Imin,Imaxloc,Iminloc,Itmax,Itmin = amplitude(ftime,Iac,sig_Iac)

    # CHECK RIDE THROUGH ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ride_through_end = ride_through_time(ftime,Itmax,Itmin,Imax,Imin,max_Iac,thresh[2])

    return ride_through_end,float(ftime[:,-1]),sig_Iac,Iac,Imax,Imin,Itmax,Itmin


