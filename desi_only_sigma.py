import numpy as np
from scipy.stats import uniform
from matplotlib import pyplot as plt
from scipy.linalg import sqrtm

import pickle

import time
import tqdm

# import os

# os.environ["OMP_NUM_THREADS"] = "1"

import multiprocess as mp
mp.set_start_method('spawn', force=True)

import os
# Ensure friendly behavior of OpenMP and multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
mp.set_start_method('spawn', force=True)

import pip

import multiprocessing
from multiprocessing import Pool

import copy

import getdist
from getdist import plots, MCSamples

from cosmoprimo.fiducial import Planck2018FullFlatLCDM
from cosmoprimo.fiducial import DESI
from cosmoprimo import *
desi = DESI(engine='camb')
planck = Planck2018FullFlatLCDM(engine='camb')
from cosmoprimo import constants

alpha_desi_z=np.loadtxt('bao_data/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt',usecols=[0,1]) #,z,alpha

alpha_desi_z_eff = np.array([0.295,0.51,0.706,0.93,1.317,1.491,2.33])

covariance_desi=np.loadtxt('bao_data/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt')

covariance_sdss_DR12_LRG=np.loadtxt('bao_data/sdss_DR12_LRG_BAO_DMDH_covtot.txt')[2:4,2:4]
alpha_sdss_z_DR12_LRG=np.loadtxt('bao_data/sdss_DR12_LRG_BAO_DMDH.dat',usecols=[0,1])[2:4] #z=0.51

covariance_sdss_DR16_LRG=np.loadtxt('bao_data/sdss_DR16_LRG_BAO_DMDH_covtot.txt')
alpha_sdss_z_DR16_LRG=np.loadtxt('bao_data/sdss_DR16_LRG_BAO_DMDH.dat',usecols=[0,1]) #z=0.698\approx 0.706


c_l_ms = 299792458
c_l_kms = 299792458/1000

def log_like(the, obs, cov):
  delta=the-obs
  invcov=np.linalg.inv(cov)
  sign, logabsdet = np.linalg.slogdet(cov)
  if sign <= 0:
    return -np.inf
  # print(delta)
  # print(invcov)
  # print(logabsdet)
  # print(-0.5*np.dot(delta, np.dot(invcov, delta))-0.5*logabsdet)
  return -0.5*np.dot(delta, np.dot(invcov, delta))-0.5*logabsdet
  # print(the,obs, cov)
  #return -0.5*(delta**2/cov + np.log(cov))

# !pip install emcee corner
import emcee
import corner

def cosmology(w0,wa,Omega_m,omega_b,h,z):
  cosmo=Cosmology(w0_fld=w0,wa_fld=wa,Omega_m=Omega_m,omega_b=omega_b, h=h)
  cosmo.set_engine('camb')
  a_p=cosmo.comoving_angular_distance(z)/cosmo.rs_drag #D_m/r_d
  a_v=c_l_kms/(cosmo.efunc(z)*100)/cosmo.rs_drag #D_H/r_d
  return a_p,a_v


def cosmology_iso(w0,wa,Omega_m,omega_b,h,z):
  cosmo=Cosmology(w0_fld=w0,wa_fld=wa,Omega_m=Omega_m,omega_b=omega_b, h=h)
  cosmo.set_engine('camb')
  d_m=cosmo.comoving_angular_distance(z) #D_m
  d_h=c_l_kms/(cosmo.efunc(z)*100) #D_H
  D_iso = (d_h*d_m**2*z)**(1/3)
  a_iso=D_iso/cosmo.rs_drag
  return a_iso

def loglike_cosmo_desi(theory, obs, cov, z): #cosmological parameters, only desi
  w0, wa, h, Omega_m, omega_b, sigma_s = theory
  if Omega_m <omega_b/h**2 or Omega_m>0.99:
    return -np.inf
  if omega_b <0.005 or omega_b>0.1:
    return -np.inf
  if h >1 or h<0.2:
    return -np.inf
  if w0<-3 or w0>1:
    return -np.inf
  if wa<-3 or wa>2:
    return -np.inf
  if sigma_s < 0 or sigma_s > 10:
    return -np.inf
  if w0+wa>0:
    return -np.inf
  alpha_p,alpha_v=cosmology(w0,wa,Omega_m,omega_b,h,[z[1],z[2],z[3],z[4],z[6]])
  alpha_iso=cosmology_iso(w0,wa,Omega_m,omega_b,h,[z[0],z[5]])
  alpha_combine=np.array([alpha_iso[0],alpha_p[0],alpha_v[0],alpha_p[1],alpha_v[1],alpha_p[2],alpha_v[2],alpha_p[3],alpha_v[3],alpha_iso[1],alpha_p[4],alpha_v[4]])
  return log_like(alpha_combine, obs, cov*(1 + sigma_s))-((omega_b-0.02218)**2/0.00055**2)*0.5


# Initialize in a ball around some nominal value, sigmas=0
with Pool() as pool:
  nwalkers=20
  ndim=6
  sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike_cosmo_desi,pool=pool,args=[(alpha_desi_z.T[1]),covariance_desi,alpha_desi_z_eff])
  p0 = emcee.utils.sample_ball(np.array([-1, 0, 0.7, 0.3, 0.02, 1.0]), np.array([0.1, 0.1, 0.1, 0.01, 0.001, 0.5]), size=nwalkers)
  # Run the sampler... this will take about 15 seconds for 500 steps.
  start = time.time()
  pos, prob, state = sampler.run_mcmc(p0, 5000, progress=True)
  end = time.time()
  multi_time = end - start
  print("Multiprocessing took {0:.1f} seconds".format(multi_time))
  # Throw away the first 50 samples

  chain_desi = sampler.flatchain[500:,:]
  np.save("MC_desi_sigmas_10_5000.npy", chain)  # Save in NumPy binary format
  np.savetxt("MC_desi_sigmas_10_5000.txt", chain)  # Save as a text file
