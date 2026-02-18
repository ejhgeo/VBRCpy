#!/usr/bin/env python3
"""
This script is for benchmarking and debugging purposes only.

Compare new MATLAB sweep (test.mat) generated using the original VBRc from within matlab 
to the sweep produced with Python (sweep_fixed.mat).
test.mat was generated with Matlab using generate_parameter_sweep.m with the defaults as 
set in vbr/Projects/bayesian_fitting, where run_bayes.m calls fit_seismic_observations.m, 
which can optionally call generate_parameter_sweep.m. Note that the version of the code 
on GitHub has slightly different inputs from what was in the original Havlin et al., 2021 
paper, so the output figures produced with this sweep file (test.mat) will be slightly 
different than those in the paper. 
"""

import scipy.io as sio
import numpy as np

# Load both sweep files
m1 = sio.loadmat('./vbr/test.mat')
m2 = sio.loadmat('./sweep_fixed.mat')

Box1 = m1['sweep']['Box'][0,0]
Box2 = m2['sweep']['Box'][0,0]

T1 = m1['sweep']['T'][0,0].flatten()
gs1 = m1['sweep']['gs'][0,0].flatten()
phi1 = m1['sweep']['phi'][0,0].flatten()
z1 = m1['sweep']['z'][0,0].flatten()

print("=== Comparing new MATLAB (test.mat) vs Python (sweep_fixed.mat) ===")
print()

# Use grid point: T=1340C, gs=1077um
i_T = 12
i_gs = 10
i_z = 0

print(f"T = {T1[i_T]:.0f}C, gs = {gs1[i_gs]:.0f}um, z = {z1[i_z]/1000:.0f}km")
print()

# Compare at phi=0 and phi=0.02 for andrade_psp
print("=== andrade_psp comparison ===")
for i_phi in [0, 8]:
    e1 = Box1[i_T, i_phi, i_gs]['andrade_psp'][0,0]
    e2 = Box2[i_T, i_phi, i_gs]['andrade_psp'][0,0]
    
    vs1 = e1['meanVs'][0,0].flatten()[i_z]
    vs2 = e2['meanVs'][0,0].flatten()[i_z]
    q1 = e1['meanQ'][0,0].flatten()[i_z]
    q2 = e2['meanQ'][0,0].flatten()[i_z]
    
    print(f"phi={phi1[i_phi]:.4f}:")
    print(f"  Vs: MATLAB={vs1:.6f}, Python={vs2:.6f}, diff={(vs2-vs1)*1000:.2f} m/s ({(vs2-vs1)/vs1*100:.4f}%)")
    print(f"  Q:  MATLAB={q1:.4f}, Python={q2:.4f}, diff={q2-q1:.4f} ({(q2-q1)/q1*100:.4f}%)")
    print()

# Compute melt effect for andrade_psp
e1_0 = Box1[i_T, 0, i_gs]['andrade_psp'][0,0]
e1_m = Box1[i_T, 8, i_gs]['andrade_psp'][0,0]
e2_0 = Box2[i_T, 0, i_gs]['andrade_psp'][0,0]
e2_m = Box2[i_T, 8, i_gs]['andrade_psp'][0,0]

vs1_0 = e1_0['meanVs'][0,0].flatten()[i_z]
vs1_m = e1_m['meanVs'][0,0].flatten()[i_z]
vs2_0 = e2_0['meanVs'][0,0].flatten()[i_z]
vs2_m = e2_m['meanVs'][0,0].flatten()[i_z]

print("=== Melt effect (phi=0.02 vs phi=0) for andrade_psp ===")
print(f"MATLAB: dVs = {(vs1_m - vs1_0)*1000:.2f} m/s ({(vs1_m - vs1_0)/vs1_0*100:.3f}%)")
print(f"Python: dVs = {(vs2_m - vs2_0)*1000:.2f} m/s ({(vs2_m - vs2_0)/vs2_0*100:.3f}%)")
print(f"Difference: {((vs2_m - vs2_0) - (vs1_m - vs1_0))*1000:.2f} m/s")
print()

# Also check eburgers_psp
print("=== eburgers_psp comparison ===")
for i_phi in [0, 8]:
    e1 = Box1[i_T, i_phi, i_gs]['eburgers_psp'][0,0]
    e2 = Box2[i_T, i_phi, i_gs]['eburgers_psp'][0,0]
    
    vs1 = e1['meanVs'][0,0].flatten()[i_z]
    vs2 = e2['meanVs'][0,0].flatten()[i_z]
    q1 = e1['meanQ'][0,0].flatten()[i_z]
    q2 = e2['meanQ'][0,0].flatten()[i_z]
    
    print(f"phi={phi1[i_phi]:.4f}:")
    print(f"  Vs: MATLAB={vs1:.6f}, Python={vs2:.6f}, diff={(vs2-vs1)*1000:.2f} m/s ({(vs2-vs1)/vs1*100:.4f}%)")
    print(f"  Q:  MATLAB={q1:.4f}, Python={q2:.4f}, diff={q2-q1:.4f} ({(q2-q1)/q1*100:.4f}%)")
    print()

# Compute melt effect for eburgers_psp
e1_0 = Box1[i_T, 0, i_gs]['eburgers_psp'][0,0]
e1_m = Box1[i_T, 8, i_gs]['eburgers_psp'][0,0]
e2_0 = Box2[i_T, 0, i_gs]['eburgers_psp'][0,0]
e2_m = Box2[i_T, 8, i_gs]['eburgers_psp'][0,0]

vs1_0 = e1_0['meanVs'][0,0].flatten()[i_z]
vs1_m = e1_m['meanVs'][0,0].flatten()[i_z]
vs2_0 = e2_0['meanVs'][0,0].flatten()[i_z]
vs2_m = e2_m['meanVs'][0,0].flatten()[i_z]

print("=== Melt effect (phi=0.02 vs phi=0) for eburgers_psp ===")
print(f"MATLAB: dVs = {(vs1_m - vs1_0)*1000:.2f} m/s ({(vs1_m - vs1_0)/vs1_0*100:.3f}%)")
print(f"Python: dVs = {(vs2_m - vs2_0)*1000:.2f} m/s ({(vs2_m - vs2_0)/vs2_0*100:.3f}%)")
print(f"Difference: {((vs2_m - vs2_0) - (vs1_m - vs1_0))*1000:.2f} m/s")
print()

# Also check xfit_premelt
print("=== xfit_premelt comparison ===")
for i_phi in [0, 8]:
    e1 = Box1[i_T, i_phi, i_gs]['xfit_premelt'][0,0]
    e2 = Box2[i_T, i_phi, i_gs]['xfit_premelt'][0,0]
    
    vs1 = e1['meanVs'][0,0].flatten()[i_z]
    vs2 = e2['meanVs'][0,0].flatten()[i_z]
    q1 = e1['meanQ'][0,0].flatten()[i_z]
    q2 = e2['meanQ'][0,0].flatten()[i_z]
    
    print(f"phi={phi1[i_phi]:.4f}:")
    print(f"  Vs: MATLAB={vs1:.6f}, Python={vs2:.6f}, diff={(vs2-vs1)*1000:.2f} m/s ({(vs2-vs1)/vs1*100:.4f}%)")
    print(f"  Q:  MATLAB={q1:.4f}, Python={q2:.4f}, diff={q2-q1:.4f} ({(q2-q1)/q1*100:.4f}%)")
    print()

e1_0 = Box1[i_T, 0, i_gs]['xfit_premelt'][0,0]
e1_m = Box1[i_T, 8, i_gs]['xfit_premelt'][0,0]
e2_0 = Box2[i_T, 0, i_gs]['xfit_premelt'][0,0]
e2_m = Box2[i_T, 8, i_gs]['xfit_premelt'][0,0]

vs1_0 = e1_0['meanVs'][0,0].flatten()[i_z]
vs1_m = e1_m['meanVs'][0,0].flatten()[i_z]
vs2_0 = e2_0['meanVs'][0,0].flatten()[i_z]
vs2_m = e2_m['meanVs'][0,0].flatten()[i_z]

print("=== Melt effect (phi=0.02 vs phi=0) for xfit_premelt ===")
print(f"MATLAB: dVs = {(vs1_m - vs1_0)*1000:.2f} m/s ({(vs1_m - vs1_0)/vs1_0*100:.3f}%)")
print(f"Python: dVs = {(vs2_m - vs2_0)*1000:.2f} m/s ({(vs2_m - vs2_0)/vs2_0*100:.3f}%)")
print(f"Difference: {((vs2_m - vs2_0) - (vs1_m - vs1_0))*1000:.2f} m/s")
print()

# Also check xfit_mxw
print("=== xfit_mxw comparison ===")
for i_phi in [0, 8]:
    e1 = Box1[i_T, i_phi, i_gs]['xfit_mxw'][0,0]
    e2 = Box2[i_T, i_phi, i_gs]['xfit_mxw'][0,0]
    
    vs1 = e1['meanVs'][0,0].flatten()[i_z]
    vs2 = e2['meanVs'][0,0].flatten()[i_z]
    q1 = e1['meanQ'][0,0].flatten()[i_z]
    q2 = e2['meanQ'][0,0].flatten()[i_z]
    
    print(f"phi={phi1[i_phi]:.4f}:")
    print(f"  Vs: MATLAB={vs1:.6f}, Python={vs2:.6f}, diff={(vs2-vs1)*1000:.2f} m/s ({(vs2-vs1)/vs1*100:.4f}%)")
    print(f"  Q:  MATLAB={q1:.4f}, Python={q2:.4f}, diff={q2-q1:.4f} ({(q2-q1)/q1*100:.4f}%)")
    print()

e1_0 = Box1[i_T, 0, i_gs]['xfit_mxw'][0,0]
e1_m = Box1[i_T, 8, i_gs]['xfit_mxw'][0,0]
e2_0 = Box2[i_T, 0, i_gs]['xfit_mxw'][0,0]
e2_m = Box2[i_T, 8, i_gs]['xfit_mxw'][0,0]

vs1_0 = e1_0['meanVs'][0,0].flatten()[i_z]
vs1_m = e1_m['meanVs'][0,0].flatten()[i_z]
vs2_0 = e2_0['meanVs'][0,0].flatten()[i_z]
vs2_m = e2_m['meanVs'][0,0].flatten()[i_z]

print("=== Melt effect (phi=0.02 vs phi=0) for xfit_mxw ===")
print(f"MATLAB: dVs = {(vs1_m - vs1_0)*1000:.2f} m/s ({(vs1_m - vs1_0)/vs1_0*100:.3f}%)")
print(f"Python: dVs = {(vs2_m - vs2_0)*1000:.2f} m/s ({(vs2_m - vs2_0)/vs2_0*100:.3f}%)")
print(f"Difference: {((vs2_m - vs2_0) - (vs1_m - vs1_0))*1000:.2f} m/s")
