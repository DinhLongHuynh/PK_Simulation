import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def pk_combine_dose_regimen(dose_iv= 500,
                             infusion_duration = 24,
                             dose_im = 1300,
                             start_im = 24,
                             interval_dose_im = [0, 24, 48,72,96,120],
                             CL = 7.5,
                             Vd = 33,
                             ke= 0.228,
                             ka = 0.028,
                             F=1,
                             time_range = 170,
                             IM_profile = False,
                             IV_profile = False,
                             combine_profile=True):
    ko = dose_iv / infusion_duration
    dose_im_F = dose_im * F

    time_points_mutual = np.arange(0, time_range, 1)
    time_point_infusion = time_points_mutual[0:(round(infusion_duration) + 1)]
    time_point_elim = time_points_mutual[(round(infusion_duration) + 1):]

    C_infusion = (ko / CL) * (1 - np.exp(-ke * time_point_infusion))
    C_elim = C_infusion[-1] * np.exp(-ke * (time_point_elim - time_point_infusion[-1]))

    concentrations_iv = np.concatenate((C_infusion, C_elim))

    time_points_im = time_points_mutual[start_im:] - start_im
    concentration_im = np.zeros(len(time_points_im))

    for dose_time in interval_dose_im:
        for i, t in enumerate(time_points_im):
            if t >= dose_time:
                concentration_im[i] += (dose_im_F / Vd) * (ka / (ka - ke)) * (
                            np.exp(-ke * (t - dose_time)) - np.exp(-ka * (t - dose_time)))

    concentrations_im = np.concatenate((np.zeros(start_im), concentration_im))
    final_concentration = concentrations_iv + concentrations_im

    fig, ax = plt.subplots(figsize=(12, 6))
    if IM_profile:
        sns.lineplot(x=time_points_mutual, y=concentrations_im, ax=ax, label='IM')
    if IV_profile:
        sns.lineplot(x=time_points_mutual, y=concentrations_iv, ax=ax, label='IV')
    if combine_profile:
        sns.lineplot(x=time_points_mutual, y=final_concentration, ax=ax, label='Superimposition')

    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration (mg/L)')
    plt.title('PK profile')
    st.pyplot(fig)

st.title("PK Simulation of Combined Dosing Regimen IV and IM/Oral Drugs")

dose_iv = st.number_input("Dose IV (mg)", value=500)
infusion_duration = st.number_input("Infusion Duration (h)", value=24)
dose_im = st.number_input("Dose IM (mg)", value=1300)
start_im = st.number_input("Start IM (h)", value=24)
interval_dose_im = st.text_input("Interval Dose IM (h)", "0, 24, 48, 72, 96, 120")
interval_dose_im = [int(x) for x in interval_dose_im.split(",")]
CL = st.number_input("Clearance (L/h)", value=7.5)
Vd = st.number_input("Volume of Distribution (L)", value=33)
ke = st.number_input("Elimination Constant (h-1)", value=0.228)
ka = st.number_input("Absorption Constant (h-1)", value=0.028)
F = st.number_input("Bioavailability", value=1.0)
time_range = st.number_input("Time Range (h)", value=170)
IM_profile = st.checkbox("Display IM Profile", value=False)
IV_profile = st.checkbox("Display IV Profile", value=False)
combine_profile = st.checkbox("Display Combined Profile", value=True)

if st.button("Run Simulation"):
    pk_combine_dose_regimen(dose_iv, infusion_duration, dose_im, start_im, interval_dose_im, CL, Vd, ke, ka, F, time_range, IM_profile, IV_profile, combine_profile)
