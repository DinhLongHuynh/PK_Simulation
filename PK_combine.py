import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define functions for each simulation

def pk_combine_dose_regimen(dose_iv=500, infusion_duration=24, dose_im=1300, start_im=24, interval_dose_im=[0, 24, 48, 72, 96, 120], 
                            CL=7.5, Vd=33, ke=0.228, ka=0.028, F=1, time_range=170, IM_profile=False, IV_profile=False, combine_profile=True):
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
                concentration_im[i] += (dose_im_F / Vd) * (ka / (ka - ke)) * (np.exp(-ke * (t - dose_time)) - np.exp(-ka * (t - dose_time)))

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

def pk_simulation(dose=100, CL_pop=2, V_pop=50, ka_pop=None, F_pop=1, n_patients=1, omegaCL=0, omegaV=0, omegaka=0, omegaF=0, 
                  C_limit=None, sampling_points=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24], logit=False):
    time = np.array(sampling_points).reshape(1, len(sampling_points))
    CV_V = norm.rvs(loc=0, scale=omegaV, size=n_patients)
    V_variability = V_pop * np.exp(CV_V)
    V_var = V_variability.reshape(n_patients, 1)
    CV_CL = norm.rvs(loc=0, scale=omegaCL, size=n_patients)
    CL_variability = CL_pop * np.exp(CV_CL)
    CL_var = CL_variability.reshape(n_patients, 1)
    CV_F = norm.rvs(loc=0, scale=omegaF, size=n_patients)
    F_variability = F_pop * np.exp(CV_F)
    F_var = F_variability.reshape(n_patients, 1)
    ke_var = CL_var / V_var

    if ka_pop is None:
        concentration = (dose * F_var / V_var) * np.exp(np.dot(-ke_var, time))
    else:
        CV_ka = norm.rvs(loc=0, scale=omegaka, size=n_patients)
        ka_variability = ka_pop * np.exp(CV_ka)
        ka_var = ka_variability.reshape(n_patients, 1)
        concentration = (dose * F_var * ka_var) / (V_var * (ka_var - ke_var)) * (np.exp(np.dot(-ke_var, time)) - np.exp(np.dot(-ka_var, time)))

    df_C = pd.DataFrame(concentration, columns=sampling_points)
    df_C.replace([np.inf, -np.inf], np.nan, inplace=True)
    concentration_ln = np.log(concentration)
    df_C_ln = pd.DataFrame(concentration_ln, columns=sampling_points)
    df_C_ln.replace([np.inf, -np.inf], np.nan, inplace=True)

    fig, ax = plt.subplots()
    for i in range(n_patients):
        if logit:
            pk_data = df_C_ln.iloc[i, :]
            sns.lineplot(x=sampling_points, y=pk_data, ax=ax, color='green')
            ax.set_ylabel('Log[Concentration] (mg/L)')
            ax.set_title('PK simulation')
            if C_limit is not None:
                plt.axhline(y=np.log(C_limit), color='r', linestyle='--')
        else:
            pk_data = df_C.iloc[i, :]
            sns.lineplot(x=sampling_points, y=pk_data, ax=ax, color='green')
            ax.set_ylabel('Concentration (mg/L)')
            ax.set_title('PK simulation')
            if C_limit is not None:
                plt.axhline(y=C_limit, color='r', linestyle='--')
        ax.set_xlabel('Time (h)')
    st.pyplot(fig)

def pd_simulation(Emax=6.43, EC50=5.38, Ebaseline=1, hill=1, n_patients=1, omegaEmax=0, omegaEC50=0, omegaEbaseline=0, omegahill=0, 
                  E_limit=None, sampling_conc=[2, 4, 5, 6, 7, 8, 10, 20, 50, 100]):
    Ebaseline_CV = norm.rvs(loc=0, scale=omegaEbaseline, size=n_patients)
    Emax_CV = norm.rvs(loc=0, scale=omegaEmax, size=n_patients)
    EC50_CV = norm.rvs(loc=0, scale=omegaEC50, size=n_patients)
    hill_CV = norm.rvs(loc=0, scale=omegahill, size=n_patients)
    Ebaseline_var = (Ebaseline * np.exp(Ebaseline_CV)).reshape(n_patients, 1)
    Emax_var = (Emax * np.exp(Emax_CV)).reshape(n_patients, 1)
    EC50_var = (EC50 * np.exp(EC50_CV)).reshape(n_patients, 1)
    hill_var = (hill * np.exp(hill_CV)).reshape(n_patients, 1)
    conc_list = np.array(sampling_conc).reshape(1, len(sampling_conc))
    E_array = Ebaseline_var + Emax_var * (conc_list ** hill_var) / (EC50_var + conc_list)
    E_df = pd.DataFrame(E_array, columns=sampling_conc)
    
    fig, ax = plt.subplots()
    for i in range(n_patients):
        pd_data = E_df.iloc[i, :]
        sns.lineplot(x=sampling_conc, y=pd_data, ax=ax, color='green')
    if E_limit is not None:
        plt.axhline(y=E_limit, color='r', linestyle='--')
    ax.set_ylabel('Effect')
    ax.set_xlabel('Concentration')
    ax.set_title('PD simulation')
    st.pyplot(fig)

# Define the layout of the app
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["PK Combined Dose Regimen", "PK Simulation", "PD Simulation"])

if page == "PK Combined Dose Regimen":
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
    IM_profile = st.checkbox("Show IM Profile", value=False)
    IV_profile = st.checkbox("Show IV Profile", value=False)
    combine_profile = st.checkbox("Show Combined Profile", value=True)

    if st.button("Run Simulation"):
        pk_combine_dose_regimen(dose_iv, infusion_duration, dose_im, start_im, interval_dose_im, CL, Vd, ke, ka, F, time_range, IM_profile, IV_profile, combine_profile)

elif page == "PK Simulation":
    st.title("PK Simulation")
    dose = st.number_input("Dose (mg)", value=100)
    CL_pop = st.number_input("CL Population Mean (L/h)", value=2.0)
    V_pop = st.number_input("V Population Mean (L)", value=50)
    ka_pop = st.number_input("ka Population Mean (h-1)", value=0.1)
    F_pop = st.number_input("F Population Mean", value=1.0)
    n_patients = st.number_input("Number of Patients", value=1)
    omegaCL = st.number_input("Omega CL", value=0.2)
    omegaV = st.number_input("Omega V", value=0.2)
    omegaka = st.number_input("Omega ka", value=0.2)
    omegaF = st.number_input("Omega F", value=0.2)
    C_limit = st.number_input("C Limit (mg/L)", value=None)
    sampling_points = st.text_input("Sampling Points (h)", "0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24")
    sampling_points = [int(x) for x in sampling_points.split(",")]
    logit = st.checkbox("Log Transformation", value=False)

    if st.button("Run Simulation"):
        pk_simulation(dose, CL_pop, V_pop, ka_pop, F_pop, n_patients, omegaCL, omegaV, omegaka, omegaF, C_limit, sampling_points, logit)

elif page == "PD Simulation":
    st.title("PD Simulation")
    Emax = st.number_input("Emax", value=6.43)
    EC50 = st.number_input("EC50", value=5.38)
    Ebaseline = st.number_input("Ebaseline", value=1.0)
    hill = st.number_input("Hill Coefficient", value=1.0)
    n_patients = st.number_input("Number of Patients", value=1)
    omegaEmax = st.number_input("Omega Emax", value=0.2)
    omegaEC50 = st.number_input("Omega EC50", value=0.2)
    omegaEbaseline = st.number_input("Omega Ebaseline", value=0.2)
    omegahill = st.number_input("Omega Hill", value=0.2)
    E_limit = st.number_input("E Limit", value=None)
    sampling_conc = st.text_input("Sampling Concentrations", "2, 4, 5, 6, 7, 8, 10, 20, 50, 100")
    sampling_conc = [int(x) for x in sampling_conc.split(",")]

    if st.button("Run Simulation"):
        pd_simulation(Emax, EC50, Ebaseline, hill, n_patients, omegaEmax, omegaEC50, omegaEbaseline, omegahill, E_limit, sampling_conc)
