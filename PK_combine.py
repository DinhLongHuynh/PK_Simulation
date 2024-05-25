from flask import Flask, request, jsonify, send_file
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

app = Flask(__name__)

@app.route('/pk_combine_dose_regimen', methods=['POST'])
def pk_combine_dose_regimen():
    data = request.json
    dose_iv = data.get('dose_iv', 500)
    infusion_duration = data.get('infusion_duration', 24)
    dose_im = data.get('dose_im', 1300)
    start_im = data.get('start_im', 24)
    interval_dose_im = data.get('interval_dose_im', [0, 24, 48, 72, 96, 120])
    CL = data.get('CL', 7.5)
    Vd = data.get('Vd', 33)
    ke = data.get('ke', 0.228)
    ka = data.get('ka', 0.028)
    F = data.get('F', 1)
    time_range = data.get('time_range', 170)
    IM_profile = data.get('IM_profile', False)
    IV_profile = data.get('IV_profile', False)
    combine_profile = data.get('combine_profile', True)

    # Calculation for iv drug profile only:
    ko = dose_iv / infusion_duration
    dose_im_F = dose_im * F

    time_points_mutual = np.arange(0, time_range, 1)
    time_point_infusion = time_points_mutual[0:(round(infusion_duration) + 1)]  # Timepoint from 0 to finishing infusion
    time_point_elim = time_points_mutual[(round(infusion_duration) + 1):]  # Timepoint from finishing infusio to the end of the time scale

    C_infusion = (ko / CL) * (1 - np.exp(-ke * time_point_infusion))  # Concentration on the infusion side
    C_elim = C_infusion[-1] * np.exp(-ke * (time_point_elim - time_point_infusion[-1]))  # Concentration on the elimination side

    concentrations_iv = np.concatenate((C_infusion, C_elim))  # Combine two concentration to have the whole iv profile

    # Calculation for im drug profile only:
    time_points_im = time_points_mutual[start_im:] - start_im
    concentration_im = np.zeros(len(time_points_im))  # Initialize an array for concentration of im drug.

    for dose_time in interval_dose_im:
        for i, t in enumerate(time_points_im):
            if t >= dose_time:
                concentration_im[i] += (dose_im_F / Vd) * (ka / (ka - ke)) * (
                            np.exp(-ke * (t - dose_time)) - np.exp(-ka * (t - dose_time)))

    concentrations_im = np.concatenate((np.zeros(start_im), concentration_im))  # Add the 0 concentration for time point before injection of im drug

    # Calculation for the combine profile:
    final_concentration = concentrations_iv + concentrations_im

    # Visualization of profiles
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

    # Save plot to a BytesIO object
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    plt.close(fig)

    return send_file(bytes_image, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
