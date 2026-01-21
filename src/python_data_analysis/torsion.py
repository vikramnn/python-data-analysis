import itertools as it
import pandas as pd
import numpy as np
from python_data_analysis.data_cleaning import separateSweeps, subtractOffset, offsetData
from scipy.signal import savgol_filter
import scipy.signal as signal
import matplotlib.pyplot as plt


def constructDF(path, shot_list, file_list, step, field_range):
    '''
    Constructing a dataframe from a list of files; meant to be
    used predominantly for LANL data
    '''
    dfs = []
    field_eval = np.arange(field_range[0], field_range[1], step)

    for angle, string in shot_list:

        file = next(it.dropwhile(lambda s: string not in s, file_list))
        df = pd.read_csv(path+file, names=['Field (T)', 'Freq (Hz)',
                         'Amp (V)'], header=None, delimiter='\t', skiprows=0)

        df = df.set_index('Field (T)')

        df = separateSweeps(df, sweep_dir='up', sweep_select='down')
        df.sort_index(inplace=True)
        field = df.index.values
        freq = df['Freq (Hz)'].values
        amp = df['Amp (V)'].values

        freq = np.interp(field_eval, field, freq)
        amp = np.interp(field_eval, field, amp)

        df = pd.DataFrame(
            {'Field (T)': field_eval, 'Freq (Hz)': freq, 'Amp (V)': amp})
        df['Angle (deg)'] = angle

        dfs.append(df)

    df = pd.concat(dfs)
    df.set_index(['Angle (deg)', 'Field (T)'], inplace=True)
    return df


def filterData(df, savgol_args=(21, 3),
               savgol_kwargs={'deriv': 1, 'mode': 'mirror'},
               butter_args=(2, 0.005)):
    '''
    Filter data and calculate derivative
    '''

    freq_temp = df['Freq (Hz)'].values
    df = df.groupby('Angle (deg)').apply(subtractOffset, 'Freq (Hz)')
    df['delta_f'] = df['Freq (Hz)']
    df['Freq (Hz)'] = freq_temp

    # Calculating derivative
    df['df/dh'] = df.groupby('Angle (deg)')['delta_f'].transform(
        lambda x: savgol_filter(x, *savgol_args, **savgol_kwargs))

    # butterworth low-pass filter
    b, a = signal.butter(*butter_args)
    df['df/dh'] = df.groupby('Angle (deg)')['df/dh'].transform(
        lambda x: signal.filtfilt(b, a, x))

    return df


def plotFieldsweeps(df,  offset_data_args=(50, 'delta_f')):
    '''
    1D plots of frequency fieldsweeps and derivative
    '''
    f_temp = df['delta_f'].values
    df = df.groupby('Angle (deg)').pipe(offsetData, *offset_data_args)
    df['delta_f_offset'] = df['delta_f']
    df['delta_f'] = f_temp

    num_colors = np.unique(df.index.get_level_values(0)).size
    cmap = plt.get_cmap('cividis')

    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=[cmap(1.*k/num_colors) for k in range(num_colors)])
    df.reset_index('Angle (deg)').groupby('Angle (deg)')[
        'delta_f_offset'].plot(ax=ax, legend=True)
    ax.legend(ncol=5, fontsize=5)
    ax.set_xlabel('$\mu_0 H$ (T)', fontsize=10)
    ax.set_ylabel('$\Delta f$ (Hz)', fontsize=10, labelpad=-5)

    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=[cmap(1.*k/num_colors) for k in range(num_colors)])
    df.reset_index('Angle (deg)').groupby('Angle (deg)')[
        'df/dh'].plot(ax=ax, legend=True)
    ax.legend(ncol=5, fontsize=5)
    ax.set_xlabel('$\mu_0 H$ (T)', fontsize=10)
    ax.set_ylabel('$df/dH$ (Hz/T)', fontsize=10, labelpad=-5)


def plotColormaps(df, field_eval, plane_string):

    angles = np.unique(df.index.get_level_values(0))
    num_colors = angles.size
    reshape_args = (num_colors, field_eval.size)

    ax_delta_f = plt.subplot(2, 1, 1)
    ax_dfdh = plt.subplot(2, 1, 2)

    Δf = df['delta_f'].values.reshape(*reshape_args)
    df_dh = df['df/dh'].values.reshape(*reshape_args)
    X, Y = np.meshgrid(field_eval, angles)

    ax_dfdh.set_xlabel('$\mu_0 H$ (T)', fontsize=10)
    ax_delta_f.set_ylabel(
        r'$\theta_{{{}}}$ (deg)'.format(plane_string), fontsize=10)
    ax_dfdh.set_ylabel(r'$\theta_{{{}}}$ (deg)'.format(
        plane_string), fontsize=10)
    pcdf = ax_delta_f.pcolormesh(X, Y, Δf)
    pcdfdh = ax_dfdh.pcolormesh(X, Y, df_dh)
    cbardf = plt.colorbar(pcdf, ax=ax_delta_f)
    cbardfdh = plt.colorbar(pcdfdh, ax=ax_dfdh)
    cbardf.set_label('$\Delta f$ (Hz)', rotation=270, fontsize=10, labelpad=10)
    cbardfdh.set_label('$df/dH$ (Hz/T)', rotation=270,
                       fontsize=10, labelpad=10)
