import itertools as it
import pandas as pd
import numpy as np
from python_data_analysis.data_cleaning import separateSweeps, subtractOffset, offsetData
from scipy.optimize import curve_fit
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


def sliceFields(df, fields, plane_string, device='delta_f', fig=None, ax=None):
    df_ang = df.reset_index().round({'Field (T)': 2}).set_index(
        ['Angle (deg)', 'Field (T)']).groupby(['Angle (deg)', 'Field (T)']).mean()
    df_ang = df_ang.reset_index().set_index('Angle (deg)')

    if fig == None:
        fig, ax = plt.subplots()

    cmap = plt.get_cmap('viridis')
    ax.set_prop_cycle(color=[cmap(1.*k/len(fields))
                      for k in range(len(fields))])

    for field in fields:

        df_sub = df_ang[df_ang['Field (T)'] == field]
        df_sub.plot(y=device, ax=ax, label='{}T'.format(field), marker='o',
                    ylabel='$\Delta f$ (Hz)', xlabel=r'$\theta_{{{}}}$ (degrees)'.format(plane_string))

    ax.legend(ncol=3, fontsize=8)


def convertToK(angles, delta_f, chi_i, chi_j, field_eval,
               one_comp=True, **kwargs):
    fig, ax = plt.subplots()

    x_eval = np.linspace(angles.min(), angles.max(), 100)
    if one_comp:
        def sinusoid(x, A, phase, bg):
            return A * np.cos(2*np.deg2rad(x) + phase) + bg

        popt, pcov = curve_fit(sinusoid, angles, delta_f, **kwargs)

        A = popt[0]

        ax.plot(x_eval, sinusoid(x_eval, *popt), label=r'$\cos{2\theta}$')
        ax.plot(angles, delta_f)
        ax.legend()
    else:
        def cos(x, A1, phase): return -A1 * np.cos(2*np.deg2rad(x) + phase)
        def sin(x, A2, phase): return A2 * np.sin(2*np.deg2rad(x) + phase)

        def sinusoid2comp(xdata, A1, A2, phase, bg):
            return cos(xdata, A1, phase) + sin(xdata, A2, phase) + bg
            # return -A1 * np.cos(2*np.deg2rad(xdata) + phase) + A2 * np.sin(2*np.deg2rad(xdata) + phase) + bg

        popt, pcov = curve_fit(sinusoid2comp, angles, delta_f, **kwargs)
        A = popt[0]

        ax.plot(x_eval, cos(x_eval, popt[0],
                popt[2]), label=r'$\cos{2\theta}$')
        ax.plot(x_eval, sin(x_eval, popt[1],
                popt[2]), label=r'$\sin{2\theta}$')
        ax.plot(x_eval, sinusoid2comp(x_eval, *popt),
                label=r'$\cos{2\theta} + \sin{2\theta}$')
        ax.legend()

    # Calculate the conversion constant C; k = C * delta_f
    C = np.abs(field_eval ** 2 * (chi_i - chi_j) / A) * \
        9.274e-1  # last factor is to get J/mol from mu_B

    return C


def extractConversion(df, field_eval, chi_i, chi_j, plane_string, tol=0.01,
                      one_comp=True, **kwargs):
    '''
    Function for extracting the conversion constant between the
    frequency shift delta_f and the magnetotropic susceptibility k

    This is done by fitting the angle dependence in linear response.

    Note: this method currently only works with magnetic crystals
    structures with at least orthorhombic symmetry

    inputs
      df: dataframe with 'Field (T)', 'Angle (deg)', and 'delta_f' as
          columns; main data set should be comprised of field sweeps
          at various angles
      field_eval: field at which the fit is to be performed
      chi_i, chi_j: values of the susceptibilities (molar or per ion,
                    input is based on user's desire)
      plane_string: specifies which plane the measurement is done in
      tol : average data between field_eval+tol and field_eval-tol
            default is 0.01

    outputs
      C: conversion constant between delta_f and k
    '''
    fig, ax = plt.subplots()

    #Slice the dataframe at the specific field
    #df_slice = df.round({'Field (T)':2}).set_index(['Angle (deg)','Field (T)']).groupby(['Angle (deg)','Field (T)']).mean().reset_index()
    df_slice = df[(df['Field (T)'] > field_eval-tol) & (df['Field (T)']
                                                        < field_eval+tol)].groupby('Angle (deg)').mean().reset_index()
    df_slice.plot(x='Angle (deg)', y='delta_f', ax=ax, label='{}T'.format(field_eval), marker='o',
                  ylabel='$\Delta f$ (Hz)', xlabel=r'$\theta_{{{}}}$ (degrees)'.format(plane_string))

    xdata = df_slice['Angle (deg)'].values
    ydata = df_slice['delta_f'].values
    x_eval = np.linspace(xdata.min(), xdata.max(), 100)

    #Fit the frequency response and extract the amplitude
    #Define cos(2theta) curve_fit function
    if one_comp:
        def sinusoid(x, A, phase, bg):
            return A * np.cos(2*np.deg2rad(x) + phase) + bg

        popt, pcov = curve_fit(sinusoid, xdata, ydata, **kwargs)

        A = popt[0]

        ax.plot(x_eval, sinusoid(x_eval, *popt), label=r'$\cos{2\theta}$')
        ax.legend()
    else:
        def cos(x, A1, phase): return -A1 * np.cos(2*np.deg2rad(x) + phase)
        def sin(x, A2, phase): return A2 * np.sin(2*np.deg2rad(x) + phase)

        def sinusoid2comp(xdata, A1, A2, phase, bg):
            return cos(xdata, A1, phase) + sin(xdata, A2, phase) + bg
            #return -A1 * np.cos(2*np.deg2rad(xdata) + phase) + A2 * np.sin(2*np.deg2rad(xdata) + phase) + bg

        popt, pcov = curve_fit(sinusoid2comp, xdata, ydata, **kwargs)
        A = popt[0]

        ax.plot(x_eval, cos(x_eval, popt[0],
                popt[2]), label=r'$\cos{2\theta}$')
        ax.plot(x_eval, sin(x_eval, popt[1],
                popt[2]), label=r'$\sin{2\theta}$')
        ax.plot(x_eval, sinusoid2comp(x_eval, *popt),
                label=r'$\cos{2\theta} + \sin{2\theta}$')
        ax.legend()

    # Calculate the conversion constant C; k = C * delta_f
    C = np.abs(field_eval ** 2 * (chi_i - chi_j) / A) * \
        9.274e-1  # last factor is to get J/mol from mu_B

    return C


def calcRatiosDerivs(df, column='k', filt_window=51, deriv_window=21, field_thresh=0.1):

    df['{}/B'.format(column)] = df[column] / df['Field (T)']
    df['{}/B2'.format(column)] = df[column] / df['Field (T)'] ** 2

    # making the quantities which divide by field nan
    # below 0.1 T, to help w/ noise and filtering
    df.loc[df['Field (T)'] < field_thresh, [
        '{}/B'.format(column), '{}/B2'.format(column)]] = np.nan

    # being sure the drop the nan values from the filter
    # filter settings are hard-coded in, should probably allow for some input
    df['{}/B2'.format(column)] = df.dropna().groupby('Angle (deg)')[
        '{}/B2'.format(column)].transform(lambda x: savgol_filter(x, filt_window, 3))
    df['d{}_h/dh'.format(column)] = df.dropna().groupby('Angle (deg)')['{}/B'.format(
        column)].transform(lambda x: savgol_filter(x, deriv_window, 3, deriv=1))
    df['d{}_h/dh'.format(column)] = df.dropna().groupby('Angle (deg)')[
        'd{}_h/dh'.format(column)].transform(lambda x: savgol_filter(x, filt_window, 3))

    df['d{}_h2/dh'.format(column)] = df.dropna().groupby('Angle (deg)')['{}/B2'.format(
        column)].transform(lambda x: savgol_filter(x, deriv_window, 3, deriv=1))
    df['d{}_h2/dh'.format(column)] = df.dropna().groupby('Angle (deg)')[
        'd{}_h2/dh'.format(column)].transform(lambda x: savgol_filter(x, filt_window, 3))

    # Calculating derivative
    df['d{}/dh'.format(column)] = df.groupby('Angle (deg)')[
        column].transform(lambda x: savgol_filter(x, 151, 3, deriv=1))

    return df
