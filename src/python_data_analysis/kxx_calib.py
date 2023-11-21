import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev, chebval
from scipy.interpolate import interp1d


def plotCalib(df_calib, thermometers):
    """
    Plot the raw calibration data in a calibration dataframe.

    Specifically: plots T vs R at different fields

    Parameters:
        df_calib:     dataframe w/ calibration data; typically Field
                      is rounded to 2 decimal places, mask is already
                      filtered to mask == 1
        thermometers: list of dataframe column names corresponding to
                      the thermometer resistances

    Returns:
        df_avg: dataframe w/ averaged resistances and temperatures
    """
    # thermometers is a list of columns in the dataframe that need to be calibrated
    fig, axes = plt.subplots(len(thermometers), 1)
    fields = np.unique(df_calib['Field (T)'].values)
    num_colors = len(fields)
    cmap = plt.get_cmap('viridis')

    for ax in axes:
       ax.set_prop_cycle(color=[cmap(1.*k/num_colors) for k in range(num_colors)]) 

    df_avg = (df_calib.groupby(['Field (T)','Temp_round (K)'], as_index=False).filter(lambda x: len(x.index)>50)
               .groupby(['Field (T)','Temp_round (K)'], as_index=False).mean())

    for i, therm in enumerate(thermometers):
         df_avg.set_index(therm).groupby('Field (T)')['Temperature (K)'].plot(ax=axes[i], linewidth=0, marker='o', ylabel='Temperature (K)', xlabel='{} ($\Omega$)'.format(therm))
    
    axes[0].legend(list(map(lambda x: '{}T'.format(x), fields)), ncol=3)
    fig.tight_layout()
    
    return df_avg


def minMax(df_avg, thermometers):
    """
    Calculate minima and maxima of the thermometer resistances.

    Use this function for 1 set of data, i.e. for 1 magnetic field 

    Parameters
        df_avg:       dataframe w/ average resistances and temperatures
        thermometers: list of dataframe column names corresponding to thermometer resistances

    Returns:
        domain_dict: dictionary with thermometer names as keys and [max,min] list as values
    """
    domain_dict = {}

    for therm in thermometers:
        domain_dict[therm] = [df_avg[therm].max(), df_avg[therm].min()]

    return domain_dict


def minMaxInterpSpline(df_interp, fields, thermometers, kind='cubic'):
    """
    Interpolate the minima and maxima of thermometer data at different magnetic fields.

    Use this function if you have many temperature calibrations at different fields
    It also plots the interpolations for each thermometer min, max

    Parameters
        df_interp:    dataframe w/ resistance values to be interpolated
        fields:       field values to interpolate between
        thermometers: list of dataframe column names corresponding to thermometer resistances
        kind='cubic': specifies the type of spline interpolation passed to
                      scipy.interpolate.interp1d

    Returns
        domain_interp_dict: dictionary with the thermometer names as keys and the interpolation
                            functions [f_interp_max, f_interp_min] as values
    """
    fig, axes = plt.subplots(len(thermometers),2)
    x_eval = np.linspace(fields[0], fields[-1], 100)

    domain_interp_dict = {}

    for i, therm in enumerate(thermometers):
        df_interp.groupby('Field (T)').max()[therm].plot(ax=axes[i,0], y=therm, linewidth=0, marker='o')
        df_interp.groupby('Field (T)').min()[therm].plot(ax=axes[i,1], y=therm, linewidth=0, marker='o')

        rmax = df_interp.groupby('Field (T)').max()[therm].values
        rmin = df_interp.groupby('Field (T)').min()[therm].values

        if len(fields) > 1:

            f_interp_max = interp1d(fields, rmax, kind=kind)
            axes[i, 0].plot(x_eval, f_interp_max(x_eval))
            axes[i, 0].set_ylabel('{}_max'.format(therm))

            f_interp_min = interp1d(fields, rmin, kind=kind)
            axes[i, 1].plot(x_eval, f_interp_min(x_eval))
            axes[i, 1].set_ylabel('{}_min'.format(therm))

            domain_interp_dict[therm] = [f_interp_max, f_interp_min]

    fig.tight_layout()
    return domain_interp_dict


def chebyCalibFields(df_avg, cheby_deg, fields, thermometers):
    """
    Perform Chebyshev calibrations at many different fields for all thermometers.

    Also plots fits at each field for each thermometer

    Parameters
        df_avg:       dataframe with R/T values to be fitted
        cheby_deg:    degree of the Chebyshev polynomial to fit
        fields:       fields at which the calibrations were performed
        thermometers: list of dataframe column names corresponding to thermometer resistances

    Returns
        coeffs_dict: dictionary where the keys are the thermometer names and the values are a
                     list of arrays of Chebyshev coefficients returned from Chebyshev.fit(...).coef;
                     the number of arrays in each list corresponds to the number of field values,
                     and the order is dictated by the order of fields passed, typically smallest to
                     largest
    """
    coeffs_dict = {}

    plot_dim = int(np.ceil(np.sqrt(len(fields))))

    for therm in thermometers:
        coeffs_dict[therm] = []

        fig, axes = plt.subplots(plot_dim, plot_dim)

        if plot_dim != 1:
            axes = axes.flatten()

        for i, field in enumerate(fields):
            df_slice = df_avg[df_avg['Field (T)'] == field]
            R = df_slice[therm].values
            temperature = df_slice['Temperature (K)'].values

            domain_R = np.log(np.array([R.min(), R.max()]))

            x = np.linspace(-1, 1, 100)

            # Fitting
            coeffs = Chebyshev.fit(np.log(R), np.log(temperature), cheby_deg, domain=domain_R).coef

            coeffs_dict[therm].append(coeffs)

            z = (x*(domain_R[1] - domain_R[0]) + domain_R[0] + domain_R[1])/2

            if plot_dim != 1:
                ax = axes[i]
            else:
                ax = axes

            ax.plot(R, temperature, linewidth=0, marker='o')
            ax.plot(np.exp(z), np.exp(chebval(x, coeffs)))
            ax.set_ylabel('T (K)'); ax.set_xlabel('{} ($\Omega$)'.format(therm))
            ax.set_title('{} T'.format(field));

        fig.tight_layout()

    return coeffs_dict


def plotCheby(Rmax, Rmin, coeffs):
    """
    To be written.

    stuff
    """
    x = np.linspace(-1, 1, 100)
    domain_logR = np.log(np.array([Rmin, Rmax]))
    z = (x*(domain_logR[1] - domain_logR[0]) + domain_logR[0] + domain_logR[1])/2

    fig, ax = plt.subplots()
    ax.plot(np.exp(z), np.exp(chebval(x, coeffs)))
    fig.tight_layout()


def interpChebyPolySpline1D(coeffs_dict, cheby_deg, fields, thermometers, kind='cubic'):
    """
    Interpolate Chebyshev coefficients for each thermometer between all fields.

    Also plots the interpolation results

    Parameters
        coeffs_dict:  dictionary of lists of chebyshev coeff arrays (values) for each thermometer (keys)
                      output of ChebyCalibFields
        cheby_deg:    degree of the Chebyshev polynomials, used to determine the number of subplots
        fields:       the fields between which the interpolation is occuring
        thermometers: names of the thermometers in the dataframe
        kind='cubic': specifies the kind of spline interpolation

    Returns
        cheby_interp_dict: keys=thermometers, values=list of interpolation functions for each Chebyshev
                           coefficient
    """
    cheby_interp_dict = {}

    x_eval = np.linspace(fields[0], fields[-1], 100)
    plot_dim = int(np.ceil(np.sqrt(cheby_deg+1)))

    for therm in thermometers:
        fig, axes = plt.subplots(plot_dim, plot_dim)
        axes = axes.flatten()
        therm_indicator = therm.split('_')[1]

        coeffs_list = coeffs_dict[therm]

        cheby_interp_dict[therm] = []

        for i in range(cheby_deg+1):
            ax = axes[i]
            ax.set_title('c{0}-{1}'.format(i, therm_indicator)); ax.set_xlabel('Field (T)')

            c_i = np.array([c[i] for c in coeffs_list])
            f_interp = interp1d(fields, c_i, kind=kind)

            cheby_interp_dict[therm].append(f_interp)

            ax.plot(fields, c_i, marker='o', linewidth=0, label='c{0}-{1}'.format(i, therm_indicator))
            ax.plot(x_eval, f_interp(x_eval))

        fig.tight_layout();

    return cheby_interp_dict


def thermometerMR(df_avg, thermometers):
    """
    Calculate the magnetoresistance of each thermometer and plotting it.

    Parameters
        df_avg: dataframe w/ averaged R/T values
        thermometers: list of thermometer column names
    """
    fig, axes = plt.subplots(len(thermometers), 1, **kwargs)
    df_mr = df_avg[["Field (T)", "Temp_round (K)"] + thermometers]
    # df_mr[thermometers] = df_mr.groupby('Temp_round (K)')[thermometers].transform(lambda x: (x-x.max())/x.max())

    # print(df_mr.set_index('Field (T)').groupby('Temp_round (K)').transform(lambda x: x-x[x['Field (T)']==0]))

    cmap = plt.get_cmap("magma")

    gb_mr = df_mr.set_index(["Temp_round (K)", "Field (T)"]).groupby("Temp_round (K)")

    keys = gb_mr.groups.keys()
    num_colors = len(keys)

    for i, therm in enumerate(thermometers):
        axes[i].set_prop_cycle(
            color=[cmap(1.0 * k / num_colors) for k in range(num_colors)]
        )
        axes[i].set_title(therm)
        axes[i].set_ylabel("$\Delta R / R$")
        for key, df_group in gb_mr:
            df_group.loc[key].transform(lambda x: (x - x.loc[0]) / x.loc[0])[
                therm
            ].plot(ax=axes[i], marker="o", label=key)
        if i == 1:
            axes[i].legend(
                ncol=2, bbox_to_anchor=(1.01, 0.5), loc="center left", fontsize="small"
            )

    fig.tight_layout()


def extractTemp(R, r_max, r_min, cheby_coefs):
    """
    Extract temperature from calibration data and Chebyshev coefficients.

    Specifically, calibration data (i.e. 1 field) and a single
    set of Chebyshev coefficients.

    To be passed to dataframe via transform function

    Parameters
        R:           resistance values
        r_max:       max resistance
        r_min:       min resistance
        cheby_coefs: array of chebyshev coefficients

    Returns
        T: temperature calculated from chebyshev fit
    """
    logR = np.log(R)
    domain = np.log([r_min, r_max])

    X = ((logR - domain[0]) - (domain[1] - logR))/(domain[1]-domain[0])
    T = np.exp(chebval(X, cheby_coefs))
    return T


def extractTempSpline(R, B, f_interp_max, f_interp_min, cheby_interp_funcs):
    """
    Extract temperature using interpolation fncts for each Cheby coefficient.

    Use this to get temperature at many different fields
    To be passed to dataframe via transform function

    Parameters
        R:                          resistance values
        B:                          magnetic field at which to evaluate the interpolation functions
        f_interp_max, f_interp_min: interpolation functions for the max/min resistance values
        cheby_interp_funcs:         list of interpolation functions for each Chebyshev coefficient;
                                    this is cheby_interp_dict (output of interpChebyPolySpline1D)
                                    evaluated for a particular thermometer

    Returns
        T: calculated temperature from the interpolated coefficients
    """
    cheby_coefs_eval = np.array([f(B) for f in cheby_interp_funcs])
    logR = np.log(R)
    r_min = f_interp_min(B)
    r_max = f_interp_max(B)
    domain = np.log([r_min, r_max])

    X = ((logR - domain[0]) - (domain[1] - logR))/(domain[1]-domain[0])
    T = np.exp(chebval(X, cheby_coefs_eval))
    return T


def compareCalibs(df_exgas, df_hivac, thermometers):
    """
    Compare the resistance values of thermometers in exchange gas and hivac.

    Plots the exchange gas and hivac calibrations for each thermometer on their
    own subplots

    Parameters:
        df_exgas: exchange gas calibration df, unmasked
        df_hivac: hivac calibration df, unmasked
        thermometers: column names for thermometers
    """
    fig, axes = plt.subplots(len(thermometers), 1, sharex=True)

    for ax, therm in zip(axes, thermometers):
        df_exgas[df_exgas['mask'] == 1].plot(ax=ax, x='Temperature (K)', y=therm, label='{} exgas'.format(therm))
        ax.set_ylabel('Resistance ($\Omega$)')

    for ax, therm in zip(axes, thermometers):
        df_hivac[df_hivac['mask'] == 1].plot(ax=ax, x='Temperature (K)', y=therm, label='{} hivac'.format(therm))

    fig.tight_layout()


def percentDev(df_exgas, df_hivac, thermometers, domain_dict, coeffs_dict, index, **kwargs):
    """
    Calculate and plot the percent deviation between hivac and exgas data.

    Plots resistance and calculated temperature between the exchange gas and
    hivac calibrations

    Parameters:
        df_exgas:     exchange gas calibration df, unmasked and unrounded
        df_hivac:     hivac calibration df, unmasked and unrounded
        thermometers: column names for thermometers
        domain_dict:  dictionary of domains for each thermoeter
                      key: thermometer, value: [r_max,r_min]
        coeffs_dict:  dictionary of chebyshev coefficients to pass to extractTemp
                      key: thermometer, value: [coeffs_array1, coeffs_array2...]
        index:        index for list of coeffs arrays
    """
    # For the raw resistance
    fig, axes = plt.subplots(len(thermometers), 1, sharex=True, **kwargs)
    groups = ['Field (T)', 'Temp_round (K)']

    df_hivac = df_hivac.round({'Field (T)':2})[(df_hivac['mask'] == 1)]
    df_hivac['Temp_round (K)'] = df_hivac['Temperature (K)'].round(1)
    df_hivac = df_hivac.groupby(groups).filter(lambda x: len(x)>50).groupby(groups).mean().reset_index()

    df_exgas = df_exgas.round({'Field (T)':2})[(df_exgas['mask'] == 1)]
    df_exgas['Temp_round (K)'] = df_exgas['Temperature (K)'].round(1)
    df_exgas = df_exgas.groupby(groups).filter(lambda x: len(x)>50).groupby(groups).mean().reset_index()

    for therm, ax in zip(thermometers, axes):
        r_exgas = df_exgas[therm].values
        r_hivac = df_hivac[therm].values
        t = df_hivac['Temperature (K)'].values
        ax.plot(t, (r_hivac-r_exgas)*100/r_exgas)
        ax.set_title(therm)
        ax.set_ylabel('($R_{hivac} - R_{exgas})/R_{exgas}$ (%)')
        ax.set_xlim([2, 11])

    axes[2].set_xlabel('Temperature (K)')
    fig.tight_layout()

    # For the thermometers
    for therm in thermometers:
        T_string = 'T_' + therm.split('_')[1]
        df_exgas[T_string] = df_exgas.groupby(['Field (T)'])[therm].transform(lambda x: extractTemp(x.values, domain_dict[therm][0], domain_dict[therm][1], coeffs_dict[therm][index]))
        df_hivac[T_string] = df_hivac.groupby(['Field (T)'])[therm].transform(lambda x: extractTemp(x.values, domain_dict[therm][0], domain_dict[therm][1], coeffs_dict[therm][index]))

    fig, axes = plt.subplots(3, 1, sharex=True, **kwargs)
    temperatures = ['T_H', 'T_C1', 'T_C2']
    for temp, ax in zip(temperatures, axes):
        t_exgas = df_exgas[temp].values
        t_hivac = df_hivac[temp].values
        t = df_hivac['Temperature (K)'].values
        ax.plot(t, (t_hivac-t_exgas)*100/t_exgas)
        ax.set_title(temp)
        ax.set_ylabel('($T_{hivac} - T_{exgas})/T_{exgas}$ (%)')
        ax.set_xlim([2, 11])

    axes[2].set_xlabel('Temperature (K)')

    fig.tight_layout()

