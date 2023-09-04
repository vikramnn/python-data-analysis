import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev, chebval
from numpy.polynomial.polynomial import polyfit, polyval
from scipy.interpolate import interp1d

def plotCalib(df_calib, thermometers):
    #thermometers is a list of columns in the dataframe that need to be calibrated
    fig, axes = plt.subplots(len(thermometers),1)
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

def minMaxInterp(df_interp, poly_order, fields):
    
    fig, ((axp_mx, axm_mx), (axp_mn, axm_mn)) = plt.subplots(2,2)

    df_interp.groupby('Field (T)').max().plot(ax=axp_mx, y='R+', linewidth=0, marker='o')
    df_interp.groupby('Field (T)').max().plot(ax=axm_mx, y='R-', linewidth=0, marker='o')
    df_interp.groupby('Field (T)').min().plot(ax=axp_mn, y='R+', linewidth=0, marker='o')
    df_interp.groupby('Field (T)').min().plot(ax=axm_mn, y='R-', linewidth=0, marker='o')

    rp_max = df_interp.groupby('Field (T)').max()['R+'].values
    rp_min = df_interp.groupby('Field (T)').min()['R+'].values
    rm_max = df_interp.groupby('Field (T)').max()['R-'].values
    rm_min = df_interp.groupby('Field (T)').min()['R-'].values

    rp_max_poly = polyfit(fields, rp_max, poly_order)
    fit = polyval(np.linspace(fields[0], fields[-1], 100), rp_max_poly)
    axp_mx.plot(np.linspace(fields[0], fields[-1], 100), fit)

    rp_min_poly = polyfit(fields, rp_min, poly_order)
    fit = polyval(np.linspace(fields[0], fields[-1], 100), rp_min_poly)
    axp_mn.plot(np.linspace(fields[0], fields[-1], 100), fit)

    rm_max_poly = polyfit(fields, rm_max, poly_order)
    fit = polyval(np.linspace(fields[0], fields[-1], 100), rm_max_poly)
    axm_mx.plot(np.linspace(fields[0], fields[-1], 100), fit)

    rm_min_poly = polyfit(fields, rm_min, poly_order)
    fit = polyval(np.linspace(fields[0], fields[-1], 100), rm_min_poly)
    axm_mn.plot(np.linspace(fields[0], fields[-1], 100), fit)
    
    fig.tight_layout()
    
    return rp_max_poly, rp_min_poly, rm_max_poly, rm_min_poly

def minMax(df_interp, thermometers):
    domain_dict = {}

    for therm in thermometers:
        domain_dict[therm] = [df_interp[therm].max(), df_interp[therm].min()]

    return domain_dict

def minMaxInterpSpline(df_interp, fields, thermometers, kind='cubic'):
    fig, axes = plt.subplots(len(thermometers),2)
    x_eval = np.linspace(fields[0], fields[-1], 100)

    domain_interp_dict = {}

    for i,therm in enumerate(thermometers):
        df_interp.groupby('Field (T)').max()[therm].plot(ax=axes[i,0], y=therm, linewidth=0, marker='o')
        df_interp.groupby('Field (T)').min()[therm].plot(ax=axes[i,1], y=therm, linewidth=0, marker='o')

        rmax = df_interp.groupby('Field (T)').max()[therm].values
        rmin = df_interp.groupby('Field (T)').min()[therm].values

        if len(fields) > 1:

            f_interp_max = interp1d(fields, rmax, kind=kind)
            axes[i,0].plot(x_eval, f_interp_max(x_eval))
        
            f_interp_min = interp1d(fields, rmin, kind=kind)
            axes[i,1].plot(x_eval, f_interp_min(x_eval))

            domain_interp_dict[therm] = [f_interp_max, f_interp_min]

    
    fig.tight_layout()
    
    return domain_interp_dict

def chebyCalibFields(df_avg, cheby_deg, fields, thermometers):

    coeffs_dict = {}
    
    plot_dim = int(np.ceil(np.sqrt(len(fields))))

    for therm in thermometers:
        coeffs_dict[therm] = []
        
        fig, axes = plt.subplots(plot_dim, plot_dim)

        if plot_dim != 1:
            axes = axes.flatten()

        for i,field in enumerate(fields):
            df_slice = df_avg[df_avg['Field (T)'] == field]
            R = df_slice[therm].values
            temperature = df_slice['Temperature (K)'].values

            domain_R = np.log(np.array([R.min(), R.max()]))

            x = np.linspace(-1,1,100)

            #Fitting
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

def interpChebyPoly(cp, cm, poly_order, fields):
    cheby_poly_coefs_p = np.array([])
    cheby_poly_coefs_m = np.array([])
    plot_dim = int(np.ceil(np.sqrt(cheby_deg)))
    fig_p, axes_p = plt.subplots(plot_dim, plot_dim)
    fig_m, axes_m = plt.subplots(plot_dim, plot_dim)
    axes_p = axes_p.flatten(); axes_m = axes_m.flatten()

    for i in range(cheby_deg+1):
        ax = axes_p[i]
        ax.set_title('c{}+'.format(i)); ax.set_xlabel('Field (T)')

        cp_i = np.array([c[i] for c in cp])
        poly_coefs_p = polyfit(fields, cp_i, poly_order)
        fit = polyval(np.linspace(fields[0], fields[-1], 100), poly_coefs_p)
        cheby_poly_coefs_p = np.append(cheby_poly_coefs_p, poly_coefs_p)

        ax.plot(fields, cp_i, marker='o', linewidth=0, label='c{}'.format(i))
        ax.plot(np.linspace(fields[0], fields[-1], 100), fit)
        fig_p.tight_layout()

        ax = axes_m[i]
        ax.set_title('c{}-'.format(i)); ax.set_xlabel('Field (T)')

        cm_i = np.array([c[i] for c in cm])
        poly_coefs_m = polyfit(fields, cm_i, poly_order)
        fit = polyval(np.linspace(fields[0], fields[-1], 100), poly_coefs_m)
        cheby_poly_coefs_m = np.append(cheby_poly_coefs_m, poly_coefs_m)

        ax.plot(fields, cm_i, marker='o', linewidth=0, label='c{}'.format(i))
        ax.plot(np.linspace(fields[0], fields[-1], 100), fit)
        fig_m.tight_layout()

    cheby_poly_coefs_p = cheby_poly_coefs_p.reshape(cheby_deg+1,poly_order+1).T
    cheby_poly_coefs_m = cheby_poly_coefs_m.reshape(cheby_deg+1,poly_order+1).T
    
    return cheby_poly_coefs_p, cheby_poly_coefs_m

def interpChebyPolySpline1D(coeffs_dict, cheby_deg, fields, thermometers, kind='cubic'):
    #lists of interpolations functions for each cheby coeff
    #Functions are evaluated at a particular field

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
            ax.set_title('c{0}-{1}'.format(i,therm_indicator)); ax.set_xlabel('Field (T)')

            c_i = np.array([c[i] for c in coeffs_list])
            f_interp = interp1d(fields, c_i, kind=kind)

            cheyb_interp_dict[therm].append(f_interp)

            ax.plot(fields, c_i, marker='o', linewidth=0, label='c{0}-{1}'.format(i, therm_indicator))
            ax.plot(x_eval, f_interp(x_eval))

        fig.tight_layout();
    
    return cheby_interp_dict

def thermometerMR(df_avg, thermometers):
    fig, axes = plt.subplots(len(thermometers),1)
    df_mr = df_avg[['Field (T)', 'Temp_round (K)'] + thermometers]
    #df_mr[thermometers] = df_mr.groupby('Temp_round (K)')[thermometers].transform(lambda x: (x-x.max())/x.max())

    #print(df_mr.set_index('Field (T)').groupby('Temp_round (K)').transform(lambda x: x-x[x['Field (T)']==0]))
        
    cmap = plt.get_cmap('magma')

    gb_mr = df_mr.set_index(['Temp_round (K)', 'Field (T)']).groupby('Temp_round (K)')
    
    keys = gb_mr.groups.keys(); num_colors = len(keys)

    for i, therm in enumerate(thermometers):
        axes[i].set_prop_cycle(color=[cmap(1.*k/num_colors) for k in range(num_colors)])
        axes[i].set_title(therm)
        axes[i].set_ylabel('$\Delta R / R$')
        for key, df_group in gb_mr:
            df_group.loc[key].transform(lambda x: (x-x.loc[0])/x.loc[0])[therm].plot(ax=axes[i], marker='o', label=key)
        axes[i].legend(ncol=2, bbox_to_anchor=(1.01,0.5), loc='center left', fontsize='small')

    fig.tight_layout()

def extractTemp(R, r_max, r_min, cheby_coefs):
    logR = np.log(R)
    domain = np.log([r_min, r_max])
    
    X = ((logR - domain[0]) - (domain[1] - logR))/(domain[1]-domain[0])
    T = np.exp(chebval(X, cheby_coefs))
    return T

def extractTempSpline(R, B, f_interp_max, f_interp_min, cheby_interp_funcs):
    cheby_coefs_eval = np.array([f(B) for f in cheby_interp_funcs])
    logR = np.log(R)
    r_min = f_interp_min(B)
    r_max = f_interp_max(B)
    domain = np.log([r_min, r_max])
    
    X = ((logR - domain[0]) - (domain[1] - logR))/(domain[1]-domain[0])
    T = np.exp(chebval(X, cheby_coefs_eval))
    return T
    
def extractTempPoly(R, B, r_max_poly, r_min_poly, cheby_poly_coefs):
    '''
    B is fixed, R is an array, domain is [log10(R_lower), log10(R_upper)]
    cheby_poly_coefs is a matrix of coefficient values at different fields
    e.g. c0(B) = c00 + c01*B + ... +c0M * B**M
         ...
         [[c00 c10 ... cM0
            .           .
           c0M c1M ... cMM]]
    '''
    cheby_coefs_eval = polyval(B, cheby_poly_coefs) # This gives a row array of the cheby coefficients at field B
    logR = np.log10(R)
    r_min = polyval(B, r_min_poly)
    r_max = polyval(B, r_max_poly)
    domain = np.log10([r_min, r_max])
    
    X = ((logR - domain[0]) - (domain[1] - logR))/(domain[1]-domain[0])
    T = 10**chebval(X, cheby_coefs_eval)
    return T

