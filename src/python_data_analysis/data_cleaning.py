import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def getPrefixMultiplier(num):
    if type(num) == str:
        return '', 1
    if num == 0:
        return '', 1
    elif num < 0:
        return '', 1
    else:
        power = np.floor(np.log10(num))
        power = power-power % 3
        if power == -12:
            return 'p', 1e12
        if power == -9:
            return 'n', 1e9
        if power == -6:
            return 'u', 1e6
        if power == -3:
            return 'm', 1e3
        if power == 0:
            return '', 1
        if power == 3:
            return 'k', 1e-3
        if power == 6:
            return 'M', 1e-6


def getUnits(param_string):
    split = param_string.split()
    if 'AC' in param_string:
        return 'AC'
    else:
        if split[1] == '(Hz)':
            return 'Hz'
        return split[1][1]


def iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


def sepZFC_FC(df):
    x = df['Temperature (K)'].values
    idx = np.argmax(np.abs(np.diff(x)))
    df_ZFC = df[:idx+1]
    df_FC = df[idx+1:]

    sep_df = {'ZFC': df_ZFC, 'FC': df_FC}
    return sep_df


def initialize(time_series):
    time_series -= time_series.min()
    return

# Hampel filter to remove outliers


def hamp_filt(vals_orig, k=21, t0=3):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 21 is equal to 10 on either side of value)
    '''

    # Make copy so original not edited
    vals = vals_orig.copy()

    # Hampel Filter
    L = 1.4826
    rolling_median = vals.rolling(window=k, center=True).median()
    def MAD(x): return np.median(np.abs(x - np.median(x)))
    rolling_MAD = vals.rolling(window=k, center=True).apply(MAD)
    threshold = t0 * L * rolling_MAD
    difference = np.abs(vals - rolling_median)

    '''
    Perhaps a condition should be added here in the case that the threshold value
    is 0.0; maybe do not mark as outlier. MAD may be 0.0 without the original values
    being equal. See differences between MAD vs SDV.
    '''

    outlier_idx = difference > threshold
    vals[outlier_idx] = np.nan
    return(vals)


def excludeData(df, cols, mask_col, lims):
    for lim in lims:
        for col in cols:
            df[col].mask(df[mask_col].between(lim[0], lim[1]), inplace=True)

    return df


def makeTitle(key, units, dev, ylabel):
    constr_title = ''
    if not iterable(key):
        constr_title = '({}{}) '.format(key, units[0])
    else:
        constr_title = '('
        for i, constr, unit in zip(range(len(key)), key, units):
            if unit == 'AC':
                constr_title += '{0}{1}'.format(constr, unit)
            else:
                constr_title += '{0:g}{1}'.format(constr * getPrefixMultiplier(constr)[
                                                  1], getPrefixMultiplier(constr)[0] + unit)
            if i != len(key)-1:
                constr_title += ','
        constr_title += ') '

    if 'X' in dev[0] or 'Y' in dev[0] or 'R' in dev[0]:
        ylabel = 'V (V)'
    elif 'Theta' in dev[0] or 'theta' in dev[0]:
        ylabel = 'Phase (deg)'
    elif 'res' in dev[0]:
        ylabel = 'R ($\Omega$)'

    return constr_title, ylabel


def separateSweeps(input_df, sweep_dir='up', sweep_select='up', sep=True, keep_both=False):
    '''
    If you have overlapping data, where one set is an up sweep and the other is a down sweep,
    this function will help separate that
    '''
    names = input_df.index.names
    input_df = input_df.reset_index(level=names[:-1]).drop(columns=names[:-1])

    if sep:
        if sweep_dir == 'up':
            sep_index = input_df.index.max()
            df_up = input_df[:sep_index]
            df_down = input_df[sep_index:]
        elif sweep_dir == 'down':
            sep_index = input_df.index.min()

            df_down = input_df[:sep_index]
            df_up = input_df[sep_index:]
        else:
            raise ValueError
        sep_df = {'up': df_up, 'down': df_down}
        if not keep_both:
            return sep_df[sweep_select]
        else:
            df_up['sweep_dir'] = 'up'
            df_down['sweep_dir'] = 'down'
            return pd.concat([df_up, df_down])
    else:
        output_df = input_df.sort_index()
        return output_df


def average(input_df, group_titles, thresh=50, tc=30):
    '''
    This function average data at a fixed number of data points (given by the innermost_constr axis)
    For instance, if sitting at certain angles collecting data for some time, this will group
    the data by angle and average. It will do this for time > tc.
    '''

    innermost_constr = group_titles[-1]

    input_df = input_df.groupby(group_titles).filter(
        lambda x: len(x.index) > thresh)

    input_df['Time (ms)'] = input_df.groupby(
        group_titles).transform(lambda x: x-x.min())['Time (ms)']

    output_df = input_df[input_df['Time (ms)']
                         > tc*1000].groupby(group_titles).mean()

    return output_df


def get_group_levels(input_df):
    levels = []
    for i in range(len(input_df.index.names)-1):
        levels.append(input_df.index.unique(i).values[0])

    return levels


def subtractOffset(input_df, devices):
    '''
    This is for eliminating the offset for the data by subtracting off the minimum indexed value (absolute)
    e.g. the data at temperature=0 or field=0
    '''
    levels = get_group_levels(input_df)

    idx = np.argmin(np.abs(input_df.index.get_level_values(-1).values))

    input_df[devices] = input_df[devices].transform(
        lambda x: x-input_df[devices].loc[(*levels,)].iloc[idx])

    output_df = input_df

    return output_df


def percentOffset(input_df, devices):
    '''
    This is for plotting Δ(quantity)/quantity
    '''
    levels = get_group_levels(input_df)

    idx = np.argmin(np.abs(input_df.index.get_level_values(-1).values))

    input_df[devices] = input_df[devices].transform(lambda x: (
        x-input_df[devices].loc[(*levels,)].iloc[idx])/input_df[devices].loc[(*levels,)].iloc[idx])

    output_df = input_df

    return output_df


def divideOffset(input_df, devices):
    '''
    This is for dividing the data by the minimum indexed value (absolute)
    e.g. the data at temperature=0 or field=0
    '''
    levels = get_group_levels(input_df)

    idx = np.argmin(np.abs(input_df.index.get_level_values(-1).values))

    input_df[devices] = input_df[devices].transform(
        lambda x: x/input_df[devices].loc[(*levels,)].iloc[idx])

    output_df = input_df

    return output_df


def subtractXmax(input_df, devices):
    '''
    This is for subtracting the data by the maximum indexed value (absolute) (e.g. where field = 9T)
    '''
    levels = get_group_levels(input_df)

    idx = np.argmax(np.abs(input_df.index.get_level_values(-1).values))

    input_df[devices] = input_df[devices].transform(
        lambda x: x-input_df[devices].loc[(*levels,)].iloc[idx])

    output_df = input_df

    return output_df


def subtractYmin(input_df, devices):
    '''
    This is for subtracting the data by the minimum in the set (not absolute) 
    '''
    input_df[devices] = input_df[devices].transform(lambda x: x - x.min())
    return input_df


def subtractAverage(input_df, devices):
    input_df[devices] = input_df[devices].transform(lambda x: x - x.mean())
    return input_df


def offsetData(input_gb, offset, devices):
    '''
    This is for offsetting the different data curves
    Need to pipe this through groupby since knowledge of all the groups is necessary; 
    each group gets different offset
    '''
    dfs = []
    for i, group in enumerate(input_gb.groups.keys()):
        df_group = input_gb.get_group(group)
        df_group[devices] = df_group[devices].apply(lambda x: x + offset*i)
        dfs.append(df_group)
    return pd.concat(dfs)


def subtractOffsetg(input_gb, xlevel, devices):
    '''
    This is for eliminating the offset for the data by subtracting off the minimum indexed value (absolute)
    e.g. the data at temperature=0 or field=0
    '''

    dfs = []
    for key, group in input_gb:
        idx = group[xlevel].idxmin()
        group[devices] = group[devices] - group[devices].loc[idx]
        dfs.append(group)

    return pd.concat(dfs)


def slice_inner(input_df, start=None, end=None):
    '''
    This function slices a dataframe on the highest level (i.e. innermost) index
    '''
    slices = []
    for i in range(len(input_df.index.names)-1):
        slices.append(slice(None, None, None))

    return input_df.loc[(*slices, slice(start, end)), :]


def unitize_groups(input_df, group_titles):
    input_df.reset_index(group_titles, inplace=True)
    for title in group_titles[:-1]:
        if '(' not in title:
            units = ''
        else:
            units = title.split()[1].split('(')[1][0]
        input_df[title] = input_df[title].transform(lambda x: '{0}{1}'.format(
            x*getPrefixMultiplier(x)[1], getPrefixMultiplier(x)[0]+units))
    output_df = input_df.set_index(group_titles)
    return output_df


def unitize_title_prefix(group_key, group_titles):
    units = []
    if not iterable(group_key):
        group_key = [group_key]
    for title in group_titles:
        if '(' not in title:
            units.append('')
        else:
            units.append(title.split()[1].split('(')[1][0])
    group_str = ",".join(
        f"{x*getPrefixMultiplier(x)[1]}{getPrefixMultiplier(x)[0]}{y}" for x, y in zip(group_key, units))
    group_str = '(' + group_str + ')'
    return group_str


def unitize_label(label, label_name):
    if '(' not in label_name:
        unit = ''
    else:
        unit = label_name.split()[1].split('(')[1][0]
    label_str = '{0}{1}'.format(
        label*getPrefixMultiplier(label)[1], getPrefixMultiplier(label)[0]+unit)
    return label_str


def plotdf(input_gb, device_info, group_titles, cmap=plt.get_cmap('magma'),
           plot_sweeps=False, sharex=True, **kwargs):
    for gb in input_gb:
        fig, axes = plt.subplots(len(device_info), 1, sharex=sharex)
        key = gb[0]
        df_group = gb[1]
        if not iterable(axes):
            axes = [axes]
        for i, dev, ax in zip(range(len(device_info)), device_info, axes):
            subdf_group = df_group.loc[key]
            title = unitize_title_prefix(key, group_titles[:-2]) + ' ' + dev[1]
            num_colors = subdf_group.index.get_level_values(0).unique().size
            ax.set(title=title, ylabel=dev[2])

            if not plot_sweeps:
                ax.set_prop_cycle(color=[cmap(1.*k/num_colors)
                                  for k in range(num_colors)])

                # for idx in df_group.index.get_level_values(0).unique():
                #     sub_df_group = df_group.loc[idx]
                #     sub_df_group[dev[0]].plot(ax=ax, label=idx)
                # if i == 0:
                #     ax.legend(ncol=3, fontsize='small')

                labels = subdf_group.index.get_level_values(0).unique()
                label_name = subdf_group.index.names[0]
                labels = list(
                    map(lambda x: unitize_label(x, label_name), labels))
                (subdf_group.reset_index(label_name)
                            .groupby(label_name)[dev[0]]
                            .plot(ax=ax, x=subdf_group.index.get_level_values(1), **kwargs))
                if i == 0:
                    ax.legend(labels, loc='best', ncol=3, fontsize='small')

            else:
                ax.set_prop_cycle(color=[cmap(1.*k/num_colors)
                                  for k in range(num_colors)])

                # for idx in df_group.index.get_level_values(0).unique():
                #     sub_df_group = df_group.loc[idx]
                #     sub_df_group[dev[0]].plot(ax=ax, label=idx)
                # if i == 0:
                #     ax.legend(ncol=3, fontsize='small')

                labels = subdf_group.index.get_level_values(0).unique()
                label_name = subdf_group.index.names[0]
                labels = list(
                    map(lambda x: unitize_label(x, label_name), labels))
                (subdf_group.reset_index(label_name)
                            .groupby(label_name)[dev[0]]
                            .plot(ax=ax, x=subdf_group.index.get_level_values(1), **kwargs))
                if i == 0:
                    ax.legend(labels, loc='best', ncol=3, fontsize='small')

            scaleAxes(ax)

    fig.tight_layout()


def plotdf_noGroup(input_df, device_info, group_titles,
                   cmap=plt.get_cmap('magma'), scale=True, **kwargs):
    fig, axes = plt.subplots(len(device_info), 1)
    if not iterable(axes):
        axes = [axes]
    for i, dev, ax in zip(range(len(device_info)), device_info, axes):
        title = dev[1]
        num_colors = input_df.index.get_level_values(0).unique().size
        ax.set(title=title, ylabel=dev[2])
        ax.set_prop_cycle(color=[cmap(1.*k/num_colors)
                          for k in range(num_colors)])

        labels = input_df.index.get_level_values(0).unique()
        label_name = input_df.index.names[0]
        labels = list(map(lambda x: unitize_label(x, label_name), labels))
        (input_df.reset_index(label_name)
                 .groupby(label_name)[dev[0]]
                 .plot(ax=ax, x=input_df.index.get_level_values(1), marker='o', **kwargs))
        if i == 0:
            ax.legend(labels, loc='best', ncol=3, fontsize='small')

        if scale:
            scaleAxes(ax)

    fig.tight_layout()


def makePlot(fig, ax, title='', xlabel='', ylabel='',
             cmap=plt.get_cmap('magma'),num_colors=1, xlim=None, ylim=None):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_prop_cycle(color=[cmap(1.*k/num_colors) for k in range(num_colors)])
    fig.tight_layout


def scaleAxes(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = max(abs(xmin), abs(xmax))
    ymax = max(abs(ymin), abs(ymax))
    xprefix, xmult = getPrefixMultiplier(xmax)
    yprefix, ymult = getPrefixMultiplier(ymax)
    if '(' in ax.get_xlabel():
        paren_prefix = '({}'.format(xprefix)
        new_label = paren_prefix.join(ax.get_xlabel().split('('))
        ax.set_xlabel(new_label)
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*xmult))
        ax.xaxis.set_major_formatter(ticks_x)
    if '(' in ax.get_ylabel():
        paren_prefix = '({}'.format(yprefix)
        new_label = paren_prefix.join(ax.get_ylabel().split('('))
        ax.set_ylabel(new_label)
        ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y*ymult))
        ax.yaxis.set_major_formatter(ticks_y)


def test():
    print('hello')
