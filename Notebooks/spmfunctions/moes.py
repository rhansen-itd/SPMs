import pandas as pd
from .misc_tools import phase_status
import datetime as dt

def arrival_on_green(df, dets, ph, tt=0, det_code=82):
    '''
    Calculates arrival on green for a particular phase on a per-cycle basis.

    Args:
    - df (DataFrame): Input DataFrame containing traffic event data.
    - dets (list): List of advanced detectors for the phase.
    - ph (int): Phase.
    - tt (int): Default=0, offset in seconds to account for travel time from arrival detector to intersection.
    - det_code (int): Default=82, detector event code. 82 for actuation, 81 for de-actuation.

    Returns:
    - DataFrame: Result DataFrame containing arrival on green data.
    '''

    df_arr = df[(((df.Code.isin([1, 8, 10, 11])) & (df.ID == ph))) | ((df.Code == det_code) & (df.ID.isin(dets)))]
    df_res = df_arr.loc[:, ['Cycle_start', 'Cycle_len']].drop_duplicates().set_index('Cycle_start')

    if tt:
        df_arr.loc[(df_arr.Code == det_code), 'TS_start'] = \
            df_arr.loc[(df_arr.Code == det_code), 'TS_start'] + dt.timedelta(seconds=tt)
        df_arr = df_arr.sort_values('TS_start')

    df_arr = phase_status(df_arr, [ph])
    df_g = df_arr[df_arr.Code == 1]
    df_arr = df_arr[df_arr.Code == det_code]
    df_aog = df_arr[df_arr[f'Ph {ph} Status'].isin(['G'])]

    df_res.loc[:, 'G'] = df_g.loc[:, ['Cycle_start', 'Duration']].groupby('Cycle_start').sum()['Duration'].fillna(0)
    df_res.loc[:, 'G_pct'] = (df_res.G / df_res.Cycle_len)
    df_res.loc[:, 'Total'] = df_arr.groupby('Cycle_start').count()['ID'].fillna(0)
    df_res.loc[:, 'AoG'] = df_aog.groupby('Cycle_start').count()['ID'].fillna(0)
    df_res.loc[:, 'AoG_pct'] = (df_res.AoG / df_res.Total).fillna(0)
    df_res.loc[:, 'Phase'] = ph
    df_res = df_res.merge(df_arr.loc[:, ['Coord_plan', 'Cycle_start']].drop_duplicates(),
                            left_index=True, right_on='Cycle_start', how='left')

    return df_res


def bin_AOG(df_res, bin_len=60):
    '''
    Bins the arrival on green data.

    Args:
    - df_res (DataFrame): Arrival on green result DataFrame.
    - bin_len (int): Default=60, length in minutes of bins.

    Returns:
    - DataFrame: Binned arrival on green data.
    '''

    df_binres = pd.DataFrame()
    for ph in df_res.Phase.unique():
        for cp in df_res.Coord_plan.unique():
            df_bin = df_res.loc[(df_res.Phase == ph) & (df_res.Coord_plan == cp),
                                ['Cycle_start', 'Cycle_len', 'G', 'Total', 'AoG']]
            if len(df_bin) > 0:
                df_bin = df_bin.set_index('Cycle_start')
                df_bin = df_bin.resample(f'{bin_len}T').sum()
                df_bin.loc[:, 'AoG_pct'] = df_bin.AoG / df_bin.Total
                df_bin.loc[:, 'Phase'] = ph
                df_bin.loc[:, 'Coord_plan'] = cp
                df_bin.loc[:, 'G_pct'] = df_bin.G / df_bin.Cycle_len
                #df_binres = df_binres.append(df_bin)
                df_binres = pd.concat([df_binres, df_bin])

    df_binres = df_binres.fillna(0).loc[(df_binres.Total != 0),
                                        ['Phase', 'Coord_plan', 'G', 'G_pct', 'Total', 'AoG', 'AoG_pct']]
    df_binres.index.name = 'Time'
    df_binres.reset_index(inplace=True)
    df_binres = df_binres.sort_values(['Phase', 'Time'])

    return df_binres


def split_failures(df_raw, ph, det, t_red=5, thresh=0.8):
    '''
    Calculates green occupancy rate, red occupancy rate, and split failure for a particular phase.

    Args:
    - df_raw (DataFrame): Raw traffic data.
    - ph (int): Phase.
    - det (int): Detector.
    - t_red (int): Default=5, time after start of red clearance to calculate red occupancy.
    - thresh (float): Default=0.8, threshold of GOR and ROR to count as split failure.

    Returns:
    - DataFrame: Result DataFrame containing calculated values.
    '''

    # Create detector dataframe: 81 is deactivation, 82 is activation
    df_d = df_raw[(df_raw.Code.isin([81, 82])) & (df_raw.ID == det)].loc[:, ['TS_start', 'Code']].sort_values('TS_start')
    df_d.loc[:, 'Prev'] = df_d.Code.shift(1)
    df_d = df_d[df_d.Code != df_d.Prev]
    df_d.loc[:, 'Next'] = df_d.Code.shift(-1)
    df_d.loc[:, 'TS_end'] = df_d.TS_start.shift(-1)
    df_d = df_d.dropna(axis=0)
    df_d.drop(['Next', 'Prev'], axis=1, inplace=True)

    resdict = {}

    for i in [0, 1]:
        # First time through the loop, set up GOR, second ROR
        if i == 0:
            c_start = 1
            c_end = 10
            idx = ['G', 'G_occ', 'GOR']

            # Create GOR phase event dataframe. 1 is begin green, 10 is begin red clearance
            df_p = df_raw[(df_raw.Code.isin([1, 10])) & (df_raw.ID == ph)].loc[:, ['TS_start', 'Code']]
            df_p.loc[:, 'Next'] = df_p.Code.shift(-1)
            df_p.loc[:, 'TS_end'] = df_p.TS_start.shift(-1)
            df_p = df_p[df_p.Code != df_p.Next].dropna(axis=0)
            df_p.drop('Next', axis=1, inplace=True)

        else:
            c_start = 10
            c_end = 1010
            idx = ['R', 'R_occ', 'ROR']

            # Create ROR phase event dataframe. 10 is begin red clearance, 1010 is red clearance + t_red
            df_p = df_raw[(df_raw.Code == 10) & (df_raw.ID == ph)].loc[:, ['TS_start', 'Code']]
            df_p.loc[:, 'TS_end'] = df_p.loc[:, 'TS_start'] + dt.timedelta(seconds=t_red)
            df_1010 = df_p.copy()
            df_1010.loc[:, 'Code'] = 1010
            df_1010.loc[:, 'TS_start'] = df_1010.loc[:, 'TS_end']
            df_p = pd.concat([df_p, df_1010], ignore_index=True).sort_values('TS_start')

        # Combine detector and phase dataframes
        dfc = pd.concat([df_p, df_d], ignore_index=True).sort_values(['TS_start', 'Code'])

        # Forward fill detector status to phase events
        dfc.loc[(dfc.Code.isin([81, 82])), 'Det'] = dfc.loc[(dfc.Code.isin([81, 82])), 'Code']
        dfc = dfc.ffill()
        dfc.loc[:, 'Next'] = dfc.Code.shift(-1)
        dfc.loc[:, 'TS_next'] = dfc.TS_start.shift(-1)
        dfc.loc[:, 'Prev'] = dfc.Code.shift(1)

        # Change end time for detection events to end of green event
        dfc.loc[((dfc.Code == 82) & (dfc.Next == c_end)), 'TS_end'] = \
            dfc[(dfc.Code == 82) & (dfc.Next == c_end)].loc[:, 'TS_next']

        # Add 82's for all cycles with no detection events but start with detection active
        df_ag = dfc[(dfc.Code == c_start) & (dfc.Next == c_end) & (dfc.Det == 82)]
        df_ag.loc[:, 'Code'] = 82
        dfc = dfc.append(df_ag, ignore_index=True).sort_values(['TS_start', 'Code'])
        dfc.loc[:, 'Next'] = dfc.Code.shift(-1)
        dfc.loc[:, 'TS_next'] = dfc.TS_start.shift(-1)
        dfc.loc[:, 'Prev'] = dfc.Code.shift(1)

        # Add 82's at the beginning of each green if next event is deactivation
        df_gstart = dfc[(dfc.Code == c_start) & (dfc.Next == 81)]
        df_gstart.loc[:, 'TS_end'] = df_gstart.TS_next
        df_gstart.loc[:, 'Code'] = 82
        dfc = dfc.append(df_gstart).sort_values(['TS_start', 'Code'])

        # Add column to group each green by
        dfc.loc[(dfc.Code == c_start), 'Period'] = dfc.loc[dfc.Code == c_start, 'TS_start']
        dfc.loc[(dfc.Code == c_end), 'Period'] = 0
        dfc = dfc.ffill().dropna(axis=0)

        # Drop end green and end detection events
        dfc = dfc[(dfc.Code.isin([c_start, 82])) & (dfc.Period != 0)]

        # Calculate duration
        dfc.loc[:, 'Duration'] = (dfc.TS_end - dfc.TS_start).dt.total_seconds()

        dfres = pd.DataFrame()

        for p, gb in dfc.groupby('Period'):
            if i == 0:
                nm = gb.iloc[0].TS_end
            else:
                nm = p

            srs = pd.Series(name=nm, index=idx)
            d = gb.iloc[0].Duration
            occ = gb[gb.Code == 82].Duration.sum()

            srs.iloc[0] = d
            srs.iloc[1] = min(occ, d)
            srs.iloc[2] = min(occ, d) / d

            dfres = dfres.append(srs)

        resdict[i] = dfres.copy()

    resdict[0].index.name = 'Time'
    resdict[1].index.name = 'Time'
    df_res = pd.merge(resdict[0], resdict[1], on='Time')
    df_res.loc[:, 'Phase'] = ph
    df_res.loc[:, 'Detector'] = det
    df_res.loc[:, 'Split Fail'] = 1 * ((df_res.GOR > thresh) & (df_res.ROR > thresh))
    df_cp = df_raw.loc[:, ['Coord_plan', 'TS_start']].drop_duplicates()
    df_cp = df_cp.set_index('TS_start')
    df_cp.index.name = 'Time'
    df_res = pd.merge(df_res, df_cp, on='Time')

    return df_res.loc[:, ['Phase', 'Detector', 'Coord_plan',
                          'G', 'G_occ', 'GOR',
                          'R', 'R_occ', 'ROR', 'Split Fail']]


def bin_SF(df_res, bin_len=60):
    '''
    Bins the split failures data.

    Args:
    - df_res (DataFrame): Split failure result DataFrame.
    - bin_len (int): Default=60, length in minutes of bins.

    Returns:
    - DataFrame: Binned split failures data.
    '''

    df_binres = pd.DataFrame()
    for ph in df_res.Phase.unique():
        for cp in df_res.Coord_plan.unique():
            df_res.loc[:, 'N_cyc'] = 1
            df_bin = df_res.loc[(df_res.Phase == ph) & (df_res.Coord_plan == cp),
                                ['G', 'G_occ', 'R', 'R_occ', 'N_cyc',
                                 'Split Fail']].resample('%sT' % bin_len).sum()
            if len(df_bin) > 0:
                df_bin.loc[:, 'GOR'] = df_bin.G_occ / df_bin.G
                df_bin.loc[:, 'ROR'] = df_bin.R_occ / df_bin.R
                df_bin.loc[:, 'SF_pct'] = df_bin.loc[:, 'Split Fail'] / df_bin.loc[:, 'N_cyc']
                df_bin.loc[:, 'Phase'] = ph
                df_bin.loc[:, 'Coord_plan'] = cp
                df_binres = df_binres.append(df_bin)

    df_binres = df_binres.loc[:, ['Phase', 'Coord_plan', 'G_occ', 'GOR', 'R_occ', 'ROR',
                                  'N_cyc', 'Split Fail', 'SF_pct']]

    df_binres = df_binres.fillna(0).loc[(df_binres.N_cyc > 0),
                                        ['Phase', 'Coord_plan', 'G_occ', 'GOR', 'R_occ', 'ROR',
                                         'N_cyc', 'Split Fail', 'SF_pct']]
    df_binres.index.name = 'Time'
    df_binres.reset_index(inplace=True)
    df_binres = df_binres.sort_values(['Phase', 'Time']).loc[:, ['Time', 'Phase', 'Coord_plan',
                                                                  'G_occ', 'GOR', 'R_occ', 'ROR',
                                                                  'Split Fail', 'N_cyc', 'SF_pct']]

    return df_binres


def combine_aog_sf_tt(intx, bin_len=60, write_to_file=None, sfx='',
                      xl_mode='w', phases=[2, 4, 6, 8], return_by_cp=False,
                      atr=None, sf_sfx='', aog_sfx='', count_sfx=''):
    '''
    Combines AOG, SF, and TT data and performs analysis.

    Args:
    - intx (str): Name of the intersection.
    - bin_len (int): Default=60, length in minutes of bins.
    - write_to_file (str): Default=None, output format ('xl', 'csv', 'pk') or None to skip writing.
    - sfx (str): Default='', suffix for output filenames.
    - xl_mode (str): Default='w', mode for Excel file ('w' for write, 'a' for append).
    - phases (list): Default=[2, 4, 6, 8], list of phases to analyze.
    - return_by_cp (bool): Default=False, return dataframes by coordination plan.
    - atr (dict): Default=None, dictionary specifying additional volume count data.
    - sf_sfx (str): Default='', suffix for SF filenames.
    - aog_sfx (str): Default='', suffix for AOG filenames.
    - count_sfx (str): Default='', suffix for count filenames.

    Returns:
    - dict: Result dataframes.
    '''

    seg_key = pd.read_csv('./inrix/INRIX_key.csv')
    df_tt = pd.read_csv('./inrix/data_%s.csv' % bin_len)
    df_tt.loc[:, 'Date Time'] = pd.to_datetime(df_tt.loc[:, 'Date Time']).apply(
        lambda x: x.replace(tzinfo=None))
    resdict = {}

    ph_dir = {2: 'WB', 4: 'SB', 6: 'EB', 8: 'NB'}

    if bin_len == 60:
        f_end = 'hourly.csv'
    elif bin_len == 15:
        f_end = '15min.csv'

    # Split failure dataframe
    df_sf = pd.DataFrame()
    path_sf = './ACHD_Eagle-{}/Split Failures/'.format(intx)
    for f_sf in [f for f in os.listdir(path_sf) if f.endswith('{}{}'.format(sf_sfx, f_end))]:
        df_sf = df_sf.append(pd.read_csv(os.path.join(path_sf, f_sf)))
    df_sf.loc[:, 'Time'] = pd.to_datetime(df_sf.Time, infer_datetime_format=True)
    df_sfb = df_sf

    # AOG dataframe
    df_aog = pd.DataFrame()
    path_aog = './ACHD_Eagle-{}/AOG/'.format(intx)
    for f_aog in [f for f in os.listdir(path_aog) if f.endswith('{}{}'.format(aog_sfx, f_end))]:
        df_aog = df_aog.append(pd.read_csv(os.path.join(path_aog, f_aog)))
    df_aog.loc[:, 'Time'] = pd.to_datetime(df_aog.Time, infer_datetime_format=True)

    # Count dataframe
    df_ct = pd.DataFrame()
    path_ct = './ACHD_Eagle-{}/Counts/'.format(intx)
    for f_ct in [f for f in os.listdir(path_ct) if f.endswith('{}{}'.format(count_sfx, f_end))]:
        df_ct = df_ct.append(pd.read_csv(os.path.join(path_ct, f_ct)))
    df_ct.loc[:, 'Time'] = pd.to_datetime(df_ct.Time, infer_datetime_format=True)

    df_b = pd.merge(df_aog, df_sf.loc[:, ['Time', 'Phase', 'Coord_plan',
                                          'G_occ', 'GOR', 'R_occ', 'ROR', 'Split Fail', 'N_cyc', 'SF_pct']],
                   'outer', on=['Time', 'Phase', 'Coord_plan'])

    df_b = pd.merge(df_b, df_ct.loc[:, ['Time',
                                        'WBL', 'WBT', 'WBR',
                                        'SBL', 'SBT', 'SBR',
                                        'EBL', 'EBT', 'EBR',
                                        'NBL', 'NBT', 'NBR']])

    for ph in phases:

        seg_row = seg_key.loc[(seg_key.Intersection == intx) & (seg_key.Phase == ph), :]
        if len(seg_row) == 0:
            continue

        if atr and bin_len == 60:
            if ph in list(atr.keys()):
                df_atr = pd.read_csv('./ACHD_Eagle-{}/Counts/{}_long.csv'.format(intx, atr[ph][0]))
                df_atr.loc[:, 'Datetime'] = pd.to_datetime(df_atr.Datetime, infer_datetime_format=True)
                df_atr = df_atr.loc[(df_atr.Dir == atr[ph][1]), ['Datetime', 'Count']]
                df_atr.rename({'Count': 'Vol'}, axis=1, inplace=True)
                df_b = pd.merge(df_b, df_atr, left_on='Time', right_on='Datetime', how='left')

        seg = seg_row['Segment ID'].iloc[0]
        df_tt_int = df_tt.loc[(df_tt.loc[:, 'Segment ID'] == seg),
                              ['Date Time', 'Travel Time(Minutes)']]
        dfbcols = ['Time', 'Coord_plan', 'G', 'G_pct', 'Total',
                   'AoG', 'AoG_pct', 'G_occ', 'GOR', 'R_occ', 'ROR', 'Split Fail', 'SF_pct']
        dfbcols = dfbcols + [c for c in df_b.columns if c.startswith(ph_dir[ph]) or c == 'Vol']
        df_j = pd.merge(df_b.loc[(df_b.Phase == ph), dfbcols],
                        df_tt_int, left_on='Time',
                        right_on='Date Time').drop('Date Time', axis=1)

        if return_by_cp:
            for cp in df_j.Coord_plan.unique():
                resdict['%s_ph%s_cp%s_tt' % (intx, ph, int(cp))] = df_j.loc[(df_j.Coord_plan == cp), :]
        else:
            resdict['{}_ph{}_tt'.format(intx, ph)] = df_j

        if 'Vol' in df_b.columns:
            df_b.drop('Vol', axis=1, inplace=True)

    if write_to_file:
        if write_to_file == 'xl':
            with pd.ExcelWriter('./ACHD_Eagle-%s/Travel_Time%s.xlsx' % (intx, sfx),
                                mode=xl_mode, engine='openpyxl') as writer:
                for nm, df in resdict.items():
                    df.to_excel(writer, sheet_name=nm, index=False)
        elif write_to_file == 'csv':
            if not os.path.exists('./ACHD_Eagle-{}/TT'.format(intx)):
                os.mkdir('./ACHD_Eagle-{}/TT'.format(intx))
            for nm, df in resdict.items():
                df.to_csv('./ACHD_Eagle-{}/TT/{}.csv'.format(intx, nm), index=False)
        elif write_to_file == 'pk':
            if not os.path.exists('./ACHD_Eagle-{}/TT'.format(intx)):
                os.mkdir('./ACHD_Eagle-{}/TT'.format(intx))
            for nm, df in resdiect.items():
                df.to_pickle('./ACHD_Eagle-{}/TT/{}.pk'.format(intx, nm))

    return resdict


def tt_vol_anova(rd, t_ba, atr_vol=True):
    ph_dir = {2: 'WB', 4: 'SB', 6: 'EB', 8: 'NB'}
    cp_days = {'1': ['Tuesday', 'Wednesday', 'Thursday'],
               '2': ['Tuesday', 'Wednesday', 'Thursday'],
               '4': ['Saturday'], '6': ['Saturday'],
               '4,6': ['Sunday']}
    intx = list(rd.keys())[0].split('_')[0]
    dfres = pd.DataFrame()
    for ph in ph_dir.keys():
        if '{}_ph{}_tt'.format(intx, ph) not in rd.keys():
            continue
        for cp in cp_days.keys():
            drct = ph_dir[ph]
            df = rd['{}_ph{}_tt'.format(intx, ph)].copy()
            df.rename({'Travel Time(Minutes)': 'TT', 'Split Fail': 'SF'}, axis=1, inplace=True)

            if 'Vol' not in df.columns and atr_vol:
                df.loc[:, 'Vol'] = df.loc[:, '{}T'.format(drct)] + df.loc[:, '{}R'.format(drct)]

            df.loc[:, 'BA'] = 0
            df.loc[(df.Time > t_ba), 'BA'] = 1

            df = filter_day(df, cp_days[cp], 'Time')

            cp_split = [n + '.0' for n in cp.split(',')]
            ttb = df.loc[(df.BA == 0) & (df.Coord_plan.astype(str).isin(cp_split)), 'TT'].describe()
            ttb.index = 'TT_B ' + ttb.index
            tta = df.loc[(df.BA == 1) & (df.Coord_plan.astype(str).isin(cp_split)), 'TT'].describe()
            tta.index = 'TT_A ' + tta.index
            volb = df.loc[(df.BA == 0) & (df.Coord_plan.astype(str).isin(cp_split)), 'Vol'].describe()
            volb.index = 'Vol_B ' + volb.index
            vola = df.loc[(df.BA == 1) & (df.Coord_plan.astype(str).isin(cp_split)), 'Vol'].describe()
            vola.index = 'Vol_A ' + vola.index

            srs = pd.concat([ttb, tta, volb, vola])
            str_days = ''
            for day in cp_days[cp]:
                str_days += day[0:3]
            srs.name = 'Ph{} CP{} {}'.format(ph, cp, str_days)

            for v in ['TT', 'AoG_pct', 'SF_pct']:
                ld = '{} ~ Vol + BA + BA * Vol'.format(v)
                lm = ols(ld, df).fit()
                lm_sum = lm.summary()
                t1 = pd.read_html(lm_sum.tables[1].as_html(), header=0, index_col=0)[0]
                rsq = float(lm_sum.tables[0][0][3].data)
                srs['{} RSQ'.format(v)] = rsq
                for i in ['Vol', 'BA', 'BA:Vol']:
                    for j in ['coef', 'P>|t|']:
                        srs['{} {} {}'.format(v, i, j)] = t1.loc[i, j]

            dfres = pd.concat([dfres, srs], axis=1)
    dfres = dfres.T
    dfres.loc[:, 'Phase'] = dfres.index.map(lambda x: x[2])
    dfres.loc[:, 'Coord_plan'] = dfres.index.map(lambda x: x.split('P')[2])
    dfres.loc[:, 'Delta_50%TT'] = (dfres.loc[:, 'TT_A 50%'] -
                                   dfres.loc[:, 'TT_B 50%']) / dfres.loc[:, 'TT_B 50%']
    dfres.loc[:, 'Delta_50%Vol'] = (dfres.loc[:, 'Vol_A 50%'] -
                                    dfres.loc[:, 'Vol_B 50%']) / dfres.loc[:, 'Vol_B 50%']
    dfres_cols = ['Coord_plan', 'Phase'] + dfres.columns[0:16].to_list() + ['Delta_50%TT'] + \
                 dfres.columns[16:32].to_list() + ['Delta_50%Vol'] + dfres.columns[32:-4].to_list()

    return dfres.loc[:, dfres_cols].sort_values(['Coord_plan', 'Phase']).reset_index(drop=True)

