import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from spmfunctions.misc_tools import add_r1r2


def create_detector_comparison_plot(df, id_pairs, path_out, colors=('red', 'blue')):
    """
    Creates an interactive plot comparing detector actuations for given ID pairs.
    
    Parameters:
    df (DataFrame): DataFrame with columns "TS_start", "Code" and "ID"
    id_pairs (list): List of tuples with ID pairs to compare
    colors (tuple): Tuple of colors for first and second detectors in each pair (default: ('red', 'blue'))
    
    Returns:
    plotly.graph_objects.Figure: Interactive plot figure
    """
    
    # Filter dataframe to where Code is 82 or 81
    filtered_df = df[df['Code'].isin([81, 82])].copy()
    
    # Sort by ID and TS_start for proper processing
    filtered_df = filtered_df.sort_values(['ID', 'TS_start']).reset_index(drop=True)
    
    # Create actuation dataframe more efficiently
    actuation_df = create_actuation_dataframe(filtered_df)
    
    # Create figure
    fig = go.Figure()
    
    # Get colors for first and second detectors
    first_color, second_color = colors
    
    # Process each ID pair
    for pair_idx, (id1, id2) in enumerate(id_pairs):
        # Get actuations for ID1 (first detector)
        actuations_id1 = actuation_df[actuation_df['ID'] == id1]

        # Get actuations for ID2 (second detector)
        actuations_id2 = actuation_df[actuation_df['ID'] == id2]
        
        # Add actuation lines for ID1 (first detector) - use first color
        for i, (_, row) in enumerate(actuations_id1.iterrows()):
            fig.add_trace(go.Scatter(
                x=[row['start_time'], row['end_time']],
                y=[pair_idx * 3 + 2, pair_idx * 3 + 2],  # Position at top of group
                mode='lines',
                line=dict(color=first_color, width=10),
                name=f'{id1}',
                hovertemplate=f'ID: {id1}<br>Start: {row["start_time"]}<br>End: {row["end_time"]}<extra></extra>'
            ))
        
        # Add actuation lines for ID2 (second detector) - use second color
        for i, (_, row) in enumerate(actuations_id2.iterrows()):
            fig.add_trace(go.Scatter(
                x=[row['start_time'], row['end_time']],
                y=[pair_idx * 3 + 1, pair_idx * 3 + 1],  # Position at middle of group
                mode='lines',
                line=dict(color=second_color, width=10),
                name=f'{id2}',
                hovertemplate=f'Detector: {id2}<br>Start: {row["start_time"]}<br>End: {row["end_time"]}<extra></extra>'
            ))

    
    # Update layout
    fig.update_layout(
        title='Detector Actuations',
        xaxis_title='Time',
        yaxis_title='Detector',
        height=150 + len(id_pairs) * 100,  # Adjust height based on number of pairs
        showlegend=False,
        hovermode='closest'
    )
    
    # Customize y-axis labels and add visual spacing
    y_tick_vals = []
    y_tick_texts = []
    for i, (id1, id2) in enumerate(id_pairs):
        y_tick_vals.extend([i * 3 + 2, i * 3 + 1])
        y_tick_texts.extend([f'{id1}', f'{id2}'])
    
    fig.update_yaxes(
        tickvals=y_tick_vals,
        ticktext=y_tick_texts,
        range=[-0.5, len(id_pairs) * 3 - 0.5],  # Add padding at top and bottom
        showgrid=False  # Remove default grid lines
    )
    
    fig.write_html(path_out)

def create_actuation_dataframe(df):
    """
    Creates actuation dataframe by grouping by ID, sorting by TS_start, 
    and using shift to pair start (82) with end (81) events.
    
    Parameters:
    df (DataFrame): Filtered dataframe with Codes 81 and 82
    
    Returns:
    DataFrame: DataFrame with columns ['ID', 'start_time', 'end_time'] for each actuation
    """
    
    if len(df) == 0:
        return pd.DataFrame(columns=['ID', 'start_time', 'end_time'])
    
    # Sort by ID and TS_start
    df_sorted = df.sort_values(['ID', 'TS_start']).reset_index(drop=True)
    
    # Add next timestamp as end_time for each row
    df_sorted['end_time'] = df_sorted.groupby('ID')['TS_start'].shift(-1)
    
    # Keep only start events (Code 82) that have a corresponding end event
    actuation_df = df_sorted[(df_sorted['Code'] == 82) & (df_sorted['end_time'].notna())].copy()
    
    # Rename columns for clarity
    actuation_df = actuation_df.rename(columns={'TS_start': 'start_time'})
    
    # Return only the needed columns
    return actuation_df[['ID', 'start_time', 'end_time']].reset_index(drop=True)

def plot_coord(df_comb, det_srs, path_out, both=[], sfx=''):
    '''
    Plots coordination/split diagram.

    Args:
    - df_comb (DataFrame): Input DataFrame containing traffic event data.
    - det_srs (srs): Detector configuration series.
    - path_out (str): Output path for the plot.
    - sfx (str): Suffix for the output filename.
    '''

    figs = []

    cl_RY = {'Rc': 'Red', 'Y': 'Yellow'}
    cl_P = {'1': 'DarkOrange', '5': 'LightSalmon', '3': 'Magenta', '7': 'Purple',
            '2': 'Blue', '6': 'Turquoise', '4': 'DarkGreen', '8': 'Lime',
            '9': 'DarkOliveGreen', '10': 'Olive', '11': 'DarkSlateGray', '12': 'Gray',
            '13': 'LightGray', '14': 'Silver', '15': 'DimGray', '16': 'Black'}

    # Colors for detectors
    cl_det = {
        'Ar': 'DimGray',       # medium gray for arrivals
        'Oc': 'Black',         # black for occupancy
        'St': 'Crimson'        # high-contrast red for stop bar
    }

    ring_dict = ring_dfs(df_comb, both=both)

    for r in [1, 2]:
        df_plt = ring_dict[r]
        df_plt['y'] = -5

        leg=list(set(int(x.split(' ')[1]) for x in df_plt.columns if len(x.split(' '))>2))+[0]
        
        # coord plan
        fig = px.scatter(df_plt, x='Cycle_start', y='y',
                         title=f'Ring {r} Coordination/Split Diagram',
                         hover_data='Coord_plan',
                         height=500)
        fig.update_traces(marker=dict(symbol='line-ew'))

        for c in df_plt.columns[1:-2].to_list():
            gyr, p = c.split()[0:2]

            if gyr == 'G':
                cl = cl_P[p]
            else:
                cl = cl_RY[gyr]

            t = px.bar(df_plt, x='Cycle_start', y=c,
                       color_discrete_sequence=[cl],
                       hover_data=c).data[0]

            t.name = f'Ph {p}'
            fig.add_trace(t)

            if int(p) in leg:
                fig.data[-1].update(showlegend=True)
                leg.remove(int(p))
                
        leg=list(set(int(x.split(' ')[1]) for x in df_plt.columns if len(x.split(' '))>2))+[0]

        fig.update_layout(xaxis_range=[df_plt.iloc[0, 0], df_plt.iloc[0, -1]], barmode='stack')

        leg=list(set(int(x.split(' ')[1]) for x in df_plt.columns if len(x.split(' '))>2))+[0]

        for idx, dets in det_srs.items():
            if int(idx[1]) in leg:

                # Determine detector type & set shift/color
                if 'Ar' in idx:
                    x_shift = -10  # seconds
                    color = cl_det['Ar']
                elif 'St' in idx:
                    x_shift = 10
                    color = cl_det['St']
                elif 'Oc' in idx:
                    x_shift = 0
                    color = cl_det['Oc']
                else:
                    # Default (no type match)
                    x_shift = 0
                    color = 'Gray'                


                for det in dets.split(','):
                    df_dt = df_comb[(df_comb.Code == 82) & (df_comb.ID == int(det))]
	
                    # Apply shift
                    df_dt['Cycle_start_shifted'] = df_dt['Cycle_start'] + pd.to_timedelta(x_shift, unit='s')

                    t = px.scatter(df_dt, x='Cycle_start_shifted', y='t_cs',
                                   hover_name='ID',
                                   hover_data='Duration',
                                   color_discrete_sequence=[color],
                                   opacity=0.75).data[0]
                    t.legendgroup = det
                    t.name = f'{idx[:5]} ( Det {int(det)})'
                    fig.add_trace(t)
                    ## SLIDER
                    # Track arrival detector traces for slider
                    if 'Ar' in idx:
                        if 'arrival_traces' not in locals():
                            arrival_traces = []
                        arrival_traces.append({
                            "trace_index": len(fig.data) - 1,
                            "orig_y": t.y.copy()
                        })
                    ## END SLIDER
                    fig.data[-1].update(showlegend=True, visible='legendonly')

        fig.layout.xaxis.title = ''
        fig.layout.yaxis.title = ''
        fig.layout.xaxis.autorange = True
        fig.layout.yaxis.autorange = True

        ## SLIDER
        if 'arrival_traces' in locals() and arrival_traces:

            steps = []

            # Build slider steps for offsets from -30 to +30 in steps of 5
            for offset in range(-30, 31, 1):

                # Build updated y-arrays (one per arrival trace)
                updated_y = [tr["orig_y"] + offset for tr in arrival_traces]

                # List the trace indices being updated
                trace_indices = [tr["trace_index"] for tr in arrival_traces]

                step = {
                    "label": f"{offset}",
                    "method": "restyle",
                    "args": [
                        {"y": updated_y},   # update y-values
                        trace_indices        # which traces to apply to
                    ],
                }
                steps.append(step)

            fig.update_layout(
                sliders=[
                    {
                        "active": 30,  # offset = 0 corresponds to index 6 in (-30..30 step 5)
                        "steps": steps,
                        "currentvalue": {
                            "prefix": "Arrival Offset (sec): "
                        },
                        "pad": {"t": 40}
                    }
                ]
            )


        # Reset for next ring
        if 'arrival_traces' in locals():
            del arrival_traces        
        ## END SLIDER

        figs.append(fig)

    path_out = os.path.join(path_out, f'Coord_Split_{sfx}.html')
    figs[0].write_html(path_out)
    with open(path_out, 'a') as f:
        f.write(figs[1].to_html(full_html=False, include_plotlyjs='cdn'))


def plot_term(df, save_path, intx, line=True, n_con=10, sfx=''):
    '''
    Plots termination plot using Plotly.
    df: df_raw
    save_path: path to output file
    intx: name of intersection
    line: default=True, plot line showing weighted average of number of max outs
    n_con: default=10, number of consecutive cycles to calculate and plot weighted average
    sfx: suffix for output filename
    '''

    fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{intx} Phase Termination"])

    # === Termination Lines ===
    if line:
        df_term = df.copy().loc[df['Code'].isin([4, 5, 6]), :]
        df_term['Termination'] = (df_term['Code'] - 4).clip(upper=1) * 0.667 + df_term['ID'] - 0.333
        df_term['Termination'] = df_term.groupby('ID')['Termination'].transform(lambda t: t.rolling(n_con, center=True).mean())

        x_range = [df_term['TS_start'].min(), df_term['TS_start'].max()]
        for y in range(2, (2 + 3 * df_term['ID'].max())):
            fig.add_trace(go.Scatter(x=x_range, y=[y / 3] * 2, mode='lines', line=dict(color='gray'), showlegend=False))

        for ID, df_ID in df_term.groupby('ID'):
            fig.add_trace(go.Scatter(x=df_ID['TS_start'], y=df_ID['Termination'],
                                     mode='lines', line=dict(color='black'), showlegend=False))

    # === Termination Points ===
    fig.add_trace(go.Scatter(x=df[df['Code'] == 4]['TS_start'],
                             y=df[df['Code'] == 4]['ID'],
                             mode='markers',
                             marker=dict(color='green', symbol='circle'),
                             name='Gap Out'))

    fig.add_trace(go.Scatter(x=df[df['Code'] == 5]['TS_start'],
                             y=df[df['Code'] == 5]['ID'],
                             mode='markers',
                             marker=dict(color='red', symbol='square'),
                             name='Max Out'))

    fig.add_trace(go.Scatter(x=df[df['Code'] == 6]['TS_start'],
                             y=df[df['Code'] == 6]['ID'],
                             mode='markers',
                             marker=dict(color='orangered', symbol='diamond'),
                             name='Force Off'))

    # === Preempt Points (Code 105) ===
    if 'TS_start' in df.columns and 'Code' in df.columns:
        df_pre = df[df['Code'] == 105]
        if not df_pre.empty:
            fig.add_trace(go.Scatter(x=df_pre['TS_start'],
                                     y=df_pre['ID'] + 0.35,
                                     mode='markers',
                                     marker=dict(color='magenta', symbol='x', size=10),
                                     name='Preempt'))

    # === Pedestrian Service Points (Code 21 preceded by Code 45) ===
    df_p = df[df['Code'].isin([21, 45])].copy()
    if not df_p.empty:
        df_p = df_p.sort_values(['ID', 'TS_start'])
        legit_21 = []

        for pid, grp in df_p.groupby('ID'):
            seen_call = False
            for idx, row in grp.iterrows():
                if row['Code'] == 45:
                    seen_call = True
                elif row['Code'] == 21:
                    if seen_call:
                        legit_21.append(idx)
                    seen_call = False

        df_legit_21 = df_p.loc[legit_21]
        if not df_legit_21.empty:
            fig.add_trace(go.Scatter(x=df_legit_21['TS_start'],
                                     y=df_legit_21['ID'] + 0.1,
                                     mode='markers',
                                     marker=dict(color='blue', symbol='triangle-up', size=8),
                                     name='Pedestrian Service'))

    # === Layout ===
    fig.update_layout(
        xaxis=dict(title='Time', type="date"),
        yaxis=dict(title='Phase'),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    save_path = os.path.join(save_path, f'Phase_Termination_{sfx}.html')
    fig.write_html(save_path)

def ring_dfs(df_comb,both=[]):
    '''
    Generates ring dataframes based on the input traffic events dataframe.

    Args:
    - df_comb (DataFrame): Input DataFrame containing traffic event data.
    - both (list): List of phases to be included in both rings, e.g. for split phasing. Default is empty list.

    Returns:
    - dict: Dictionary containing ring dataframes for Ring 1 and Ring 2.
    '''

    df_comb = add_r1r2(df_comb,both=both)

    resdict = {}
    for r in [1, 2]:
        seq = df_comb.loc[:, f'R{r}_Ph'].dropna().unique()
        df_seq = pd.DataFrame([x.split(',') for x in seq], index=seq)
        df_seq.columns = [f' ({x})' for x in range(1, len(df_seq.columns) + 1)]
        df_seq = df_seq.astype('str') + df_seq.columns
        cols = []
        for c in df_seq.columns:
            for x in (df_seq.loc[:, c].dropna().unique()):
                for y in ['G', 'Y', 'Rc']:
                    if 'None' not in x:
                        cols.append(f'{y} {x}')

        df_ring = pd.DataFrame(columns=cols)

        temp_dfs = []
        for seq, row in df_seq.iterrows():
            if seq.startswith('None'):
                continue

            idx = []
            for v in row:
                if v.startswith('None'):
                    continue
                for st in ['G', 'Y', 'Rc']:
                    idx.append(f'{st} {v}')

            df = df_comb[(df_comb.ID.isin([int(x) for x in seq.split(',')])) & (df_comb.Code.isin([1, 8, 9])) &
                         (df_comb.loc[:, f'R{r}_Ph'] == seq)]

            for cs, dfgb in df.groupby('Cycle_start'):
                srs = dfgb.Duration
                if len(srs) != len(idx):
                    continue
                srs.index = idx
                srs.name = cs
                temp_dfs.append(pd.DataFrame(srs).T)
                #df_ring = pd.concat([df_ring, pd.DataFrame(srs).T])
        
        df_ring = pd.concat([df_ring] + [df for df in temp_dfs if not df.empty])

        df_ring = df_ring.fillna(0)
        df_ring.index.name = 'Cycle_start'
        df_ring = pd.merge(df_ring, df_comb.loc[:, ['Cycle_start', 'Coord_plan']], how='left', on='Cycle_start').drop_duplicates()
        df_ring.loc[:, 'Coord_plan'] = df_ring.Coord_plan

        resdict[r] = df_ring

    return resdict