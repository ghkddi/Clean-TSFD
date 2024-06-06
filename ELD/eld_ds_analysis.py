import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def count_zeros(source_df):
    zero_count = dict()
    for column in source_df.columns:
        zero_count[column] = len(source_df[source_df[column] == 0.0])
    return zero_count


def determine_chunks_value(source_series, target_value=0.0):
    # lists to test the function
    # a = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # b = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    # c = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # d = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    # e = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1]
    # f = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    # g = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    chunks = dict()
    last_idx = 0
    for idx in range(len(source_series)):
        # print(idx, last_idx)
        if idx >= last_idx:
            if source_series[idx] == target_value:
                n_idx = idx + 1
                # print('next', n_idx)
                if n_idx == len(source_series):
                    chunks[idx] = n_idx - idx
                else:
                    # print('while')
                    while source_series[n_idx] == target_value:
                        n_idx += 1
                        if n_idx == len(source_series):
                            chunks[idx] = n_idx - idx
                            break
                    chunks[idx] = n_idx - idx
                    last_idx = n_idx + 1
        else:
            continue
    return chunks


def determine_chunk_size_category(target_size):
    categorical_chunk_size = {
        'c1': {"min": 0, "max": 1},
        'c2': {"min": 2, "max": 10},
        'c3': {"min": 11, "max": 20},
        'c4': {"min": 21, "max": 30},
        'c5': {"min": 31, "max": 40},
        'c6': {"min": 41, "max": 50},
        'c7': {"min": 51, "max": 60},
        'c8': {"min": 61, "max": 70},
        'c9': {"min": 71, "max": 80},
        'c10': {"min": 81, "max": 90},
        'c11': {"min": 91, "max": 100},
        'c12': {"min": 101, "max": 150},
        'c13': {"min": 151, "max": 200},
        'c14': {"min": 201, "max": 250},
        'c15': {"min": 251, "max": 300},
        'c16': {"min": 301, "max": 350},
        'c17': {"min": 351, "max": 400},
        'c18': {"min": 401, "max": 450},
        'c19': {"min": 451, "max": 500},
        'c20': {"min": 501, "max": 1000},
        'c21': {"min": 1001, "max": 2000},
        'c22': {"min": 2001, "max": 3000},
        'c23': {"min": 3001, "max": 4000},
        'c24': {"min": 4001, "max": 5000},
        'c25': {"min": 5001, "max": 6000},
        'c26': {"min": 8001, "max": 9000},
        'c27': {"min": 9001, "max": 10000},
        'c28': {"min": 10001, "max": 35000},
        'c29': {"min": 35001, "max": 130000},
    }
    for ccs in categorical_chunk_size.keys():
        if categorical_chunk_size[ccs]['min'] <= target_size <= categorical_chunk_size[ccs]['max']:
            return ccs
    return 'c30'


def plot_participants_and_heatmap(target_participant, source_df, categorical_df, considered_cmap, considered_norm,
                                  add_cbar=False, validation_index=None, evaluation_index=None):
    if add_cbar:
        fig, ax = plt.subplot_mosaic("AAL;BBL", figsize=(20, 15))
        sns.heatmap(np.array(categorical_df[target_participant]).reshape(1, len(categorical_df)),
                    cmap=considered_cmap, norm=considered_norm, ax=ax['A'], cbar_ax=ax['L'])
    else:
        fig, ax = plt.subplot_mosaic("AA;BB", figsize=(20, 15))
        sns.heatmap(np.array(categorical_df[target_participant]).reshape(1, len(categorical_df)),
                    cmap=considered_cmap, norm=considered_norm, ax=ax['A'], cbar=False)
    fig.suptitle(f"Visualization of participant {target_participant}")
    ax['A'].set_axis_off()
    considered_participant = source_df[target_participant].resample('h').sum()
    if validation_index is not None and evaluation_index is not None:
        ax['B'].axvspan(source_df.index[0].to_pydatetime(), validation_index, color="black", alpha=0.15)
        ax['B'].axvspan(validation_index, evaluation_index, color="tab:olive", alpha=0.15)
        ax['B'].axvline(validation_index, ls=":", color="k")
        ax['B'].axvline(evaluation_index, ls=":", color="k")
    considered_participant.plot(ax=ax['B'])
    plt.show()


def format_date_ticks(old_ticks: list[plt.Text]) -> list[str]:
    text = [l.get_text() for l in old_ticks]  # plt.Text to str
    return pd.to_datetime(text).date  # str to datetime, then format as desired


def plot_categorical_heatmap(df_to_plot, plot_title, considered_cmap, considered_norm, verbose=False):
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    sns.heatmap(df_to_plot, cmap=considered_cmap, norm=considered_norm, ax=ax)
    # get and transform old ticks
    new_ticks = format_date_ticks(ax.get_xticklabels())
    ax.set_xticklabels(new_ticks)
    ax.set_title(plot_title)
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(f"ELD/figures/{plot_title}.png", dpi=300)
    if verbose:
        plt.show()


if __name__ == '__main__':
    import os
    import argparse
    import matplotlib
    import numpy as np
    from zipfile import ZipFile
    from matplotlib.colors import ListedColormap

    parser = argparse.ArgumentParser(description='ELD datasets Analysis')
    parser.add_argument('--source-path', type=str, required=True,
                        help='Input the path for the parquet file created during analysis')
    parser.add_argument('--considered-file', type=str, required=True,
                        help='Dataset file to check (Original, Informer, Autoformer')
    args = parser.parse_args()

    source_data_path = args.source_path
    considered_file = args.considered_file

    file_name = {
        "original": {"path": "Original/", "zip": "electricityloaddiagrams20112014.zip", "file": "LD2011_2014.txt",
                     "num_participant": 370, "granularity": "15min"},
        "informer": {"path": "", "zip": "ECL.csv-20230804T085611Z-001.zip", "file": "ECL.csv",
                     "num_participant": 321, "granularity": "1h"},
        "autoformer": {"path": "", "zip": "autoformer_electricity.zip", "file": "electricity.csv",
                       "num_participant": 321, "granularity": "1h"},
    }

    date_label = 'date'
    threshold_in_months = 2
    if threshold_in_months == 6:
        # 6 months
        half_year_time_steps = 183 * 24
    elif threshold_in_months == 4:
        # 4 months
        half_year_time_steps = 122 * 24
    elif threshold_in_months == 2:
        # 2 months
        half_year_time_steps = 61 * 24
    else:
        raise NotImplementedError

    if file_name[considered_file]["granularity"] == "1h":
        half_year_time_steps *= 1
    elif file_name[considered_file]["granularity"] == "15min":
        half_year_time_steps *= 4
    else:
        raise NotImplementedError

    if file_name[considered_file]["path"] == "":
        considered_path = source_data_path
        with ZipFile(os.path.join(considered_path, file_name[considered_file]["zip"])) as zf:
            with zf.open(file_name[considered_file]["file"], 'r') as f:
                eld_df = pd.read_csv(f, header=0)
    else:
        considered_path = os.path.join(source_data_path, file_name[considered_file]["path"])
        with ZipFile(os.path.join(considered_path, file_name[considered_file]["zip"])) as zf:
            with zf.open(file_name[considered_file]["file"], 'r') as f:
                col_type = dict()
                for i in range(1, file_name[considered_file]["num_participant"]+1):
                    col_type['MT_%03d' % i] = str
                eld_df = pd.read_csv(f, header=0, delimiter=';', dtype=col_type).rename(
                    columns={'Unnamed: 0': date_label})
                data_col_names = list(col_type.keys())
                # The original dataset separate decimal with comma not point
                for k in data_col_names:
                    eld_df[k] = eld_df[k].str.replace(',', '.')
                    eld_df[k] = pd.to_numeric(eld_df[k])

    eld_df[date_label] = pd.to_datetime(eld_df[date_label])
    print(eld_df.info())
    print(eld_df.describe())
    # Save temporary files to simplify loading the dataset
    eld_df.to_parquet(os.path.join(considered_path, f"{considered_file}_eld.parquet"))
    eld_df.to_csv(os.path.join(considered_path, f"{considered_file}_eld.csv"), index=False)

    if considered_file == "original":
        validation_start = datetime.datetime(2013, 1, 1, 0, 0)
        evaluation_start = datetime.datetime(2014, 1, 1, 0, 0)
    else:
        validation_start = eld_df.loc[int(len(eld_df) * 0.7), date_label].to_pydatetime()
        evaluation_start = eld_df.loc[int(len(eld_df) * 0.8), date_label].to_pydatetime()

    print(eld_df[date_label].min(), eld_df[date_label].max())
    eld_datarange = pd.date_range(eld_df[date_label].min(), eld_df[date_label].max(),
                                  freq=file_name[considered_file]["granularity"], inclusive='both')
    # print(len(eld_df), len(eld_datarange))
    if len(eld_df) == len(eld_datarange):
        print(f"Dataset has the expected number of timesteps: {len(eld_df)}")
    else:
        # Missing timesteps
        pass
    print(eld_df)

    print("Create plotting dataframe")
    plot_df = eld_df.set_index(date_label)
    norm_plot_df = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min())

    plot_desc = plot_df.describe()
    print("Create categorical plotting dataframe for heatmap")
    categorical_plot_df = plot_df.copy()
    for column in plot_df.columns:
        p50 = plot_desc.loc['mean', column]
        p25 = p50 * 25 / 50
        p75 = p50 * 75 / 50
        p100 = p50 * 100 / 50
        categorical_plot_df[column] = plot_df[column].case_when([
            (plot_df.eval(f"{column} == 0"), 0),
            (plot_df.eval(f"0 < {column} <= {p25}"), 1),
            (plot_df.eval(f"{p25} < {column} <= {p50}"), 2),
            (plot_df.eval(f"{p50} < {column} <= {p75}"), 3),
            (plot_df.eval(f"{p75} < {column} <= {p100}"), 4),
            (plot_df.eval(f"{p100} < {column}"), 5),
        ])

    # Create colormap for categorical representation
    my_cmap = ListedColormap(["black", "lightseagreen", "tab:green", "gold", "darkorange", "tab:red"])
    bounds = [-.5, .5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, my_cmap.N)

    # Plot the categorical map of all participants
    plot_categorical_heatmap(categorical_plot_df.T, f"Electricity consumption of all participants from {considered_file}",
                             my_cmap, norm)

    # Investigate participants with several zeros
    zc = count_zeros(plot_df)

    # Determine the size of the zero chunks, big chunks is an issue as we cannot interpolate appropriately
    eld_participant_chunks = dict()
    for column in plot_df.columns:
        eld_participant_chunks[column] = determine_chunks_value(plot_df[column].values)

    # Create a dict without the index of the zero block
    eld_participant_chunk_sizes = dict()
    for element in eld_participant_chunks.keys():
        eld_participant_chunk_sizes[element] = list()
        for chunk_idx in eld_participant_chunks[element].keys():
            eld_participant_chunk_sizes[element].append(eld_participant_chunks[element][chunk_idx])

    participant_chunks_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in eld_participant_chunk_sizes.items()]))
    print(participant_chunks_df)

    late_arrival_participants = list(participant_chunks_df.T[participant_chunks_df.loc[0, :] > half_year_time_steps].T.columns)
    print(f"There are {len(late_arrival_participants)} participants arriving late in this dataset")
    if len(late_arrival_participants) > 0:
        plot_categorical_heatmap(categorical_plot_df[sorted(late_arrival_participants)].T,
                                 f"Electricity consumption of late arrival participants from {considered_file}",
                                 my_cmap, norm)

    ''' All participants that have only zero data for the first year are ok when reducing the dataset to 3 years. But 
    lots of participants have missing data during the whole training period or the even during evaluation. Such 
    participants will impact the evaluation metrics outputs. Indeed, for zero values as target models that will 
    correctly predict zero or close to zero values will have good performance, while for non zero values models 
    that will predict zero will perform poorly. Such specific participants should be removed from a benchmark 
    dataset that aims to compare model architecture performance for long inputs to long output prediction scenario.
    '''

    other_participants = list(set(participant_chunks_df.columns) - set(late_arrival_participants))

    # participants that never have zeros in their data
    no_zero_participants = list()
    early_departure = list()
    for column in other_participants:
        unique_chunks = participant_chunks_df[column].unique()
        if len(unique_chunks) == 1:
            no_zero_participants.append(column)
        else:
            if unique_chunks[-2] > half_year_time_steps:
                early_departure.append(column)

    # Check early departure
    print(f"There are {len(early_departure)} participants leaving early in this dataset")
    if len(early_departure) > 0:
        plot_categorical_heatmap(categorical_plot_df[sorted(early_departure)].T,
                                 f"Electricity consumption of early departure participants from {considered_file}",
                                 my_cmap, norm)

    # Check no zero participants
    print(f"There are {len(no_zero_participants)} participants without any zero values in this dataset")
    if len(no_zero_participants) > 0:
        plot_categorical_heatmap(categorical_plot_df[sorted(no_zero_participants)].T,
                                 f"Electricity consumption of all participants without any zero values from {considered_file}",
                                 my_cmap, norm)

    # Strange participants
    if considered_file == "original":
        plot_participants_and_heatmap('MT_156', plot_df, categorical_plot_df, my_cmap, norm,
                                      validation_index=validation_start, evaluation_index=evaluation_start)
    elif considered_file == "informer":
        plot_participants_and_heatmap('MT_127', plot_df, categorical_plot_df, my_cmap, norm,
                                      validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_142', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_143', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_144', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_183', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_191', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_193', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_207', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_209', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_210', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_217', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_219', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_225', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_232', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_235', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_241', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_246', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_252', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_258', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_260', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_265', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_269', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_270', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_275', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)
        # plot_participants_and_heatmap('MT_282', plot_df, categorical_plot_df, my_cmap, norm,
        #                          validation_index=validation_start, evaluation_index=evaluation_start)

    # Remaining participants have lots of zero block and will need specific attention
    other_participants = list(set(other_participants) - set(no_zero_participants) - set(early_departure))
    print(f"There are {len(other_participants)} participants remaining to be checked.")
    if len(other_participants) > 0:
        plot_categorical_heatmap(categorical_plot_df[sorted(other_participants)].T,
                                 f"Electricity consumption of other participants from {considered_file}",
                                 my_cmap, norm)
    print("Total participants: ", file_name[considered_file]["num_participant"], " /// ",
          len(late_arrival_participants) + len(no_zero_participants) + len(early_departure) + len(other_participants),
          " /// ",
          len(late_arrival_participants), len(no_zero_participants), len(early_departure), len(other_participants))

    # # diversity of no_zero_participants
    # plot_df[no_zero_participants].plot(alpha=0.1, color='gray', figsize=(15, 6), legend=False)
    # # plot_df[late_arrival_participants].plot(alpha=0.1, color='gray', figsize=(15, 6), legend=False)
    # plot_df[early_departure].plot(alpha=0.1, color='gray', figsize=(15, 6), legend=False)
    # plot_df[other_participants].plot(alpha=0.1, color='gray', figsize=(15, 6), legend=False)

    chunk_size_count_df = pd.DataFrame(columns=['participants', 'chunk_size', 'chunk_count'])
    for column in other_participants:
        unique_chunk_size = participant_chunks_df[column].unique()
        for ucs in unique_chunk_size:
            if not np.isnan(ucs):
                chunk_size_count_df.loc[len(chunk_size_count_df)] = [
                    column, ucs, len(participant_chunks_df[participant_chunks_df[column] == ucs])]
    #
    # # chunk_size_count_df['log_count'] = chunk_size_count_df['chunk_count'].apply(lambda x: np.log(x + 1))
    #
    # # sns.scatterplot(data=chunk_size_count_df, x='participants', y='chunk_size', size="log_count")
    # sns.scatterplot(data=chunk_size_count_df, x='participants', y='chunk_size')
    # plt.show()
    #
    # chunk_size_count_df['cat_chunk_size'] = chunk_size_count_df['chunk_size'].apply(lambda x: determine_chunk_size_category(x))
    #
    # cat_chunk_size_count_df = chunk_size_count_df[['participants', 'chunk_count', 'cat_chunk_size']].groupby(
    #     ['participants', 'cat_chunk_size']).sum().reset_index()
    # cat_chunk_size_count_df['log_count'] = cat_chunk_size_count_df['chunk_count'].apply(lambda x: np.log(x + 1))
    # sns.scatterplot(data=cat_chunk_size_count_df, x='participants', y='cat_chunk_size', size="log_count", alpha=0.5)
