import matplotlib
import matplotlib.pyplot as plt


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


def plot_clients_profile(target_client, source_df, datetime_label, validation_index=None, evaluation_index=None):
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    fig.suptitle(f"Visualization of client {target_client}")
    considered_client = source_df.set_index(datetime_label)[target_client]
    if validation_index is not None and evaluation_index is not None:
        ax.axvspan(source_df[datetime_label].min().to_pydatetime(), validation_index, color="black", alpha=0.15)
        ax.axvspan(validation_index, evaluation_index, color="tab:olive", alpha=0.15)
        ax.axvline(validation_index, ls=":", color="k")
        ax.axvline(evaluation_index, ls=":", color="k")
    considered_client.plot(ax=ax)
    fig.tight_layout()
    fig.savefig(f"ELD/figures/overall_profile_{target_client}.png", dpi=300)
    # Avoid figure to be randomly placed
    # move_figure(fig, 0, 0)
    plt.show()


if __name__ == '__main__':
    import os
    import argparse
    import pandas as pd
    from datetime import datetime
    from matplotlib.colors import ListedColormap
    from eld_ds_analysis import determine_chunks_value, plot_categorical_heatmap

    parser = argparse.ArgumentParser(description='Original ELD Dataset Transformation to Hourly dataset')
    parser.add_argument('--source-path', type=str, required=True,
                        help='Input the path for the parquet file created during analysis')
    args = parser.parse_args()

    source_data_path = args.source_path

    informer_ecl_df = pd.read_parquet(os.path.join(source_data_path, f"informer_eld.parquet"))

    training_start = datetime(2012, 1, 1, 0, 0)
    validation_start = datetime(2013, 1, 1, 0, 0)
    evaluation_start = datetime(2014, 1, 1, 0, 0)

    informer_ecl_df = informer_ecl_df[informer_ecl_df['date'] >= training_start]

    print(informer_ecl_df)

    # clients list
    client_ids = list(informer_ecl_df.columns[1:])

    # Late arrival threshold: 2 months
    late_arrival_threshold = 61 * 24

    # Limit for consecutive zero: 1 Day
    consecutive_missing_time_steps_threshold = 24

    # Determine the size of the zero chunks, big chunks is an issue as we cannot interpolate appropriately
    eld_client_chunks = dict()
    for pid in client_ids:
        eld_client_chunks[pid] = determine_chunks_value(informer_ecl_df[pid].values)

    # Create a dict without the index of the zero block
    eld_client_chunk_sizes = dict()
    for element in eld_client_chunks.keys():
        eld_client_chunk_sizes[element] = list()
        for chunk_idx in eld_client_chunks[element].keys():
            eld_client_chunk_sizes[element].append(eld_client_chunks[element][chunk_idx])

    client_chunks_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in eld_client_chunk_sizes.items()]))
    print(client_chunks_df)

    # clients are considered to arrive late if, their first values are a set of consecutive zeros above the
    # threshold
    late_arrival_clients = list()
    clients_with_missing_portions = list()
    early_departure_clients = list()
    for pid in eld_client_chunks.keys():
        for k, v in eld_client_chunks[pid].items():
            if k == 0:
                # first data are a chunk of consecutive zeros
                if v > consecutive_missing_time_steps_threshold:
                    late_arrival_clients.append(pid)
            else:
                if k + v == len(informer_ecl_df):
                    if pid not in late_arrival_clients:
                        early_departure_clients.append(pid)
            if v > consecutive_missing_time_steps_threshold:
                if pid not in late_arrival_clients and pid not in early_departure_clients:
                    if pid not in clients_with_missing_portions:
                        clients_with_missing_portions.append(pid)

    print(f"There are {len(late_arrival_clients)} clients arriving late in this dataset")
    # for pid in late_arrival_clients:
    #     plot_clients_profile(pid, informer_ecl_df, 'date', validation_start, evaluation_start)

    print(f"There are {len(clients_with_missing_portions)} clients with gap of missing data in this dataset")
    # for pid in clients_with_missing_portions:
    #     plot_clients_profile(pid, informer_ecl_df, 'date', validation_start, evaluation_start)

    print(f"There are {len(early_departure_clients)} clients leaving early in this dataset")
    # for pid in early_departure_clients:
    #     plot_clients_profile(pid, informer_ecl_df, 'date', validation_start, evaluation_start)

    # 132 have regular no consumption day, therefore it could be normal, and we will keep it

    remaining_clients = list(set(client_ids) - set(late_arrival_clients) -
                             set(clients_with_missing_portions) - set(early_departure_clients))
    print(f"There are {len(remaining_clients)} clients remaining in this dataset")
    # for pid in remaining_clients:
    #     plot_clients_profile(pid, informer_ecl_df, 'date', validation_start, evaluation_start)

    client_to_remove = ['MT_106', 'MT_245', 'MT_298', 'MT_182', 'MT_146', 'MT_127', 'MT_307', 'MT_122', 'MT_057',
                        'MT_310', 'MT_032', 'MT_002', 'MT_114']

    for ptr in client_to_remove:
        plot_clients_profile(ptr, informer_ecl_df, 'date',
                             validation_index=informer_ecl_df.loc[int(len(informer_ecl_df) * 0.7), 'date'].to_pydatetime(),
                             evaluation_index=informer_ecl_df.loc[int(len(informer_ecl_df) * 0.8), 'date'].to_pydatetime())

    client_to_keep = ['MT_118', 'MT_107', 'MT_299']

    revised_ecl_df = informer_ecl_df.drop(columns=client_to_remove)

    print("Create plotting dataframe")
    plot_df = revised_ecl_df.set_index('date')
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

    # Plot the categorical map of all clients
    plot_categorical_heatmap(categorical_plot_df.T,
                             "Electricity consumption of all clients from revised ECL",
                             my_cmap, norm)

    revised_ecl_df.to_csv(os.path.join("revised_datasets/", f"PELD_1H_3Y_308.csv"), index=False)
    print(revised_ecl_df.info())
    print(informer_ecl_df.info())

