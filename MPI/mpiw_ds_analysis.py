import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype


def plot_mpi_dataset(source_data, validation_start, evaluation_start, datetime_label, raw_source='mpi', phase='analysis'):
    plot_df = source_data.copy()
    ffig, fax = plt.subplot_mosaic("AB;CD;EF;GH;IJ;KL;MN;OP", sharex=True, figsize=(20, 15))
    if len(plot_df) > 53000:
        plot_df['date'] = plot_df[datetime_label].apply(lambda x: x.date())
        plot_df['hour'] = plot_df[datetime_label].apply(lambda x: x.hour)
        plot_df = plot_df.drop(columns=datetime_label).groupby(['date', 'hour']).mean().reset_index()
        plot_df.loc[:, datetime_label] = pd.to_datetime(
            plot_df.date.astype(str) + ' ' + plot_df.hour.astype(str) + ':00')
        plot_df = plot_df.drop(columns=['date', 'hour'])
        plot_df['is_ts_missing'] = plot_df.is_ts_missing.apply(lambda v: 0 if v == 0 else 1)
        print(plot_df)
    missing_df = plot_df[plot_df['is_ts_missing'] > 0]
    wv_value_error = plot_df[plot_df['is_wv_value_error'] > 0]
    if 'SWDR_value_error' in plot_df.columns:
        SWDR_value_error_df = plot_df[plot_df['is_SWDR_value_error'] > 0]
    mPAR_value_error_df = plot_df[plot_df['is_maxPAR_value_error'] > 0]
    if raw_source == 'autoformer':
        CO2_value_error_df = plot_df[plot_df['is_OT_value_error'] > 0]
    else:
        CO2_value_error_df = plot_df[plot_df['is_CO2_value_error'] > 0]
    print(missing_df)
    for ax_label in "ABCDEFGHIJKLMNOP":
        fax[ax_label].axvspan(plot_df.loc[0, datetime_label].to_pydatetime(), validation_start,
                              color="black", alpha=0.15)
        fax[ax_label].axvspan(validation_start, evaluation_start, color="tab:olive", alpha=0.15)
        fax[ax_label].axvline(validation_start, ls=":", color="black")
        fax[ax_label].axvline(evaluation_start, ls=":", color="black")
        if len(missing_df) > 0:
            for idx in missing_df.index:
                fax[ax_label].axvline(missing_df.loc[idx, datetime_label].to_pydatetime(),
                                      alpha=0.5, color="tab:purple")
        if 'SWDR_value_error' in plot_df.columns:
            if len(SWDR_value_error_df) > 0:
                for idx in SWDR_value_error_df.index:
                    fax[ax_label].axvline(SWDR_value_error_df.loc[idx, datetime_label].to_pydatetime(),
                                          alpha=0.5, color="tab:pink")
        if len(wv_value_error) > 0:
            for idx in wv_value_error.index:
                fax[ax_label].axvline(wv_value_error.loc[idx, datetime_label].to_pydatetime(),
                                      alpha=0.5, color="tab:pink")
        if len(mPAR_value_error_df) > 0:
            for idx in mPAR_value_error_df.index:
                fax[ax_label].axvline(mPAR_value_error_df.loc[idx, datetime_label].to_pydatetime(),
                                      alpha=0.5, color="tab:pink")
        if len(CO2_value_error_df) > 0:
            for idx in CO2_value_error_df.index:
                fax[ax_label].axvline(CO2_value_error_df.loc[idx, datetime_label].to_pydatetime(),
                                      alpha=0.5, color="tab:pink")
    print("finish plotting vertical lines")
    fax['A'].set_title("Temperature features")
    plot_df.set_index(datetime_label)[['T (degC)', 'Tdew (degC)', 'Tlog (degC)']].plot(ax=fax['A'])
    print("finish plotting A")
    fax['B'].set_title("Temperature log")
    plot_df.set_index(datetime_label)[['Tpot (K)']].plot(ax=fax['B'], alpha=0.5)
    print("finish plotting B")
    fax['C'].set_title("Atmospheric Pressure")
    plot_df.set_index(datetime_label)[['p (mbar)']].plot(ax=fax['C'], alpha=0.5)
    print("finish plotting C")
    fax['D'].set_title("Wator Vapor pressure features")
    plot_df.set_index(datetime_label)[['VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)']].plot(ax=fax['D'], alpha=0.5)
    print("finish plotting D")
    fax['E'].set_title("Relative Humidity")
    plot_df.set_index(datetime_label)[['rh (%)']].plot(ax=fax['E'])
    print("finish plotting E")
    fax['F'].set_title("Specific Humidity")
    plot_df.set_index(datetime_label)[['sh (g/kg)']].plot(ax=fax['F'])
    print("finish plotting F")
    fax['G'].set_title("Water vapor concentration")
    plot_df.set_index(datetime_label)[['H2OC (mmol/mol)']].plot(ax=fax['G'])
    print("finish plotting G")
    fax['H'].set_title("Air Density")
    plot_df.set_index(datetime_label)[['rho (g/m**3)']].plot(ax=fax['H'])
    print("finish plotting H")
    fax['I'].set_title("Wind Velocity")
    plot_df.set_index(datetime_label)[['wv (m/s)', 'max. wv (m/s)']].plot(ax=fax['I'], alpha=0.5)
    print("finish plotting I")
    fax['J'].set_title("Wind Direction")
    plot_df.set_index(datetime_label)[['wd (deg)']].plot(ax=fax['J'], alpha=0.5)
    print("finish plotting J")
    fax['K'].set_title("Rain quantity")
    plot_df.set_index(datetime_label)[['rain (mm)']].plot(ax=fax['K'], alpha=0.5)
    print("finish plotting K")
    fax['L'].set_title("Rain duration")
    plot_df.set_index(datetime_label)[['raining (s)']].plot(ax=fax['L'], alpha=0.5)
    print("finish plotting L")
    fax['M'].set_title("Photosynthetically active radiation")
    plot_df.set_index(datetime_label)[['PAR (micromol/m**2/s)']].plot(ax=fax['M'])
    print("finish plotting M")
    fax['N'].set_title("Photosynthetically active radiation")
    plot_df.set_index(datetime_label)[['max. PAR (micromol/m**2/s)']].plot(ax=fax['N'])
    print("finish plotting N")
    fax['O'].set_title("Surface Shortwave Downward Radiation")
    plot_df.set_index(datetime_label)[['SWDR (W/m**2)']].plot(ax=fax['O'])
    print("finish plotting O")
    fax['P'].set_title("CO2 Concentration")
    if raw_source == 'autoformer':
        plot_df.set_index(datetime_label)[['OT']].plot(ax=fax['P'])
    else:
        plot_df.set_index(datetime_label)[['CO2 (ppm)']].plot(ax=fax['P'])
    print("finish plotting P")
    ffig.tight_layout()
    ffig.savefig(f"MPI/figures/{raw_source}_weather_{phase}_plot.png", dpi=300)
    plt.show()


def identify_ts_to_be_modified(target_row, value_error_columns_list):
    if target_row['is_ts_missing'] == 1:
        return 1
    for ve_column in value_error_columns_list:
        if target_row[ve_column] == 1:
            return 1
    return 0


def determine_column_with_value_error(source_dataframe, value_error_columns_list, value_error):
    output_df = source_dataframe.copy()
    no_value_error_column = list()
    for current_column in output_df.columns:
        if is_numeric_dtype(output_df[current_column]):
            if current_column[:8] == "max. PAR":
                error_value_column = "is_maxPAR_value_error"
            else:
                error_value_column = f'is_{current_column.split("(")[0].strip().replace(" ", "_")}_value_error'
            output_df[error_value_column] = output_df[current_column].apply(
                lambda v: check_for_value_error(v, value_error))
            if output_df[error_value_column].sum() == 0:
                no_value_error_column.append(error_value_column)
            else:
                value_error_columns_list.append(error_value_column)
    output_df = output_df.drop(columns=no_value_error_column)
    return output_df


if __name__ == '__main__':
    import os
    import argparse
    from mpi_utils import *
    from zipfile import ZipFile
    from datetime import datetime

    parser = argparse.ArgumentParser(description='MPI Weather Dataset Errors Analysis')
    parser.add_argument('--source-path', type=str, required=True,
                        help='Input the path for the parquet file created during analysis')
    args = parser.parse_args()

    source_data_path = args.source_path

    # Path for the original MPI data source
    original_mpi_path = os.path.join(source_data_path, "Original/MaxPlanckInstitute/roof/")
    original_mpi_filename_template = "mpi_roof_%d%s"

    # Path for the data provided in Autoformer repository
    autoformer_mpi_path = os.path.join(source_data_path, "Autoformer/")
    autoformer_mpi_filename = "weather.csv"

    # Rename some columns for avoiding reading format error
    column_names_dc = {'SWDR (W/m²)': 'SWDR (W/m**2)',
                       'PAR (µmol/m²/s)': 'PAR (micromol/m**2/s)',
                       'max. PAR (µmol/m²/s)': 'max. PAR (micromol/m**2/s)'}

    # Load the original dataset from Autoformer paper
    print(" ######################################################################################################### ")
    print(" ###################################### Checking Autoformer dataset ###################################### ")
    autoformer_wth_df = pd.read_csv(os.path.join(autoformer_mpi_path, autoformer_mpi_filename), header=0,
                                    encoding='utf-8')
    autoformer_wth_df = autoformer_wth_df.rename(columns=column_names_dc)
    autoformer_wth_df['date'] = pd.to_datetime(autoformer_wth_df['date'], format="%Y-%m-%d %H:%M:%S")
    print(autoformer_wth_df.info())
    print(autoformer_wth_df.describe())

    autoformer_validation_start = autoformer_wth_df.loc[int(len(autoformer_wth_df) * 0.7), 'date'].to_pydatetime()
    autoformer_evaluation_start = autoformer_wth_df.loc[int(len(autoformer_wth_df) * 0.8), 'date'].to_pydatetime()

    before_drop_duplicates_len = len(autoformer_wth_df)
    # Dataset might have duplicated row
    autoformer_wth_df = autoformer_wth_df.drop_duplicates()
    print(f"Drop_duplicates turn the dataframe from {before_drop_duplicates_len} to {len(autoformer_wth_df)}")

    # Check if there are any missing time steps
    autoformer_wth_df = check_for_missing_time_steps_and_identify(autoformer_wth_df, '10min', 'date', verbose=True)

    autoformer_value_error_columns = list()
    autoformer_wth_df = determine_column_with_value_error(autoformer_wth_df, autoformer_value_error_columns, -9999)
    plot_mpi_dataset(autoformer_wth_df, autoformer_validation_start, autoformer_evaluation_start, 'date',
                     raw_source='autoformer')

    autoformer_wth_df['is_ts_modified'] = autoformer_wth_df.apply(
        lambda row: identify_ts_to_be_modified(row, autoformer_value_error_columns), axis=1)
    autoformer_wth_df[['date', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)',
                       'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)',
                       'max. wv (m/s)', 'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR (W/m**2)',
                       'PAR (micromol/m**2/s)', 'max. PAR (micromol/m**2/s)', 'Tlog (degC)', 'OT', 'is_ts_missing'] +
                      autoformer_value_error_columns + ['is_ts_modified']].to_parquet(
        os.path.join(autoformer_mpi_path, "weather_errors_id.parquet"))

    # Load original data form the source
    print(" ######################################################################################################### ")
    print(" ####################################### Checking Original dataset ####################################### ")
    wth_df = None

    target_validation_start = datetime(2022, 1, 1, 0, 0)
    target_evaluation_start = datetime(2023, 1, 1, 0, 0)

    for year in [2020, 2021, 2022, 2023]:
        for segment in ["a", "b"]:
            file_path = os.path.join(original_mpi_path, original_mpi_filename_template % (year, segment) + ".zip")
            print(year, segment, file_path)
            with ZipFile(file_path) as zf:
                with zf.open(original_mpi_filename_template % (year, segment) + ".csv", 'r') as f:
                    read_df = pd.read_csv(f, header=0, encoding='latin')
                    read_df['Date Time'] = pd.to_datetime(read_df['Date Time'], format="%d.%m.%Y %H:%M:%S")
                    before_drop_duplicates_len = len(read_df)
                    # Dataset might have duplicated row
                    read_df = read_df.drop_duplicates()
                    print(f"Drop_duplicates turn the dataframe from {before_drop_duplicates_len} to {len(read_df)}")
                    # Check if there are any missing time steps
                    read_df = check_for_missing_time_steps_and_identify(read_df, '10min', 'Date Time', verbose=True)

                    # read_df['is_replaced_value_error'] = read_df.apply(
                    #     lambda row: check_for_missing_data_replaced(row, -9999), axis=1)
                    if wth_df is None:
                        wth_df = read_df.copy()
                    else:
                        wth_df = pd.concat([wth_df, read_df])
                    read_df = None
    wth_df = wth_df.reset_index().drop(columns='index')
    wth_df = wth_df.rename(columns=column_names_dc)
    print(wth_df.info())
    print(wth_df.describe())

    wth_value_error_columns = list()
    wth_df = determine_column_with_value_error(wth_df, wth_value_error_columns, -9999)
    plot_mpi_dataset(wth_df, target_validation_start, target_evaluation_start, 'Date Time')

    wth_df['is_ts_modified'] = wth_df.apply(
        lambda row: identify_ts_to_be_modified(row, wth_value_error_columns), axis=1)
    wth_df[['Date Time', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)',
            'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)',
            'max. wv (m/s)', 'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR (W/m**2)', 'PAR (micromol/m**2/s)',
            'max. PAR (micromol/m**2/s)', 'Tlog (degC)', 'CO2 (ppm)', 'is_ts_missing'] +
           wth_value_error_columns + ['is_ts_modified']].to_parquet(
        os.path.join(original_mpi_path, "mpi_weather_errors_id.parquet"))

    # Check any existing relation between the different parameters
    # sns.pairplot(
    #     autoformer_wth_df[autoformer_wth_df['missing_data'] == 0].drop(columns=['Date Time', 'missing_data']))

    # T (degC) <-> VPmax
    # T (degC) <-> VPdef (somehow)
    # Tpot (K) <-> VPdef (somehow)
    # rh <-> VPdef (somehow)
    # rho <-> VPdef (somehow)
    # Tdew (degC) <-> VPact
    # Tdew (degC) <-> sh (close)
    # Tdew (degC) <-> H2OC
    # sh <-> H2OC
    # wv <-> max wv
    # PAR <-> SWDR (somehow)
    # PAR <-> max PAR (min boundary)
