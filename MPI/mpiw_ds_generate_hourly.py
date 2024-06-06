import datetime

import pandas as pd

if __name__ == '__main__':
    import os
    import argparse
    from datetime import timedelta

    parser = argparse.ArgumentParser(description='MPI Weather Dataset Transformation to Hourly dataset')
    parser.add_argument('--source-path', type=str, required=True,
                        help='Input the path for the parquet file created during analysis')
    args = parser.parse_args()

    source_data_path = args.source_path

    mpi_wth_minute_df = pd.read_csv(os.path.join("revised_datasets/", "MPIW_10T_4Y_R.csv"))
    mpi_wth_minute_df['Date Time'] = pd.to_datetime(mpi_wth_minute_df['Date Time'])
    print(mpi_wth_minute_df)

    # Data from at time t are data collected from t-10min and t. If minute of t are 00, it then means that data were
    # collected from the preceding hour. Therefore, shift all date to 5min in order to better illustrate this.

    mpi_wth_minute_df['Date Time'] = mpi_wth_minute_df['Date Time'].apply(lambda t: t - timedelta(seconds=5*60))
    print(mpi_wth_minute_df)
    print(list(mpi_wth_minute_df.columns))

    mean_features = ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)',
                     'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'wd (deg)',
                     'Tlog (degC)', 'CO2 (ppm)', 'date', 'hour']
    max_features = ['max. PAR (micromol/m**2/s)', 'max. wv (m/s)', 'date', 'hour']
    sum_features = ['rain (mm)', 'raining (s)', 'SWDR (W/m**2)', 'PAR (micromol/m**2/s)', 'date', 'hour',
                    'is_ts_missing', 'is_wv_value_error', 'is_SWDR_value_error', 'is_maxPAR_value_error',
                    'is_CO2_value_error', 'is_ts_modified']

    hourly_df = mpi_wth_minute_df.copy()
    hourly_df['date'] = hourly_df['Date Time'].apply(lambda x: x.date())
    hourly_df['hour'] = hourly_df['Date Time'].apply(lambda x: x.hour)
    hourly_mean_df = hourly_df.drop(columns='Date Time').groupby(['date', 'hour']).mean().reset_index()
    hourly_sum_df = hourly_df.drop(columns='Date Time').groupby(['date', 'hour']).sum().reset_index()
    hourly_max_df = hourly_df.drop(columns='Date Time').groupby(['date', 'hour']).max().reset_index()
    mpi_hourly_df = pd.merge(hourly_mean_df[mean_features], hourly_sum_df[sum_features], on=['date', 'hour'],
                             how='left')
    mpi_hourly_df = pd.merge(mpi_hourly_df, hourly_max_df[max_features], on=['date', 'hour'], how='left')
    mpi_hourly_df['ts'] = mpi_hourly_df.apply(lambda row: f"{row['date']} {row['hour']}:00", axis=1)
    mpi_hourly_df['datetime'] = pd.to_datetime(mpi_hourly_df.ts)
    mpi_wth_hourly_df = mpi_hourly_df[['datetime'] + list(mpi_wth_minute_df.columns[1:])].copy()
    for column in mean_features:
        mpi_wth_hourly_df[column] = mpi_hourly_df[column].round(2)
    print(mpi_wth_hourly_df)

    mpi_wth_hourly_df.drop(columns=['date', 'hour']).to_csv(os.path.join(
        "revised_datasets/", "MPIW_1H_4Y_R.csv"), index=False)

    print(mpi_wth_hourly_df[mpi_wth_hourly_df['datetime'] == datetime.datetime(2022, 1, 1, 0, 0)])
    print(mpi_wth_hourly_df[mpi_wth_hourly_df['datetime'] == datetime.datetime(2023, 1, 1, 0, 0)])
