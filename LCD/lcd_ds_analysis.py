import seaborn as sns
import matplotlib.pyplot as plt


def conversion_fahrenheit_to_celsius(target_fahrenheit_value, is_rounded=False):
    celsius_value = (target_fahrenheit_value - 32) * 5 / 9
    if is_rounded:
        return round(celsius_value, 1)
    else:
        return celsius_value


def check_for_32f_errors(target_row):
    if target_row['WetBulbFahrenheit'] == 32:
        if target_row['WetBulb_wrong_conversion'] == 1:
            return 1
    return 0


def check_for_common_conversion_errors(target_row):
    if target_row['DryBulb_wrong_conversion'] == 1:
        if target_row['WetBulb_wrong_conversion'] == 1:
            if target_row['DewPoint_wrong_conversion'] == 1:
                return 1
    return 0


def check_for_other_conversion_errors(target_row):
    if target_row['WetBulb_wrong_conversion'] == 1:
        if target_row['32F_errors'] == 0:
            if target_row['common_conversion_errors'] == 0:
                return 1
    return 0


def identify_ts_to_be_modified(target_row):
    if (target_row['32F_errors'] == 1 or target_row['common_conversion_errors'] == 1 or
            target_row['wet_conversion_errors'] == 1 or target_row['pressure_relation_errors'] == 1):
        return 1
    return 0


def plot_fahrenheit_vs_celsius(source_data, target_columns=None):
    vsfig, vax = plt.subplots(1, 1)
    if target_columns is None:
        tgt_cols = ['WetBulb', 'DryBulb', 'DewPoint']
    else:
        tgt_cols = target_columns
    if 'WetBulb' in tgt_cols:
        vax.scatter(source_data['WetBulbFahrenheit'].values, source_data['WetBulbCelsius'].values,
                    label='WetBulb', color='tab:olive', alpha=0.25, marker='o')
        vax.scatter(source_data['WetBulbFahrenheit'].values, source_data['WetBulbCelsiusInt'].values,
                    label='WetBulbInt', color='tab:red', alpha=0.25, marker='o')
    if 'DryBulb' in tgt_cols:
        vax.scatter(source_data['DryBulbFahrenheit'].values, source_data['DryBulbCelsius'].values,
                    label='DryBulb', color='tab:orange', alpha=0.25, marker='x')
    if 'DewPoint' in tgt_cols:
        vax.scatter(source_data['DewPointFahrenheit'].values, source_data['DewPointCelsius'].values,
                    label='DewPoint', color='tab:cyan', alpha=0.25, marker='+')
    plt.show()


def check_temperature_features_correlation(source_df, target_hue):
    ffig, fax = plt.subplot_mosaic("AB;CD;EF;GG", figsize=(20, 15))
    f_df = source_df[source_df['is_ts_modified'] == 0]
    sns.scatterplot(data=f_df, x='DryBulbFahrenheit', y='WetBulbFahrenheit', hue=target_hue, ax=fax['A'])
    fax['A'].set_title("Fahrenheit")
    fax['A'].set_ylabel("WetBulb")
    fax['A'].set_xlabel("DryBulb")
    sns.scatterplot(data=f_df, x='WetBulbCelsius', y='DryBulbCelsius', hue=target_hue, ax=fax['B'])
    fax['B'].set_title("Celsius")
    fax['B'].set_ylabel("WetBulb")
    fax['B'].set_xlabel("DryBulb")
    sns.scatterplot(data=f_df, x='WetBulbFahrenheit', y='DewPointFahrenheit', hue=target_hue, ax=fax['C'])
    fax['C'].set_ylabel("WetBulb")
    fax['C'].set_xlabel("DewPoint")
    sns.scatterplot(data=f_df, x='WetBulbCelsius', y='DewPointCelsius', hue=target_hue, ax=fax['D'])
    fax['D'].set_ylabel("WetBulb")
    fax['D'].set_xlabel("DewPoint")
    sns.scatterplot(data=f_df, x='DryBulbFahrenheit', y='DewPointFahrenheit', hue=target_hue, ax=fax['E'])
    fax['E'].set_ylabel("DryBulb")
    fax['E'].set_xlabel("DewPoint")
    sns.scatterplot(data=f_df, x='DryBulbCelsius', y='DewPointCelsius', hue=target_hue, ax=fax['F'])
    fax['F'].set_ylabel("DryBulb")
    fax['F'].set_xlabel("DewPoint")
    sns.scatterplot(data=f_df, x='StationPressure', y='Altimeter', hue=target_hue, ax=fax['G'])
    plt.show()


# wth_df fahrenheit_celsius_columns 'datetime' dataset_conversion_range_std
def plot_indicator_diff(source_data, considered_columns, error_std_threshold, validation_start, evaluation_start,
                        datetime_label='datetime'):
    # Add axvline for end of training and end of validation to identify where the errors occurs the most
    fig, ax = plt.subplots(4, 1, sharex='col', figsize=(20, 15))
    for e, fcc in enumerate(considered_columns):
        ax[e].set_title(fcc)
        ax[e].axvspan(source_data.loc[0, datetime_label].to_pydatetime(), validation_start, color="k", alpha=0.15)
        ax[e].axvspan(validation_start, evaluation_start, color="tab:olive", alpha=0.15)
        ax[e].axvline(validation_start, ls=":", color="black")
        ax[e].axvline(evaluation_start, ls=":", color="black")
        if fcc == "WetBulb":
            err_df = source_data[source_data['32F_errors'] == 1]
            for ts in err_df.index:
                ax[e].axvline(err_df.loc[ts, datetime_label].to_pydatetime(), color='tab:red', alpha=0.5)
            err_df = source_data[source_data['wet_conversion_errors'] == 1]
            for ts in err_df.index:
                ax[e].axvline(err_df.loc[ts, datetime_label].to_pydatetime(), color='tab:pink', alpha=0.5)

        err_df = source_data[source_data['common_conversion_errors'] == 1]
        for ts in err_df.index:
            ax[e].axvline(err_df.loc[ts, datetime_label].to_pydatetime(), color='tab:purple', alpha=0.5)
        source_data.set_index(datetime_label)[fcc + 'Celsius_diff'].plot(ax=ax[e])
        ax[e].axhline(error_std_threshold, color='orange')
        ax[e].axhline(-error_std_threshold, color='orange')
    ax[3].set_title("Pressure to Altimeter relation errors")
    ax[3].axvspan(source_data.loc[0, datetime_label].to_pydatetime(), validation_start, color="k", alpha=0.15)
    ax[3].axvspan(validation_start, evaluation_start, color="tab:olive", alpha=0.15)
    ax[3].axvline(validation_start, ls=":", color="black")
    ax[3].axvline(evaluation_start, ls=":", color="black")
    err_df = source_data[source_data['pressure_relation_errors'] == 1]
    for ts in err_df.index:
        ax[3].axvline(err_df.loc[ts, datetime_label].to_pydatetime(), color='tab:brown', alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"LCD/figures/lcd_weather_diff_plot.png", dpi=300)
    plt.show()


def plot_lcd_dataset(source_data, validation_start, evaluation_start, datetime_label, raw_source='lcd', phase='analysis'):
    plot_df = source_data.copy()
    ffig, fax = plt.subplot_mosaic("AB;CD;EF;GH;IJ", sharex=True, figsize=(20, 15))
    missing_df = plot_df[plot_df['is_ts_missing'] > 0]
    ttf_error_df = plot_df[plot_df['32F_errors'] > 0]
    common_conv_error_df = plot_df[plot_df['common_conversion_errors'] > 0]
    wetbulb_conv_error_df = plot_df[plot_df['wet_conversion_errors'] > 0]
    pressure_relation_error_df = plot_df[plot_df['pressure_relation_errors'] > 0]
    for ax_label in "ABCDEFGHIJ":
        fax[ax_label].axvspan(plot_df.loc[0, datetime_label].to_pydatetime(), validation_start, color="k", alpha=0.15)
        fax[ax_label].axvspan(validation_start, evaluation_start, color="tab:olive", alpha=0.15)
        fax[ax_label].axvline(validation_start, ls=":", color="black")
        fax[ax_label].axvline(evaluation_start, ls=":", color="black")
        if len(missing_df) > 0:
            print(missing_df)
            for idx in missing_df.index:
                fax[ax_label].axvline(missing_df.loc[idx, datetime_label].to_pydatetime(),
                                      alpha=0.5, color="tab:purple")
        if len(ttf_error_df) > 0:
            for idx in ttf_error_df.index:
                fax[ax_label].axvline(ttf_error_df.loc[idx, datetime_label].to_pydatetime(),
                                      alpha=0.5, color="tab:red")
        if len(common_conv_error_df) > 0:
            for idx in common_conv_error_df.index:
                fax[ax_label].axvline(common_conv_error_df.loc[idx, datetime_label].to_pydatetime(),
                                      alpha=0.5, color="tab:purple")
        if len(wetbulb_conv_error_df) > 0:
            for idx in wetbulb_conv_error_df.index:
                fax[ax_label].axvline(wetbulb_conv_error_df.loc[idx, datetime_label].to_pydatetime(),
                                      alpha=0.5, color="tab:pink")
        if len(pressure_relation_error_df) > 0:
            for idx in pressure_relation_error_df.index:
                fax[ax_label].axvline(pressure_relation_error_df.loc[idx, datetime_label].to_pydatetime(),
                                      alpha=0.5, color="tab:brown")
    fax['A'].set_title("Visibility")
    plot_df.set_index(datetime_label)[['Visibility']].plot(ax=fax['A'])
    fax['B'].set_title("Relative Humidity")
    plot_df.set_index(datetime_label)[['RelativeHumidity']].plot(ax=fax['B'])
    fax['C'].set_title("Dry Bulb")
    plot_df.set_index(datetime_label)[['DryBulbCelsius', 'DryBulbFahrenheit']].plot(ax=fax['C'], alpha=0.5)
    fax['D'].set_title("Dew Point")
    plot_df.set_index(datetime_label)[['DewPointCelsius', 'DewPointFahrenheit']].plot(ax=fax['D'], alpha=0.5)
    fax['E'].set_title("Wind Speed")
    plot_df.set_index(datetime_label)[['WindSpeed']].plot(ax=fax['E'])
    fax['F'].set_title("Wind Direction")
    plot_df.set_index(datetime_label)[['WindDirection']].plot(ax=fax['F'])
    fax['G'].set_title("Station Pressure")
    plot_df.set_index(datetime_label)[['StationPressure']].plot(ax=fax['G'])
    fax['H'].set_title("Altimeter")
    plot_df.set_index(datetime_label)[['Altimeter']].plot(ax=fax['H'])
    fax['I'].set_title("Wet Bulb Fahrenheit")
    plot_df.set_index(datetime_label)[['WetBulbFahrenheit']].plot(ax=fax['I'])
    fax['J'].set_title("Wet Bulb Celsius")
    plot_df.set_index(datetime_label)[['WetBulbCelsius']].plot(ax=fax['J'])
    ffig.tight_layout()
    ffig.savefig(f"LCD/figures/{raw_source}_weather_{phase}_plot.png", dpi=300)
    plt.show()


def plot_lcd_ds_and_relation(source_data, validation_start, evaluation_start, datetime_label, raw_source='lcd',
                             phase='analysis'):
    plot_df = source_data.copy()
    ffig, fax = plt.subplot_mosaic("ABC;DEF;GHI;JKL;MNO", figsize=(20, 15))
    missing_df = plot_df[plot_df['is_ts_missing'] > 0]
    ttf_error_df = plot_df[plot_df['32F_errors'] > 0]
    common_conv_error_df = plot_df[plot_df['common_conversion_errors'] > 0]
    wetbulb_conv_error_df = plot_df[plot_df['wet_conversion_errors'] > 0]
    pressure_relation_error_df = plot_df[plot_df['pressure_relation_errors'] > 0]
    for ax_label in "ABCDEGHJKMN":
        fax[ax_label].axvspan(plot_df.loc[0, datetime_label].to_pydatetime(), validation_start, color="k", alpha=0.15)
        fax[ax_label].axvspan(validation_start, evaluation_start, color="tab:olive", alpha=0.15)
        fax[ax_label].axvline(validation_start, ls=":", color="black")
        fax[ax_label].axvline(evaluation_start, ls=":", color="black")
        if len(missing_df) > 0:
            print(missing_df)
            for idx in missing_df.index:
                fax[ax_label].axvline(missing_df.loc[idx, datetime_label].to_pydatetime(),
                                      alpha=0.5, color="tab:purple")
        if len(ttf_error_df) > 0:
            for idx in ttf_error_df.index:
                fax[ax_label].axvline(ttf_error_df.loc[idx, datetime_label].to_pydatetime(),
                                      alpha=0.5, color="tab:red")
        if len(common_conv_error_df) > 0:
            for idx in common_conv_error_df.index:
                fax[ax_label].axvline(common_conv_error_df.loc[idx, datetime_label].to_pydatetime(),
                                      alpha=0.5, color="tab:purple")
        if len(wetbulb_conv_error_df) > 0:
            for idx in wetbulb_conv_error_df.index:
                fax[ax_label].axvline(wetbulb_conv_error_df.loc[idx, datetime_label].to_pydatetime(),
                                      alpha=0.5, color="tab:pink")
        if len(pressure_relation_error_df) > 0:
            for idx in pressure_relation_error_df.index:
                fax[ax_label].axvline(pressure_relation_error_df.loc[idx, datetime_label].to_pydatetime(),
                                      alpha=0.5, color="tab:brown")
    fax['A'].set_title("Visibility")
    plot_df.set_index(datetime_label)[['Visibility']].plot(ax=fax['A'])
    fax['B'].set_title("Relative Humidity")
    plot_df.set_index(datetime_label)[['RelativeHumidity']].plot(ax=fax['B'])
    fax['C'].set_title("Wind Speed & Direction")
    plot_df.set_index(datetime_label)[['WindSpeed', 'WindDirection']].plot(ax=fax['C'], alpha=0.5)

    fax['D'].set_title("Dry Bulb (F)")
    plot_df.set_index(datetime_label)[['DryBulbFahrenheit']].plot(ax=fax['D'])
    fax['E'].set_title("Dry Bulb (C)")
    plot_df.set_index(datetime_label)[['DryBulbCelsius']].plot(ax=fax['E'])
    fax['F'].set_title("Dry Bulb - F vs. C")
    sns.scatterplot(data=plot_df, x='DryBulbFahrenheit', y='DryBulbCelsius', hue='common_conversion_errors', ax=fax['F'])

    fax['G'].set_title("Dew Point (F)")
    plot_df.set_index(datetime_label)[['DewPointFahrenheit']].plot(ax=fax['G'])
    fax['H'].set_title("Dew Point (C)")
    plot_df.set_index(datetime_label)[['DewPointCelsius']].plot(ax=fax['H'])
    fax['I'].set_title("Dew Point - F vs. C")
    sns.scatterplot(data=plot_df, x='DewPointFahrenheit', y='DewPointCelsius', hue='common_conversion_errors', ax=fax['I'])

    fax['J'].set_title("Station Pressure")
    plot_df.set_index(datetime_label)[['StationPressure']].plot(ax=fax['J'])
    fax['K'].set_title("Altimeter")
    plot_df.set_index(datetime_label)[['Altimeter']].plot(ax=fax['K'])
    fax['L'].set_title("Station Pressure vs. Altimeter")
    sns.scatterplot(data=plot_df, x='StationPressure', y='Altimeter', hue='pressure_relation_errors', ax=fax['L'])

    fax['M'].set_title("Wet Bulb (C)")
    plot_df.set_index(datetime_label)[['WetBulbFahrenheit']].plot(ax=fax['M'])
    fax['N'].set_title("Wet Bulb (F)")
    plot_df.set_index(datetime_label)[['WetBulbCelsius']].plot(ax=fax['N'])
    fax['O'].set_title("Wet Bulb - F vs. C")
    sns.scatterplot(data=plot_df, x='WetBulbFahrenheit', y='WetBulbCelsius', hue='wet_conversion_errors', ax=fax['O'])
    ffig.tight_layout()
    ffig.savefig(f"LCD/figures/{raw_source}_relation_{phase}_plot.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    import os
    import argparse
    import pandas as pd
    from zipfile import ZipFile
    from datetime import datetime
    from lcd_utils import check_for_missing_time_steps_and_identify

    parser = argparse.ArgumentParser(description='Bulb Weather Dataset Errors Correction')
    parser.add_argument('--source-path', type=str, required=True,
                        help='Input the path for the parquet file created during analysis')
    args = parser.parse_args()

    source_data_path = args.source_path
    original_wth_filename = "WTH.csv-20230804T085608Z-001.zip"

    target_validation_start = datetime(2012, 1, 1, 0, 0)
    target_evaluation_start = datetime(2013, 1, 1, 0, 0)

    with ZipFile(os.path.join(source_data_path, original_wth_filename)) as zf:
        with zf.open("WTH.csv", 'r') as f:
            read_df = pd.read_csv(f, header=0, delimiter=',')
            wth_df = read_df[['date', 'Visibility']].copy()
            col_int = ["DryBulbFarenheit", "DryBulbCelsius", "WetBulbFarenheit", "DewPointFarenheit",
                       "DewPointCelsius", "RelativeHumidity", "WindSpeed", "WindDirection"]
            for col in col_int:
                if col[-9:] == "Farenheit":
                    wth_df[col[:-9]+"Fahrenheit"] = pd.to_numeric(read_df[col], downcast='integer')
                else:
                    wth_df[col] = pd.to_numeric(read_df[col], downcast='integer')
            wth_df['datetime'] = pd.to_datetime(wth_df['date'])
            wth_df['hour'] = wth_df['datetime'].apply(lambda x: x.hour)
            wth_df['month'] = wth_df['datetime'].apply(lambda x: x.month)
            wth_df['year'] = wth_df['datetime'].apply(lambda x: x.year)
            wth_df['hm'] = wth_df['datetime'].apply(lambda x: f"{x.month}.{x.hour}")
            wth_df['yhm'] = wth_df['datetime'].apply(lambda x: f"{x.year}.{x.month}.{x.hour}")
            for col in ["StationPressure", "Altimeter", "WetBulbCelsius"]:
                wth_df[col] = read_df[col]

    print(wth_df.info())
    print(wth_df.describe())

    informer_validation_start = wth_df.loc[int(len(wth_df) * 0.7), 'datetime'].to_pydatetime()
    informer_evaluation_start = wth_df.loc[int(len(wth_df) * 0.8), 'datetime'].to_pydatetime()

    before_drop_duplicates_len = len(wth_df)
    # Dataset might have duplicated row
    wth_df = wth_df.drop_duplicates()
    print(f"Drop_duplicates turn the dataframe from {before_drop_duplicates_len} to {len(wth_df)}")

    # Check if there are any missing time steps
    wth_df = check_for_missing_time_steps_and_identify(wth_df, '1h', 'datetime', verbose=True)

    # Looking at conversion Fahrenheit <-> Celsius error
    fahrenheit_celsius_columns = ["DryBulb", "WetBulb", "DewPoint"]
    for fcc in fahrenheit_celsius_columns:
        wth_df['Real' + fcc + "Celsius"] = wth_df[fcc+"Fahrenheit"].apply(
            lambda x: conversion_fahrenheit_to_celsius(x, is_rounded=True))
        wth_df[fcc + 'Celsius_diff'] = wth_df['Real' + fcc + "Celsius"] - wth_df[fcc + "Celsius"]

    desc = wth_df[[x+'Celsius_diff' for x in fahrenheit_celsius_columns]].describe()

    dataset_conversion_range_std = (desc.loc['std', 'DryBulbCelsius_diff'] + desc.loc['std', 'DewPointCelsius_diff'])/2
    print(desc)
    print(dataset_conversion_range_std)

    for fcc in fahrenheit_celsius_columns:
        wth_df[fcc+'_wrong_conversion'] = wth_df[fcc + 'Celsius_diff'].apply(
            lambda x: 1 if abs(x) > dataset_conversion_range_std else 0)
        print(f"There are {len(wth_df[wth_df[fcc+'_wrong_conversion'] == 1])} conversion issues with {fcc}")

    wth_df['32F_errors'] = wth_df.apply(lambda row: check_for_32f_errors(row), axis=1)
    err_32f_df = wth_df[wth_df['32F_errors'] == 1]
    print(f"There are {len(err_32f_df)} time steps for which there are errors with 32F.")
    print(err_32f_df[['date', 'Visibility', 'DryBulbFahrenheit', 'DryBulbCelsius', 'WetBulbFahrenheit',
                      'DewPointFahrenheit', 'DewPointCelsius', 'RelativeHumidity', 'WindSpeed', 'WindDirection',
                      'StationPressure', 'Altimeter', 'WetBulbCelsius']])

    wth_df['common_conversion_errors'] = wth_df.apply(lambda row: check_for_common_conversion_errors(row), axis=1)
    err_zero_cfc_df = wth_df[wth_df['common_conversion_errors'] == 1]
    print(f"There are {len(err_zero_cfc_df)} time steps for which there are common conversion errors.")
    print(err_zero_cfc_df[['date', 'Visibility', 'DryBulbFahrenheit', 'DryBulbCelsius', 'WetBulbFahrenheit',
                           'DewPointFahrenheit', 'DewPointCelsius', 'RelativeHumidity', 'WindSpeed', 'WindDirection',
                           'StationPressure', 'Altimeter', 'WetBulbCelsius']])

    wth_df['wet_conversion_errors'] = wth_df.apply(lambda row: check_for_other_conversion_errors(row), axis=1)
    err_zero_wfc_df = wth_df[wth_df['wet_conversion_errors'] == 1]
    print(f"There are {len(err_zero_wfc_df)} time steps for which there are conversion errors for WetBulb.")
    print(err_zero_wfc_df[['WetBulbFahrenheit', 'WetBulbCelsius', 'RealWetBulbCelsius']].mean())
    print("Other WetBulb conversion errors seems to be missing values, i.e., Celsius = Fahrenheit = 0")
    print(err_zero_wfc_df[['date', 'Visibility', 'DryBulbFahrenheit', 'DryBulbCelsius', 'WetBulbFahrenheit',
                           'DewPointFahrenheit', 'DewPointCelsius', 'RelativeHumidity', 'WindSpeed', 'WindDirection',
                           'StationPressure', 'Altimeter', 'WetBulbCelsius']])

    # Looking at Altimeter <-> Station Pressure errors
    # These errors appeared for Station Pressure ==  21.478686.
    # In fact, other pressure and altimeter values have only 2 digits
    wth_df['pressure_relation_errors'] = wth_df['StationPressure'].apply(lambda x: 1 if 21.47 < x < 21.48 else 0)
    # sns.scatterplot(data=wth_df, x='StationPressure', y='Altimeter', hue='pressure_relation_errors')
    # plt.show()
    err_pre_alt_relation_df = wth_df[wth_df['pressure_relation_errors'] == 1]
    print(f"There are {len(err_pre_alt_relation_df)} time steps for which there are pressure to altimeter relation errors.")
    print(err_pre_alt_relation_df[['date', 'Visibility', 'DryBulbFahrenheit', 'DryBulbCelsius', 'WetBulbFahrenheit',
                                   'DewPointFahrenheit', 'DewPointCelsius', 'RelativeHumidity', 'WindSpeed',
                                   'WindDirection', 'StationPressure', 'Altimeter', 'WetBulbCelsius']])

    wth_df['is_ts_modified'] = wth_df.apply(lambda row: identify_ts_to_be_modified(row), axis=1)

    plot_indicator_diff(wth_df, fahrenheit_celsius_columns, dataset_conversion_range_std,
                        informer_validation_start, informer_evaluation_start, 'datetime')

    wth_df['WetBulbCelsiusInt'] = wth_df['RealWetBulbCelsius'].apply(lambda x: round(x)).astype(int)

    plot_fahrenheit_vs_celsius(wth_df)

    plot_lcd_dataset(wth_df, informer_validation_start, informer_evaluation_start, 'datetime')

    plot_lcd_ds_and_relation(wth_df, informer_validation_start, informer_evaluation_start, 'datetime')

    tgt_hue = 'hm'
    check_temperature_features_correlation(wth_df, tgt_hue)

    for column in ['32F_errors', 'common_conversion_errors', 'wet_conversion_errors',
                   'pressure_relation_errors', 'is_ts_missing', 'is_ts_modified']:
        wth_df[column] = pd.to_numeric(wth_df[column], downcast='integer')

    wth_df[['datetime', 'Visibility', 'DryBulbFahrenheit', 'DryBulbCelsius', 'WetBulbFahrenheit', 'DewPointFahrenheit',
            'DewPointCelsius', 'RelativeHumidity', 'WindSpeed', 'WindDirection', 'StationPressure', 'Altimeter',
            'WetBulbCelsius', 'RealWetBulbCelsius', '32F_errors', 'common_conversion_errors', 'wet_conversion_errors',
            'pressure_relation_errors', 'is_ts_missing', 'is_ts_modified']].to_parquet(
        os.path.join(source_data_path, "lcd_errors_id.parquet"))
