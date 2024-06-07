from matplotlib.dates import date2num


def plot_feature_from_one_source(target_df, target_index, consecutive_count=0):
    ofig, oax = plt.subplot_mosaic("ABCD;EFGH;IJKL;MNOP", figsize=(20, 15))
    ax_labels = "ABCEFGIJKMNO"
    start_modification = target_df.loc[target_index, 'datetime']
    if consecutive_count != 0:
        end_modification = target_df.loc[target_index + consecutive_count - 1, 'datetime']
    for i in range(len(ax_labels)):
        oax[ax_labels[i]].axvline(start_modification, color='red')
        if consecutive_count != 0:
            oax[ax_labels[i]].axvline(end_modification, color='red')

    target_df.set_index('datetime')[['Visibility']].plot(ax=oax['A'])
    target_df.set_index('datetime')[['DryBulbFahrenheit']].plot(ax=oax['B'])
    target_df.set_index('datetime')[['DryBulbCelsius']].plot(ax=oax['C'])
    sns.scatterplot(data=target_df, x='DryBulbFahrenheit', y='DryBulbCelsius', hue='is_ts_modified', ax=oax['D'])
    oax['E'].sharex(oax['A'])
    target_df.set_index('datetime')[['RelativeHumidity']].plot(ax=oax['E'])
    oax['F'].sharex(oax['B'])
    target_df.set_index('datetime')[['DewPointFahrenheit']].plot(ax=oax['F'])
    oax['G'].sharex(oax['C'])
    target_df.set_index('datetime')[['DewPointCelsius']].plot(ax=oax['G'])
    sns.scatterplot(data=target_df, x='DewPointFahrenheit', y='DewPointCelsius', hue='is_ts_modified', ax=oax['H'])
    oax['I'].sharex(oax['A'])
    target_df.set_index('datetime')[['WindSpeed']].plot(ax=oax['I'])
    oax['J'].sharex(oax['B'])
    target_df.set_index('datetime')[['WetBulbFahrenheit']].plot(ax=oax['J'])
    oax['K'].sharex(oax['C'])
    target_df.set_index('datetime')[['WetBulbCelsius']].plot(ax=oax['K'])
    sns.scatterplot(data=target_df, x='WetBulbFahrenheit', y='WetBulbCelsius', hue='is_ts_modified', ax=oax['L'])
    oax['M'].sharex(oax['A'])
    target_df.set_index('datetime')[['WindDirection']].plot(ax=oax['M'])
    oax['N'].sharex(oax['B'])
    target_df.set_index('datetime')[['StationPressure']].plot(ax=oax['N'])
    oax['O'].sharex(oax['C'])
    target_df.set_index('datetime')[['Altimeter']].plot(ax=oax['O'])
    sns.scatterplot(data=target_df, x='StationPressure', y='Altimeter', hue='is_ts_modified', ax=oax['P'])
    for i in range(len(ax_labels)):
        if consecutive_count != 0:
            oax[ax_labels[i]].axvspan(start_modification, end_modification, color='red', alpha=0.5)
    plt.show()


def plot_isolated_errors(source_data, target_index, timestep_range=12):
    tmp_df = source_data.loc[target_index - timestep_range:target_index + timestep_range, :]
    plot_feature_from_one_source(tmp_df, target_index)


def plot_consecutive_errors(source_data, target_index, consecutive_count, timestep_range=12):
    tmp_df = source_data.loc[target_index-timestep_range:target_index+consecutive_count+timestep_range, :]
    print(consecutive_count)
    plot_feature_from_one_source(tmp_df, target_index, consecutive_count=consecutive_count)


def plot_feature_from_two_sources(source_data, corrected_data, target_index, consecutive_count=0,
                                  timesteps_with_no_similar_condition=None):
    tfig, tax = plt.subplot_mosaic("ABCDE;FGHIJ;KLMNO;PQRST", figsize=(20, 15))
    ax_labels = "ABCFGHKLMPQR"
    start_modification = source_data.loc[target_index, 'datetime']
    if consecutive_count != 0:
        end_modification = source_data.loc[target_index + consecutive_count - 1, 'datetime']
    for i in range(len(ax_labels)):
        tax[ax_labels[i]].axvline(start_modification, color='red')
        if consecutive_count != 0:
            tax[ax_labels[i]].axvline(end_modification, color='red')
        if timesteps_with_no_similar_condition is not None:
            for ts in timesteps_with_no_similar_condition:
                if target_index <= ts < target_index + consecutive_count:
                    tax[ax_labels[i]].axvline(source_data.loc[ts, 'datetime'], color='black', ls=":")

    tmp_df = pd.merge(source_data, corrected_data, on='datetime', suffixes=('_ori', '_cor'))
    tmp_df.set_index('datetime')[['Visibility_ori', 'Visibility_cor']].plot(ax=tax['A'])
    tmp_df.set_index('datetime')[['DryBulbFahrenheit_ori', 'DryBulbFahrenheit_cor']].plot(ax=tax['B'])
    tmp_df.set_index('datetime')[['DryBulbCelsius_ori', 'DryBulbCelsius_cor']].plot(ax=tax['C'])
    sns.scatterplot(data=source_data, x='DryBulbFahrenheit', y='DryBulbCelsius', ax=tax['D'])
    tax['D'].set_title('Original')
    sns.scatterplot(data=corrected_data, x='DryBulbFahrenheit', y='DryBulbCelsius', ax=tax['E'])
    tax['E'].set_title('Corrected')
    tax['F'].sharex(tax['A'])
    tmp_df.set_index('datetime')[['RelativeHumidity_ori', 'RelativeHumidity_cor']].plot(ax=tax['F'])
    tax['G'].sharex(tax['B'])
    tmp_df.set_index('datetime')[['DewPointFahrenheit_ori', 'DewPointFahrenheit_cor']].plot(ax=tax['G'])
    tax['H'].sharex(tax['C'])
    tmp_df.set_index('datetime')[['DewPointCelsius_ori', 'DewPointCelsius_cor']].plot(ax=tax['H'])
    sns.scatterplot(data=source_data, x='DewPointFahrenheit', y='DewPointCelsius', ax=tax['I'])
    sns.scatterplot(data=corrected_data, x='DewPointFahrenheit', y='DewPointCelsius', ax=tax['J'])
    tax['K'].sharex(tax['A'])
    tmp_df.set_index('datetime')[['WindSpeed_ori', 'WindSpeed_cor']].plot(ax=tax['K'])
    tax['L'].sharex(tax['B'])
    tmp_df.set_index('datetime')[['WetBulbFahrenheit_ori', 'WetBulbFahrenheit_cor']].plot(ax=tax['L'])
    tax['M'].sharex(tax['C'])
    tmp_df.set_index('datetime')[['WetBulbCelsius_ori', 'WetBulbCelsius_cor']].plot(ax=tax['M'])
    sns.scatterplot(data=source_data, x='WetBulbFahrenheit', y='WetBulbCelsius', ax=tax['N'])
    sns.scatterplot(data=corrected_data, x='WetBulbFahrenheit', y='WetBulbCelsius', ax=tax['O'])
    tax['P'].sharex(tax['A'])
    tmp_df.set_index('datetime')[['WindDirection_ori', 'WindDirection_cor']].plot(ax=tax['P'])
    tax['Q'].sharex(tax['B'])
    tmp_df.set_index('datetime')[['StationPressure_ori', 'StationPressure_cor']].plot(ax=tax['Q'])
    tax['R'].sharex(tax['C'])
    tmp_df.set_index('datetime')[['Altimeter_ori', 'Altimeter_cor']].plot(ax=tax['R'])
    sns.scatterplot(data=source_data, x='StationPressure', y='Altimeter', ax=tax['S'])
    sns.scatterplot(data=corrected_data, x='StationPressure', y='Altimeter', ax=tax['T'])
    plt.show()
    for i in range(len(ax_labels)):
        if consecutive_count != 0:
            tax[ax_labels[i]].axvspan(start_modification.to_pydatetime(), end_modification.to_pydatetime(),
                                      color='red', alpha=0.5)
    plt.show()


def plot_isolated_errors_and_corrections(source_data, corrected_data, target_index, timestep_range=12):
    ori_tmp_df = source_data.loc[target_index-timestep_range:target_index+timestep_range, :]
    cor_tmp_df = corrected_data.loc[target_index-timestep_range:target_index+timestep_range, :]
    plot_feature_from_two_sources(ori_tmp_df, cor_tmp_df, target_index, consecutive_count=0)


def plot_consecutive_errors_and_corrections(source_data, corrected_data, target_index, consecutive_count,
                                            timestep_range=12, timesteps_with_no_similar_condition=None):
    ori_tmp_df = source_data.loc[target_index-timestep_range:target_index+consecutive_count+timestep_range, :]
    cor_tmp_df = corrected_data.loc[target_index-timestep_range:target_index+consecutive_count+timestep_range, :]
    plot_feature_from_two_sources(ori_tmp_df, cor_tmp_df, target_index, consecutive_count=consecutive_count,
                                  timesteps_with_no_similar_condition=None)


def find_similar_month_values(considered_dataframe, target_row, selection_type='mean'):
    tmp_df = considered_dataframe[(considered_dataframe['month'] == target_row['month']) &
                                  (considered_dataframe['hour'] == target_row['hour'])]
    print(f" Similar Visibility: {len(tmp_df[tmp_df['Visibility'] == target_row['Visibility']])}")
    print(f" Similar WindSpeed: {len(tmp_df[tmp_df['WindSpeed'] == target_row['WindSpeed']])}")
    print(f" Similar WindDirection: {len(tmp_df[tmp_df['WindDirection'] == target_row['WindDirection']])}")
    print(f" Similar V + WS: ",
          (len(tmp_df[(tmp_df['Visibility'] == target_row['Visibility']) &
                      (tmp_df['WindSpeed'] == target_row['WindSpeed'])])))
    print(f" Similar V + WD: ",
          (len(tmp_df[(tmp_df['Visibility'] == target_row['Visibility']) &
                      (tmp_df['WindDirection'] == target_row['WindDirection'])])))
    print(f" Similar WS + WD: ",
          (len(tmp_df[(tmp_df['WindSpeed'] == target_row['WindSpeed']) &
                      (tmp_df['WindDirection'] == target_row['WindDirection'])])))
    print(f" Similar V + WS + WD: ",
          (len(tmp_df[(tmp_df['Visibility'] == target_row['Visibility']) &
                      (tmp_df['WindSpeed'] == target_row['WindSpeed']) &
                      (tmp_df['WindDirection'] == target_row['WindDirection'])])))
    previous_measurements = (tmp_df[(tmp_df['Visibility'] == target_row['Visibility']) &
                                    (tmp_df['WindSpeed'] == target_row['WindSpeed']) &
                                    (tmp_df['WindDirection'] == target_row['WindDirection'])])
    if (len(previous_measurements)) >= 1:
        return previous_measurements.mean()
    else:
        previous_measurements = (tmp_df[(tmp_df['WindSpeed'] == target_row['WindSpeed']) &
                                        (tmp_df['WindDirection'] == target_row['WindDirection'])])
        if (len(previous_measurements)) >= 1:
            return previous_measurements.mean()
        else:
            return None


def plot_pressure_to_altimeter_relation(altimeter_values, pressure_values, regressor=None):
    afig, aax = plt.subplots(1, 1)
    aax.scatter(altimeter_values, pressure_values, color='blue')
    if regressor is not None:
        aax.plot(altimeter_values, regressor.predict(altimeter_values), color='red')
    aax.set_xlabel("Altimeter")
    aax.set_ylabel("Station Pressure")
    plt.show()


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


if __name__ == '__main__':
    import os
    import argparse
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from lcd_ds_analysis import conversion_fahrenheit_to_celsius, plot_fahrenheit_vs_celsius

    parser = argparse.ArgumentParser(description='Bulb Weather Dataset Errors Correction')
    parser.add_argument('--source-path', type=str, required=True,
                        help='Input the path for the parquet file created during analysis')
    args = parser.parse_args()

    # source_data_path = "E:/SourceData/TransformerPapers/"
    source_data_path = args.source_path
    identification_wth_filename = "lcd_errors_id.parquet"

    lcd_df = pd.read_parquet(os.path.join(source_data_path, identification_wth_filename))
    print(lcd_df)
    # Create a DataFrame where correction will be applied
    corrected_lcd_df = lcd_df.copy()

    # Define relation between altimeter and pressure
    valid_df = lcd_df[lcd_df['is_ts_modified'] == 0]
    altimeter_to_pressure_regr = LinearRegression()
    train_x = valid_df['Altimeter'].values.reshape(len(valid_df), 1)
    train_y = valid_df['StationPressure'].values.reshape(len(valid_df), 1)
    altimeter_to_pressure_regr.fit(train_x, train_y)

    plot_pressure_to_altimeter_relation(train_x, train_y, regressor=altimeter_to_pressure_regr)

    # ################################# Correct 32F errors ###################################################
    # ## Check for chunks of errors
    chunks_32f = determine_chunks_value(lcd_df['32F_errors'], target_value=1)
    print(chunks_32f)
    # There are only isolated errors
    for k, v in chunks_32f.items():
        # plot_isolated_errors(bwth_df, k)
        if lcd_df.loc[k, 'pressure_relation_errors'] == 1:
            print("pressure errors")
            plot_isolated_errors(lcd_df, k)
        corrected_lcd_df.loc[k, 'WetBulbCelsius'] = conversion_fahrenheit_to_celsius(
            corrected_lcd_df.loc[k, 'WetBulbFahrenheit'], is_rounded=True)

    # Check the conversion affine function
    comp32_fig, cax = plt.subplots(1, 2)
    sns.scatterplot(data=lcd_df, x='WetBulbFahrenheit', y='WetBulbCelsius', ax=cax[0])
    sns.scatterplot(data=corrected_lcd_df, x='WetBulbFahrenheit', y='WetBulbCelsius', ax=cax[1])
    plt.show()

    # ################################# Correct WetBulb errors ###################################################
    # Correct other conversion errors for wet bulb
    # ## Check for chunks of errors
    chunks_wfc = determine_chunks_value(lcd_df['wet_conversion_errors'], target_value=1)
    features_to_correct = ['WetBulbFahrenheit', 'RelativeHumidity', 'WetBulbCelsius']
    print(chunks_wfc)
    # There are some isolated errors but 10 chunks of consecutive errors. Especially the last chunks has 24 consecutive
    for k, v in chunks_wfc.items():
        if v == 1:
            # Isolated errors
            # plot_isolated_errors(bwth_df, k)
            for ftc in features_to_correct:
                corrected_lcd_df.loc[k, ftc] = np.nan
            if lcd_df.loc[k, 'pressure_relation_errors'] == 1:
                corrected_lcd_df.loc[k, 'StationPressure'] = np.nan
                corrected_lcd_df.loc[k, 'Altimeter'] = np.nan
    corrected_lcd_df = corrected_lcd_df.interpolate(method='linear')
    # Plots to check the effect of correction of isolated errors
    # for k, v in chunks_wfc.items():
    #     if v == 1:
    #         plot_isolated_errors_and_corrections(bwth_df, cbwth_df, k)

    for k, v in chunks_wfc.items():
        if v > 1:
            # Consecutive errors
            # plot_consecutive_errors(bwth_df, k, v)
            for idx in range(v):
                for ftc in features_to_correct:
                    corrected_lcd_df.loc[k + idx, ftc] = np.nan
                if lcd_df.loc[k + idx, 'pressure_relation_errors'] == 1:
                    expected_pressure = altimeter_to_pressure_regr.predict(
                        corrected_lcd_df.loc[k + idx, 'Altimeter'].reshape(1, 1))
                    corrected_lcd_df.loc[k + idx, 'StationPressure'] = expected_pressure[0][0]

    corrected_lcd_df = corrected_lcd_df.interpolate(method='linear')
    # Plots to check the effect of correction of isolated errors
    # for k, v in chunks_wfc.items():
    #     if v > 1:
    #         plot_consecutive_errors_and_corrections(bwth_df, cbwth_df, k, v)

    # ################################# Correct Common errors ###################################################
    # Replace common conversion errors
    # ## Check for chunks of errors
    chunks_cfc = determine_chunks_value(lcd_df['common_conversion_errors'], target_value=1)
    features_to_correct = ['DryBulbFahrenheit', 'DryBulbCelsius', 'DewPointFahrenheit', 'DewPointCelsius',
                           'WetBulbFahrenheit', 'RelativeHumidity', 'WetBulbCelsius']
    print(chunks_cfc)
    # There are some isolated errors but 10 chunks of consecutive errors
    for k, v in chunks_cfc.items():
        if v == 1:
            # Isolated errors
            # plot_isolated_errors(bwth_df, k)
            for ftc in features_to_correct:
                corrected_lcd_df.loc[k, ftc] = np.nan
            # if bwth_df.loc[k, 'pressure_relation_errors'] == 1:
            #     cbwth_df.loc[k, 'StationPressure'] = np.nan
            #     cbwth_df.loc[k, 'Altimeter'] = np.nan
    corrected_lcd_df = corrected_lcd_df.interpolate(method='linear')
    # Plots to check the effect of correction of isolated errors
    # for k, v in chunks_cfc.items():
    #     if v == 1:
    #         plot_isolated_errors_and_corrections(bwth_df, cbwth_df, k)

    temporal_df = corrected_lcd_df.copy()
    temporal_df['hour'] = temporal_df.datetime.apply(lambda x: x.hour)
    temporal_df['month'] = temporal_df.datetime.apply(lambda x: x.month)

    idx_no_previous_measurements = list()

    for k, v in chunks_cfc.items():
        if v > 1:
            # Consecutive errors
            for idx in range(v):
                pm_row = find_similar_month_values(temporal_df[temporal_df['is_ts_modified'] == 0],
                                                   temporal_df.loc[k+idx, :])
                if pm_row is None:
                    # No previous similar configuration
                    idx_no_previous_measurements.append(k+idx)
                    for ftc in features_to_correct:
                        corrected_lcd_df.loc[k + idx, ftc] = np.nan
                else:
                    for ftc in features_to_correct:
                        if ftc == 'WetBulbCelsius':
                            corrected_lcd_df.loc[k + idx, ftc] = pm_row[ftc]
                        else:
                            corrected_lcd_df.loc[k + idx, ftc] = round(pm_row[ftc])
            #     if bwth_df.loc[k+idx, 'pressure_relation_errors'] == 1:
            #         expected_pressure = altimeter_to_pressure_regr.predict(
            #             cbwth_df.loc[k+idx, 'Altimeter'].reshape(1, 1))
            #         cbwth_df.loc[k+idx, 'StationPressure'] = expected_pressure[0][0]
            # plot_consecutive_errors(bwth_df, k, v)

    corrected_lcd_df = corrected_lcd_df.interpolate(method='linear')

    # print(idx_no_previous_measurements)
    # # Plots to check the effect of correction of isolated errors
    # for k, v in chunks_cfc.items():
    #     if v > 1:
    #         print(k)
    #         plot_consecutive_errors_and_corrections(bwth_df, cbwth_df, k, v,
    #                                                 timesteps_with_no_similar_condition=idx_no_previous_measurements)

    # interpolate between integer will result in float number. But some columns are only int so round these values

    for column in ["DryBulbFahrenheit", "DryBulbCelsius", "WetBulbFahrenheit", "DewPointFahrenheit",
                   "DewPointCelsius", "RelativeHumidity"]:
        corrected_lcd_df[column] = corrected_lcd_df[column].apply(lambda x: round(x))
        corrected_lcd_df[column] = pd.to_numeric(corrected_lcd_df[column], downcast='integer')

    # Other columns have only 1 or to digits so round them to be consistent
    corrected_lcd_df["WetBulbCelsius"] = corrected_lcd_df["WetBulbCelsius"].apply(lambda x: round(x, 1))
    corrected_lcd_df["StationPressure"] = corrected_lcd_df["StationPressure"].apply(lambda x: round(x, 2))

    print(corrected_lcd_df.info())

    # Create new target WetBulbCelsius as Integer and Real WetBulbCelsius from conversion from Fahrenheit

    corrected_lcd_df['RealWetBulbCelsius'] = corrected_lcd_df["WetBulbFahrenheit"].apply(
        lambda x: conversion_fahrenheit_to_celsius(x, is_rounded=True))

    # Plot some relation before correction
    lcd_df['WetBulbCelsiusInt'] = lcd_df['WetBulbCelsius'].apply(lambda x: round(x))
    plot_fahrenheit_vs_celsius(lcd_df)
    plot_pressure_to_altimeter_relation(lcd_df['Altimeter'].values.reshape(-1, 1),
                                        lcd_df['StationPressure'].values.reshape(-1, 1),
                                        regressor=altimeter_to_pressure_regr)

    # Plot some relation after correction
    corrected_lcd_df['WetBulbCelsiusInt'] = corrected_lcd_df['WetBulbCelsius'].apply(lambda x: round(x))
    plot_fahrenheit_vs_celsius(corrected_lcd_df)
    plot_pressure_to_altimeter_relation(corrected_lcd_df['Altimeter'].values.reshape(-1, 1),
                                        corrected_lcd_df['StationPressure'].values.reshape(-1, 1),
                                        regressor=altimeter_to_pressure_regr)

    # Plot relative humidity before and after correction
    fig, ax = plt.subplots(1, 1)
    lcd_df.set_index('datetime')[['RelativeHumidity']].plot(ax=ax)
    corrected_lcd_df.set_index('datetime')[['RelativeHumidity']].plot(ax=ax)
    plt.show()

    main_columns = ['datetime', 'Visibility', 'DryBulbFahrenheit', 'DryBulbCelsius', 'WetBulbFahrenheit',
                    'DewPointFahrenheit', 'DewPointCelsius', 'RelativeHumidity', 'WindSpeed', 'WindDirection',
                    'StationPressure', 'Altimeter']
    errors_ids_columns = ['32F_errors', 'common_conversion_errors', 'wet_conversion_errors', 'pressure_relation_errors',
                          'is_ts_missing', 'is_ts_modified']

    corrected_lcd_df.to_parquet(os.path.join(source_data_path, "corrected_bulb_weather.parquet"))
    corrected_lcd_df[main_columns + ['WetBulbCelsius'] + errors_ids_columns].rename(
        columns={'datetime': 'date'}).to_csv(os.path.join("revised_datasets/", "LCDWf_1H_4Y_USUNK.csv"), index=False)
    corrected_lcd_df[main_columns + ['WetBulbCelsiusInt'] + errors_ids_columns].rename(
        columns={'datetime': 'date'}).to_csv(os.path.join("revised_datasets/", "LCDWi_1H_4Y_USUNK.csv"), index=False)
    corrected_lcd_df[main_columns + ['RealWetBulbCelsius'] + errors_ids_columns].rename(
        columns={'datetime': 'date'}).to_csv(os.path.join("revised_datasets/", "LCDWr_1H_4Y_USUNK.csv"), index=False)
