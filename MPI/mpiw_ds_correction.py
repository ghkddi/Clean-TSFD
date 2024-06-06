

def plot_feature_from_one_source(target_df, target_index, datetime_label, consecutive_count=0):
    ofig, oax = plt.subplot_mosaic("ABCDE;FGHIJ;KLMNO;PQRST;UVWXY", figsize=(20, 15))
    ax_labels = "ABCDFGHIKLMNPQRSUVWX"
    start_modification = target_df.loc[target_index, datetime_label]
    if consecutive_count != 0:
        end_modification = target_df.loc[target_index + consecutive_count - 1, datetime_label]
    for i in range(len(ax_labels)):
        oax[ax_labels[i]].axvline(start_modification, color='red')
        if consecutive_count != 0:
            oax[ax_labels[i]].axvline(end_modification, color='red')
    target_df.set_index(datetime_label)[['T (degC)']].plot(ax=oax['A'])
    target_df.set_index(datetime_label)[['Tpot (K)']].plot(ax=oax['B'])
    target_df.set_index(datetime_label)[['Tdew (degC)']].plot(ax=oax['C'])
    target_df.set_index(datetime_label)[['Tlog (degC)']].plot(ax=oax['D'])
    sns.scatterplot(data=target_df, x='T (degC)', y='VPmax (mbar)', hue='is_ts_modified', ax=oax['E'])
    oax['F'].sharex(oax['A'])
    target_df.set_index(datetime_label)[['p (mbar)']].plot(ax=oax['F'])
    oax['G'].sharex(oax['B'])
    target_df.set_index(datetime_label)[['VPmax (mbar)']].plot(ax=oax['G'])
    oax['H'].sharex(oax['C'])
    target_df.set_index(datetime_label)[['VPact (mbar)']].plot(ax=oax['H'])
    oax['I'].sharex(oax['D'])
    target_df.set_index(datetime_label)[['VPdef (mbar)']].plot(ax=oax['I'])
    sns.scatterplot(data=target_df, x='Tdew (degC)', y='VPact (mbar)', hue='is_ts_modified', ax=oax['J'])
    oax['K'].sharex(oax['A'])
    target_df.set_index(datetime_label)[['rh (%)']].plot(ax=oax['K'])
    oax['L'].sharex(oax['B'])
    target_df.set_index(datetime_label)[['rho (g/m**3)']].plot(ax=oax['L'])
    oax['M'].sharex(oax['C'])
    target_df.set_index(datetime_label)[['sh (g/kg)']].plot(ax=oax['M'])
    oax['N'].sharex(oax['D'])
    target_df.set_index(datetime_label)[['H2OC (mmol/mol)']].plot(ax=oax['N'])
    sns.scatterplot(data=target_df, x='sh (g/kg)', y='H2OC (mmol/mol)', hue='is_ts_modified', ax=oax['O'])
    oax['P'].sharex(oax['A'])
    target_df.set_index(datetime_label)[['wv (m/s)']].plot(ax=oax['P'])
    oax['Q'].sharex(oax['B'])
    target_df.set_index(datetime_label)[['max. wv (m/s)']].plot(ax=oax['Q'])
    oax['R'].sharex(oax['C'])
    target_df.set_index(datetime_label)[['rain (mm)']].plot(ax=oax['R'])
    oax['S'].sharex(oax['D'])
    target_df.set_index(datetime_label)[['raining (s)']].plot(ax=oax['S'])
    sns.scatterplot(data=target_df, x='wv (m/s)', y='max. wv (m/s)', hue='is_ts_modified', ax=oax['T'])
    oax['U'].sharex(oax['A'])
    target_df.set_index(datetime_label)[['SWDR (W/m**2)']].plot(ax=oax['U'])
    oax['V'].sharex(oax['B'])
    target_df.set_index(datetime_label)[['PAR (micromol/m**2/s)']].plot(ax=oax['V'])
    oax['W'].sharex(oax['C'])
    target_df.set_index(datetime_label)[['max. PAR (micromol/m**2/s)']].plot(ax=oax['W'])
    oax['X'].sharex(oax['D'])
    target_df.set_index(datetime_label)[['CO2 (ppm)']].plot(ax=oax['X'])
    sns.scatterplot(data=target_df, x='PAR (micromol/m**2/s)', y='max. PAR (micromol/m**2/s)', hue='is_ts_modified', ax=oax['Y'])
    # sns.scatterplot(data=target_df, x='PAR (micromol/m**2/s)', y='SWDR (W/m**2)', hue='is_ts_modified', ax=oax['Y'])
    for i in range(len(ax_labels)):
        if consecutive_count != 0:
            oax[ax_labels[i]].axvspan(start_modification, end_modification, color='red', alpha=0.5)
    plt.show()


def plot_isolated_errors(source_data, target_index, datetime_label, timestep_range=12):
    tmp_df = source_data.loc[target_index - timestep_range:target_index + timestep_range, :]
    plot_feature_from_one_source(tmp_df, target_index, datetime_label)


def plot_consecutive_errors(source_data, target_index, consecutive_count, datetime_label, timestep_range=12):
    tmp_df = source_data.loc[target_index-timestep_range:target_index+consecutive_count+timestep_range, :]
    print(consecutive_count)
    plot_feature_from_one_source(tmp_df, target_index, datetime_label, consecutive_count=consecutive_count)


def plot_regressor_relation(x_values, y_values, x_label, y_label, regressor=None, polynomial_features=None):
    afig, aax = plt.subplots(1, 1)
    aax.scatter(x_values, y_values, color='blue')
    if regressor is not None:
        if polynomial_features is not None:
            aax.plot(sorted(x_values), regressor.predict(polynomial_features.transform(sorted(x_values))), color='red')
        else:
            aax.plot(sorted(x_values), regressor.predict(sorted(x_values)), color='red')
    aax.set_xlabel(x_label)
    aax.set_ylabel(y_label)
    plt.show()


def compute_regressor_between_data(valid_data_dataframe, x_label, y_label, polynomial):
    train_x = valid_data_dataframe[x_label].values.reshape(len(valid_data_dataframe), 1)
    train_y = valid_data_dataframe[y_label].values.reshape(len(valid_data_dataframe), 1)
    regressor = LinearRegression()
    if polynomial > 0:
        polynomial_features = PolynomialFeatures(degree=polynomial, include_bias=False)
        train_poly_x = polynomial_features.fit_transform(train_x)
        regressor.fit(train_poly_x, train_y)
        plot_regressor_relation(train_x, train_y, x_label, y_label, regressor=regressor,
                                polynomial_features=polynomial_features)
        return regressor, polynomial_features
    else:
        regressor.fit(train_x, train_y)
        plot_regressor_relation(train_x, train_y, x_label, y_label, regressor=regressor,
                                polynomial_features=None)
        return regressor, None


def determine_correct_value_for_value_error(error_row, correct_known_data, x_label, y_label, polynomial=None):
    print(error_row)
    current_minute = error_row['minute']
    current_hour = error_row['hour']
    current_month = error_row['month']
    tmp_df = correct_known_data[(correct_known_data['minute'] == current_minute) &
                                (correct_known_data['hour'] == current_hour) &
                                (correct_known_data['month'] == current_month)]
    if len(tmp_df) > 1:
        if polynomial is None:
            pass
        else:
            regressor, polynomial_features = compute_regressor_between_data(tmp_df, x_label, y_label, polynomial)
            print(error_row[x_label], type(error_row[x_label]))
            print(np.reshape(error_row[x_label], (1, 1)))
            if polynomial_features is not None:
                return regressor.predict(polynomial_features.transform(np.reshape(error_row[x_label], (1, 1))))[0][0]
            else:
                return regressor.predict(np.reshape(error_row[x_label], (1, 1)))[0][0]
    elif len(tmp_df) == 1:
        print(tmp_df)
        print(error_row)
        if abs(error_row[x_label] - tmp_df[x_label]) < 10:
            return tmp_df[x_label]
        else:
            return np.nan
    else:
        return np.nan


def get_closet_similar_value_by_temporal_information(errors_df, correct_known_data, datetime_label, label_to_correct,
                                                     diff_threshold=200):
    '''
    :param errors_df:
    :param correct_known_data:
    :param datetime_label:
    :param label_to_correct:
    :param diff_threshold: 20 features (minus the one to correct), aim for less than 10 difference per features so 200
    :return:
    '''
    output_df = errors_df.copy()
    print(label_to_correct)
    valid_data_errors_df = errors_df.drop(columns=[datetime_label, 'minute', 'hour', 'month', label_to_correct,
                                                   'is_ts_missing', 'is_ts_modified'])
    for err_idx in errors_df.index:
        current_minute = errors_df.loc[err_idx, 'minute']
        current_hour = errors_df.loc[err_idx, 'hour']
        current_month = errors_df.loc[err_idx, 'month']
        tmp_df = correct_known_data[(correct_known_data['minute'] == current_minute) &
                                    (correct_known_data['hour'] == current_hour) &
                                    (correct_known_data['month'] == current_month)]
        if len(tmp_df) > 0:
            valid_data_tmp_df = tmp_df.drop(columns=[datetime_label, 'minute', 'hour', 'month', label_to_correct,
                                                     'is_ts_missing', 'is_ts_modified'])
            diff_df = valid_data_tmp_df.sub(valid_data_errors_df.loc[err_idx, :], axis='columns')
            print(diff_df)
            closest_df = diff_df.abs().sum(axis=1)
            print(closest_df)
            closest_idx = closest_df.idxmin()
            print(err_idx, closest_df.max(), closest_df.min(), closest_df.loc[closest_idx])
            if closest_df.loc[closest_idx] < diff_threshold:
                print(output_df.loc[err_idx, label_to_correct])
                print(closest_idx, label_to_correct)
                print(correct_known_data.loc[closest_idx, label_to_correct])
                output_df.at[err_idx, label_to_correct] = correct_known_data.loc[closest_idx, label_to_correct]
            else:
                output_df.at[err_idx, label_to_correct] = np.nan
        else:
            output_df.at[err_idx, label_to_correct] = np.nan
    return output_df


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
    from datetime import datetime
    from mpiw_ds_analysis import plot_mpi_dataset
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    parser = argparse.ArgumentParser(description='MPI Weather Dataset Errors Correction')
    parser.add_argument('--source-path', type=str, required=True,
                        help='Input the path for the parquet file created during analysis')
    args = parser.parse_args()

    source_data_path = args.source_path
    original_mpi_path = os.path.join(source_data_path, "Original/MaxPlanckInstitute/roof/")
    original_mpi_identification_wth_filename = "mpi_weather_errors_id.parquet"

    value_errors_dict = {
        "is_wv_value_error": {"column": 'wv (m/s)', "threshold": 200},
        "is_OT_value_error": {"column": 'OT', "threshold": 200},
        "is_CO2_value_error": {"column": 'CO2 (ppm)', "threshold": 200},
        "is_maxPAR_value_error": {"column": 'max. PAR (micromol/m**2/s)', "threshold": 200},
        "is_SWDR_value_error": {"column": 'SWDR (W/m**2)', "threshold": 200},
    }

    # ##################################################################################################################
    # Dataset from Autoformer
    poly_param = 0
    autoformer_mpi_path = os.path.join(source_data_path, "Autoformer/")
    autoformer_identification_wth_filename = "weather_errors_id.parquet"

    autoformer_wth_df = pd.read_parquet(os.path.join(autoformer_mpi_path, autoformer_identification_wth_filename))
    print(autoformer_wth_df)
    autoformer_validation_start = autoformer_wth_df.loc[int(len(autoformer_wth_df) * 0.7), 'date'].to_pydatetime()
    autoformer_evaluation_start = autoformer_wth_df.loc[int(len(autoformer_wth_df) * 0.8), 'date'].to_pydatetime()
    # Create a DataFrame where correction will be applied
    corrected_autoformer_wth_df = autoformer_wth_df.copy()

    temp_df = autoformer_wth_df.copy()
    temp_df['minute'] = temp_df['date'].apply(lambda x: x.minute)
    temp_df['hour'] = temp_df['date'].apply(lambda x: x.hour)
    temp_df['month'] = temp_df['date'].apply(lambda x: x.month)

    valid_df = temp_df[temp_df['is_ts_modified'] == 0].copy()

    # Correct value errors (-9999)
    for df_column in value_errors_dict.keys():
        if df_column in temp_df.columns:
            # ## Check for chunks of errors
            chunks_value_errors = determine_chunks_value(temp_df[df_column], target_value=1)
            print(chunks_value_errors)

            value_errors_df = temp_df[temp_df[df_column] == 1].copy()
            corresponding_column = value_errors_dict[df_column]['column']
            res_df = get_closet_similar_value_by_temporal_information(
                value_errors_df, valid_df, 'date', corresponding_column,
                diff_threshold=value_errors_dict[df_column]['threshold'])

            for idx in res_df.index:
                corrected_autoformer_wth_df.at[idx, corresponding_column] = res_df.loc[idx, corresponding_column]

            if df_column == "is_maxPAR_value_error":
                # For the remaining NaN it is possible to use other data as there is somehow a regression
                # correspondence between PAR and max. PAR
                remaining_error_df = res_df[res_df['max. PAR (micromol/m**2/s)'].isna()].copy()
                valid_PAR_df = valid_df[['PAR (micromol/m**2/s)', 'max. PAR (micromol/m**2/s)', 'minute', 'hour', 'month']]
                remaining_error_df['corrected'] = remaining_error_df.apply(
                    lambda row: determine_correct_value_for_value_error(row, valid_PAR_df,
                                                                        'PAR (micromol/m**2/s)',
                                                                        'max. PAR (micromol/m**2/s)',
                                                                        polynomial=poly_param), axis=1)
                for idx in remaining_error_df.index:
                    corrected_autoformer_wth_df.at[idx, 'max. PAR (micromol/m**2/s)'] = remaining_error_df.loc[
                        idx, 'corrected']

    plot_mpi_dataset(autoformer_wth_df, autoformer_validation_start, autoformer_evaluation_start, 'date',
                     raw_source='autoformer', phase='replacement_original')
    plot_mpi_dataset(corrected_autoformer_wth_df, autoformer_validation_start, autoformer_evaluation_start, 'date',
                     raw_source='autoformer', phase='replacement_corrected')

    corrected_autoformer_wth_df = corrected_autoformer_wth_df.interpolate(method='linear')
    plot_mpi_dataset(corrected_autoformer_wth_df, autoformer_validation_start, autoformer_evaluation_start, 'date',
                     raw_source='autoformer', phase='replacement_corrected_interpolation')

    print(corrected_autoformer_wth_df.columns)
    corrected_autoformer_wth_df.to_parquet(os.path.join(source_data_path, "corrected_autoformer_weather.parquet"))
    corrected_autoformer_wth_df.to_csv(
        os.path.join("revised_datasets/", "MPIW_10T_1Y_R.csv"), index=False)

    # ##################################################################################################################
    # Original data
    poly_param = 0
    mpi_wth_df = pd.read_parquet(os.path.join(original_mpi_path, original_mpi_identification_wth_filename))
    print(mpi_wth_df)
    target_validation_start = datetime(2022, 1, 1, 0, 0)
    target_evaluation_start = datetime(2023, 1, 1, 0, 0)
    # Create a DataFrame where correction will be applied
    corrected_mpi_wth_df = mpi_wth_df.copy()

    temp_df = mpi_wth_df.copy()
    temp_df['minute'] = temp_df['Date Time'].apply(lambda x: x.minute)
    temp_df['hour'] = temp_df['Date Time'].apply(lambda x: x.hour)
    temp_df['month'] = temp_df['Date Time'].apply(lambda x: x.month)

    valid_df = temp_df[temp_df['is_ts_modified'] == 0].copy()

    # Correct value errors (-9999)
    for df_column in value_errors_dict.keys():
        if df_column in temp_df.columns:
            # ## Check for chunks of errors
            chunks_value_errors = determine_chunks_value(temp_df[df_column], target_value=1)
            print(chunks_value_errors)

            value_errors_df = temp_df[temp_df[df_column] == 1].copy()
            corresponding_column = value_errors_dict[df_column]['column']
            res_df = get_closet_similar_value_by_temporal_information(
                value_errors_df, valid_df, 'Date Time', corresponding_column,
                diff_threshold=value_errors_dict[df_column]['threshold'])

            for idx in res_df.index:
                corrected_mpi_wth_df.at[idx, corresponding_column] = res_df.loc[idx, corresponding_column]

            if df_column == "is_maxPAR_value_error":
                # For the remaining NaN it is possible to use other data as there is somehow a regression correspondence
                # between PAR and max. PAR
                remaining_error_df = res_df[res_df['max. PAR (micromol/m**2/s)'].isna()].copy()
                valid_PAR_df = valid_df[['PAR (micromol/m**2/s)', 'max. PAR (micromol/m**2/s)', 'minute', 'hour', 'month']]
                remaining_error_df['corrected'] = remaining_error_df.apply(
                    lambda row: determine_correct_value_for_value_error(row, valid_PAR_df,
                                                                        'PAR (micromol/m**2/s)',
                                                                        'max. PAR (micromol/m**2/s)',
                                                                        polynomial=poly_param),
                    axis=1)
                for idx in remaining_error_df.index:
                    corrected_mpi_wth_df.at[idx, 'max. PAR (micromol/m**2/s)'] = remaining_error_df.loc[
                        idx, 'corrected']

            if df_column == "is_SWDR_value_error":
                # For the remaining NaN it is possible to use other data as there is somehow a regression correspondance
                # between PAR and SWDR
                remaining_error_df = res_df[res_df['SWDR (W/m**2)'].isna()].copy()
                valid_PAR_df = valid_df[['PAR (micromol/m**2/s)', 'SWDR (W/m**2)', 'minute', 'hour', 'month']]
                remaining_error_df['corrected'] = remaining_error_df.apply(
                    lambda row: determine_correct_value_for_value_error(row, valid_PAR_df,
                                                                        'PAR (micromol/m**2/s)', 'SWDR (W/m**2)',
                                                                        polynomial=poly_param),
                    axis=1)
                for idx in remaining_error_df.index:
                    corrected_mpi_wth_df.at[idx, 'SWDR (W/m**2)'] = remaining_error_df.loc[
                        idx, 'corrected']

    plot_mpi_dataset(mpi_wth_df, target_validation_start, target_evaluation_start, 'Date Time',
                     phase='replacement_original')
    plot_mpi_dataset(corrected_mpi_wth_df, target_validation_start, target_evaluation_start, 'Date Time',
                     phase='replacement_corrected')

    corrected_mpi_wth_df = corrected_mpi_wth_df.interpolate(method='linear')
    plot_mpi_dataset(corrected_mpi_wth_df, target_validation_start, target_evaluation_start, 'Date Time',
                     phase='replacement_corrected_interpolation')

    corrected_mpi_wth_df.to_parquet(os.path.join(source_data_path, "corrected_mpi_weather_4y_10min.parquet"))
    corrected_mpi_wth_df.to_csv(
        os.path.join("revised_datasets/", "MPIW_10T_4Y_R.csv"), index=False)
