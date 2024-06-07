import pandas as pd


def assert_the_dataset_have_all_time_steps(start_datetime, end_datetime, granularity, length_of_dataset):
    data_range = pd.date_range(start_datetime, end_datetime, freq=granularity, inclusive='both')
    if len(data_range) == length_of_dataset:
        return None
    else:
        return data_range


def add_missing_time_steps(correct_data_range, considered_dataframe, datetime_label):
    tmp_df = pd.DataFrame()
    tmp_df[datetime_label] = correct_data_range
    tmp_df = pd.merge(tmp_df, considered_dataframe, on=datetime_label, how='left')
    return tmp_df


def check_for_missing_time_steps_and_identify(considered_dataframe, granularity, datetime_label, verbose=False):
    tmp_df = considered_dataframe.copy()
    dr = assert_the_dataset_have_all_time_steps(tmp_df.iloc[0, 0],
                                                tmp_df.iloc[len(tmp_df) - 1, 0],
                                                granularity, len(tmp_df))
    if dr is not None:
        if len(dr) < len(tmp_df):
            # Check if there is duplicated time steps with different values
            tss = tmp_df[datetime_label]
            print(tmp_df[tss.isin(tss[tss.duplicated()])].sort_values(datetime_label))
            assert len(dr) >= len(tmp_df), "There is an issue with the data range"
        else:
            tmp_df = add_missing_time_steps(dr, tmp_df, datetime_label)
    tmp_df['is_ts_missing'] = [0] * len(tmp_df)

    # Identify missing time steps
    for idx in tmp_df[tmp_df.isna().any(axis=1)].index:
        tmp_df.iat[idx, -1] = 1

    print(f"Dataset has {len(tmp_df[tmp_df['is_ts_missing'] == 1])} missing time steps.")
    if dr is not None and verbose:
        print(tmp_df[tmp_df['is_ts_missing'] == 1])

    return tmp_df


def check_for_missing_data_replaced(target_row, target_value):
    if target_value in target_row.values:
        return 1
    else:
        return 0


def check_for_value_error(target_value, considered_error_value):
    if target_value == considered_error_value:
        return 1
    else:
        return 0
