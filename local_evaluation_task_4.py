from submission_model_task_4 import Submission
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import os
from trainer_task import reindex_stu
import torch


def pivot_df(df, values):
    """
    Convert dataframe of question and answerrecords to pivoted array, filling in missing columns if some questions are 
    unobserved.
    """
    data = df.pivot(index='UserId', columns='QuestionId', values=values)
    index = data.index
    # Add rows for any questions not in the test set
    data_cols = data.columns
    all_cols = np.arange(948)
    missing = set(all_cols) - set(data_cols)
    for i in missing:
        data[i] = np.nan
    data = data.reindex(sorted(data.columns), axis=1)

    data = data.to_numpy()
    data[np.isnan(data)] = -1
    return data,index


if __name__ == "__main__":
    data_path = os.path.normpath(
        'train_data/valid_task.csv')
    data_stu_path = os.path.normpath(
        'train_data/data_stu.csv')
    df = pd.read_csv(data_path)
    df_stu = pd.read_csv(data_stu_path)
    data,index = pivot_df(df, 'AnswerValue')
    binary_data,_ = pivot_df(df, 'IsCorrect')
    df_stu = reindex_stu(df_stu)
    data_feature = df_stu.iloc[index]
    data_feature = data_feature.to_numpy()
    # Array containing -1 for unobserved, 0 for observed and not target (can query), 1 for observed and target (held out
    # for evaluation).
    targets,_ = pivot_df(df, 'IsTarget')

    observations = np.zeros_like(data)
    masked_data = data * observations
    masked_binary_data = binary_data * observations

    can_query = (targets == 0).astype(int)
    submission = Submission()

    for i in range(10):
        print('Feature selection step {}'.format(i+1))
        next_questions = submission.select_feature(
            masked_data, masked_binary_data, can_query,data_feature)
        print(next_questions[0])
        # Validate not choosing previously selected question here

        for i in range(can_query.shape[0]):
            # Validate choosing queriable target here
            assert can_query[i, next_questions[i]] == 1
            can_query[i, next_questions[i]] = 0

            # Validate choosing unselected target here
            assert observations[i, next_questions[i]] == 0

            observations[i, next_questions[i]] = 1
            masked_data = data * observations
            masked_binary_data = binary_data * observations

        # Update model with new data, if required
        submission.update_model(masked_data, masked_binary_data, can_query)

    preds = submission.predict(masked_data, masked_binary_data,data_feature)

    pred_list = preds[np.where(targets == 1)]
    target_list = binary_data[np.where(targets == 1)]
    acc = (pred_list == target_list).astype(int).sum()/len(target_list)
    print('Final accuracy: {}'.format(acc))
