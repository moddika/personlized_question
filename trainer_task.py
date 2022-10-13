import numpy as np
import matplotlib.pyplot as plt
import random
import scipy as sp
import pandas as pd
import torch
import os
import argparse
import heapq
from dataset_task import  FFDataset, ff_collate
from utils import open_json, dump_json, compute_auc, compute_accuracy
from model_task import FFModel
import time
from tqdm import tqdm


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
DEBUG = False
epoch_information = []

def hash_subject2question(df,question_id,column):
    subject_dic = {}
    for x in range(len(df[question_id])):
        subject_dic[df[question_id][x]] = df[column][x]
    return subject_dic

def reindex_stu(df):
    df.index = df['UserId']
    df.drop(columns = ['UserId','DateOfBirth','Unnamed: 0'],inplace = True)
    return df

def pivot_df(df, values):
    """
    Convert dataframe of question and answerrecords to pivoted array, filling in missing columns if some questions are 
    unobserved.
    """
    data = df.pivot(index='UserId', columns='QuestionId', values=values)

    # Add rows for any questions not in the test set
    data_cols = data.columns ##question_id
    max_colum_index = data_cols.max()
    all_cols = np.arange(max_colum_index)
    print(data.shape)
    missing = set(all_cols) - set(data_cols)
    for i in missing:
        data[i] = np.nan

    data = data.reindex(sorted(data.columns), axis=1)

    data = data.to_numpy()
    data[np.isnan(data)] = -1
    return data

def train_model():
    global epoch_information
    model.train()
    max_epoch = 120
    N = [idx for idx in range(100, 100+max_epoch)]
    for batch in train_loader:
        optimizer.zero_grad()
        if epoch<=50 or random.random()>=params.mix_active:
            loss, _ = model(batch)
        else:
            loss, _ = model.forward_active(batch)
        loss.backward()
        optimizer.step()
    model.eval()
    scores = []
    for idx in tqdm(N):
        scores.append(test_model(id_=idx )[2])
    final_score = sum(scores)/(len(N)+1e-8)
    file_name = './model/model_task_'+params.file_name+'_e_'+str(epoch)+'.pt'
    heapq.heappush(epoch_information, (final_score, file_name))
    remove_filename = heapq.heappop(epoch_information)[1] if len(epoch_information) > 5 else None
    if file_name != remove_filename:
        torch.save({'model_state_dict': model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"epoch":epoch},file_name)
        if remove_filename:
            os.remove(remove_filename)
    dump_json('/home/blackbox/workspace/hjy/NeurIPSEducation2020/model/'+params.file_name+'.json', epoch_information)    
    print('Test_Epoch: {} Scores are: {}'.format(epoch, scores))
    model.train()


def test_model(id_):
    valid_dataset.seed = id_
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, collate_fn=collate_fn, batch_size=64, num_workers=num_workers, shuffle=False, drop_last=False)
    total_loss, all_preds, all_targets = 0., [], []
    n_batch = 0
    for batch in tqdm(valid_loader):
        with torch.no_grad():
            output = model.test(batch)
        target = batch['output_labels'].float().numpy()
        mask = batch['output_mask'].numpy() == 1
        all_preds.append(output[mask])
        all_targets.append(target[mask])
        n_batch += 1
    all_pred = np.concatenate(all_preds, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    return total_loss/n_batch, auc, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML')
    parser.add_argument('--model', type=str, default='ff', help='type')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='type')
    parser.add_argument('--question_dim', type=int, default=4, help='type')
    parser.add_argument('--concat_hidden_dim', type=int, default=128, help='type')
    parser.add_argument('--concat_dim', type=int,default=1024, help='type')
    parser.add_argument('--lr', type=float, default=2e-5, help='type')
    parser.add_argument('--dropout', type=float, default=0.5, help='type')
    parser.add_argument('--mix_active', type=float, default=0.67, help='type')
    # parser.add_argument()
    params = parser.parse_args()
    file_name = [params.model, params.hidden_dim, params.question_dim, params.lr, params.dropout, params.mix_active, params.concat_hidden_dim, params.concat_dim]
    file_name =  [str(d) for d in file_name]
    params.file_name = '_'.join(file_name)
    ################ seed producing
    seedNum = 221
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    random.seed(seedNum)

    data_stu_path = os.path.normpath('./train_data/data_stu.csv') 
    train_data_path = os.path.normpath('./train_data/data_train.csv')        
    valid_data_path = os.path.normpath('./train_data/data_train.csv')
    valid_df = pd.read_csv(valid_data_path)
    valid_data = pivot_df(valid_df, 'AnswerValue')#n_student, 948:    1 to 4 and -1
    valid_binary_data = pivot_df(valid_df, 'IsCorrect')  # n_student, 948: 1 to 0 and -1
    train_df = pd.read_csv(train_data_path)
    # n_student, 948:    1 to 4 and -1
    train_data = pivot_df(train_df, 'AnswerValue')
    # n_student, 948: 1 to 0 and -1
    train_binary_data = pivot_df(train_df, 'IsCorrect')
    data_stu = pd.read_csv("./train_data/data_stu.csv")
    # quetion_subject = hash_subject2question(train_df,'QuestionId','SubjectId')
    data_feature = reindex_stu(data_stu)
    data_feature = np.array(data_feature)
    train_dataset = FFDataset(train_data, train_binary_data,  data_feature)
    valid_dataset = FFDataset(valid_data, valid_binary_data,  data_feature)
    num_workers = 3
    collate_fn =   ff_collate()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=64, num_workers=num_workers, shuffle=True, drop_last=True)
    model = FFModel(hidden_dim=params.hidden_dim, dim=params.question_dim, concat_dim=params.concat_dim, concat_hidden_dim=params.concat_hidden_dim,dropout=params.dropout).to(device)
    # file_name = './model/model_task_ff_1024_4_2e-05_0.5_0.67_128_1024_e_18.pt'
    # checkpoint = torch.load(file_name, map_location=device)
    # words = file_name.split('_')
    # last_epoch = words[-1][:2]
    # print(last_epoch)
    # model.load_state_dict(checkpoint['model_state_dict'])
    last_epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=1e-8)
    start_time = time.time()
    for epoch in tqdm(range(int(last_epoch),250)):
        train_model()
    end_time = time.time()
    print("Time Elapsed: {} hours".format((end_time-start_time)/3600.))

