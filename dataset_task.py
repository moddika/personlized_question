import numpy as np
import torch
from torch.utils import data
import time
import torch
import random
from utils import open_json, dump_json

question_num = 948

class FFDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, answers, labels, stu_meta,seed =None):
        'Initialization'
        self.answers = answers
        self.labels = labels
        self.seed = seed
        #self.targets = targets
        self.stu_meta = stu_meta

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.answers)

    def __getitem__(self, index):
        'Generates one sample of data'
        ans = self.answers[index]
        label = self.labels[index]
        stu_feature = self.stu_meta[index]
        stu_feature = np.array(stu_feature)
        observed_index = np.where(label != -1.)[0]
        if not self.seed:
            np.random.shuffle(observed_index)
        else:
            random.Random(index+self.seed).shuffle(observed_index)
        N = len(observed_index)
        target_index = observed_index[-N//5:]
        trainable_index = observed_index[:-N//5]
        
        input_ans = ans[trainable_index]
        input_label = label[trainable_index]
        input_question = trainable_index

        #output_ans = ans[target_index]
        output_label = label[target_index]
        output_question = target_index

        output = {'input_label': torch.FloatTensor(input_label), 'input_question': torch.FloatTensor(input_question),
                  'output_question': torch.FloatTensor(output_question), 'output_label': torch.FloatTensor(output_label),
                  'stu_feature':torch.FloatTensor(stu_feature),'input_ans': torch.FloatTensor(input_ans)}

        return output


class ff_collate(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        B = len(batch)
        input_labels =  torch.zeros(B,question_num).long()
        output_labels = torch.zeros(B, question_num).long()
        input_ans = torch.ones(B, question_num).long()
        input_mask  = torch.zeros(B,question_num).long()
        output_mask = torch.zeros(B, question_num).long()
        stu_features = []
        for b_idx in range(B):
            input_labels[b_idx, batch[b_idx]['input_question'].long()] =  batch[b_idx]['input_label'].long()
            input_ans[b_idx, batch[b_idx]['input_question'].long()] = batch[b_idx]['input_ans'].long()
            input_mask[b_idx, batch[b_idx]['input_question'].long()] = 1
            output_labels[b_idx, batch[b_idx]['output_question'].long()] =  batch[b_idx]['output_label'].long()
            output_mask[b_idx, batch[b_idx]['output_question'].long()] = 1
            stu_features.append(batch[b_idx]['stu_feature'])
        stu_features = torch.tensor(np.array([item.numpy() for item in stu_features]))
        output = {'input_labels':input_labels, 'input_ans':input_ans, 'input_mask':input_mask, 'output_labels':output_labels, 'output_mask':output_mask,'stu_features':stu_features}
        return output

