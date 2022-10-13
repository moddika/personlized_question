import cudf as pd_cf
from cudf.utils.dtypes import datetime
import matplotlib.pyplot as plt
import os

os.system('export DISPLAY=:0.0')

data_ans = pd_cf.read_csv("data/answer_metadata.csv")
#AnswerId             DateAnswered Confidence  GroupId  QuizId SchemeOfWorkId [1508917 rows x 6 columns]
data_ans = data_ans[['AnswerId','DateAnswered']]
#AnswerId             DateAnswered  [1508917 rows x 2 columns]


data_ques = pd_cf.read_csv("data/question_metadata.csv")
#QuestionId         SubjectId   [948 rows x 2 columns]


data_stu = pd_cf.read_csv("data/student_metadata.csv")
#UserId  Gender              DateOfBirth PremiumPupil   [6148 rows x 4 columns]
data_stu = data_stu[['UserId','Gender','DateOfBirth']]
#UserId  Gender              DateOfBirth    [6148 rows x 3 columns]



data_relation = pd_cf.read_csv("data/train_task_4.csv")
#QuestionId  UserId  AnswerId  IsCorrect  CorrectAnswer  AnswerValue    [1307347 rows x 6 columns]


def age_extraction(data):
    age_sum = 0
    now = datetime.datetime.now()
    age_list = []
    data = data.to_pandas()
    for x in data['DateOfBirth']:
        if x !=  None:
            x.strip()
            x.strip("\n")
            year, month = int(x.split('-')[0]), int(x.split('-')[1])
            age =  (now.year-year)- (now.month-month)/12.
            age_list.append(age)
            age_sum += age
        else:
            age_list.append(x)
    data['age'] = age_list
    mean_age = age_sum / len(age_list)
    data['age'].fillna(value = mean_age,inplace = True)
    return data

data_stu = age_extraction(data_stu)

plt.hist(data_stu.age)
plt.plot()
plt.show()