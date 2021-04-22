import os

import pandas as pd
import shutil

src_dir = r"D:\Git_tasks\task_new\sednn\mixture2clean_dnn\metadata\test_speech"
dest_dir = r"D:\Git_tasks\task_new\sednn\mixture2clean_dnn\metadata\sub_test_speech"

df = pd.read_csv(r'D:\Git_tasks\task_new\sednn\mixture2clean_dnn\metadata\test_timit.csv', header=None, index_col=0)  # 不设置header会自动把第一行作列属性。index_col设为0，不显示行号

for row in df.iterrows():
    # print(row[0])
    filepath = os.path.join(src_dir, row[0])
    print(filepath)
    new_filepath = os.path.join(dest_dir, row[0])
    print(new_filepath)
    shutil.copyfile(filepath, new_filepath)
