# Python处理数据集（修改文件名以包含父目录名）
import os
import shutil

src_dir = r"D:\Git_tasks\task_new\sednn\mixture2clean_dnn\metadata\timit\data\TEST"
dest_dir = r"D:\Git_tasks\task_new\sednn\mixture2clean_dnn\metadata\test_speech"


def listDirRecurrently(path):
    filelist = os.listdir(path)
    for file in filelist:
        filepath = os.path.join(path, file)
        if os.path.isdir(filepath):
            listDirRecurrently(filepath)
        else:
            if file.endswith('WAV'):
                print(filepath)
                categories = filepath.split('\\')
                new_file_name = 'TEST_' + categories[-3] + '_' + categories[-2] + '_' + categories[-1]
                new_filepath = os.path.join(dest_dir, new_file_name)
                print(new_file_name)
                print(new_filepath)
                shutil.copyfile(filepath, new_filepath)

if __name__ == '__main__':
    listDirRecurrently(src_dir)