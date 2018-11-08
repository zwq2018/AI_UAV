import pandas as pd
from pandas import DataFrame
import numpy as np

log_index = {}
file_data = {}
out_name_list = []
user_output = DataFrame()
variable_package_name = ''
variable_name = ''


def write_data_to_excel(out_name, need_variable_package_name, need_variable_name):
    post_sep_data = separated_log_data(file_data[out_name])  # 分离数据

    data_frame = DataFrame(post_sep_data)  # 把变量包的变量数据的list转换成DataFrame类型
    data_grouped = data_frame.groupby([0])  # 用GroupBy对log_data_frame进行分组处理

    out_file_name = out_name.replace('.log', '.xlsx')
    out_writer = pd.ExcelWriter(out_file_name)

    for name, groups in data_grouped:
        del groups[0]  # 删除第一列，第一列全部位变量名称，注意DataFrame的行索引和列索引方式
        groups = DataFrame(groups, dtype=np.float)  # 把DataFrame的对象转换成float类型
        groups = groups.dropna(axis=1)  # 把全为NA的列删去
        group_data = groups.agg(['max', 'min', 'mean', 'std', np.var])
        group_data.columns = log_index[name]  # 修改列索引
        group_data.to_excel(out_writer, name)  # 把汇总信息输出到Excel表格里面
        if name == need_variable_package_name:  # 保存用户需要的信息
            print(out_name)
            print(need_variable_package_name)
            print(group_data[need_variable_name])
    out_writer.save()  # 千万别忘了写这句


def separated_log_data(current_file):
    log_data = []
    ard_status = False
    # print(current_file)
    for file_line in current_file:
        post_separation_line = [x.strip() for x in file_line.split(',')]  # 分离所有数据
        if post_separation_line[0] == 'FMT':
            if post_separation_line[3] != 'FMT':
                if post_separation_line[3] != 'PARM' and post_separation_line[3] != 'MSG':
                    del post_separation_line[4]  # 无效数据，删去
                    del post_separation_line[0:3]  # 删除前三个元素，因为无效数据
                    if post_separation_line[0] not in log_index.keys():
                        log_index[post_separation_line[0]] = post_separation_line[1:]  # 字典的方式存放每个变量包的变量名称
        elif post_separation_line[0] != 'PARM' and post_separation_line[0] != 'MSG':  # 第三种情况的处理
            if post_separation_line[0] == 'MKF1' and post_separation_line[10] == '1':
                ard_status = True
            elif post_separation_line[0] == 'MKF1' and post_separation_line[10] == '0':
                ard_status = False

            if ard_status:  # ard_status=True代表解锁状态，只要解锁状态才对数据进行处理
                log_data.append(post_separation_line)
    return log_data


def read_data_from_log(current_file):
    with open(current_file) as read_current_file:
        current_file_data = read_current_file.readlines()
    file_data[current_file] = current_file_data


