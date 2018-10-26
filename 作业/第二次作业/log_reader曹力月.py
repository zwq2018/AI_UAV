import numpy as np
import pandas as pd
import csv
import os
import os.path
import openpyxl
# from xlwt import *

"""part1:不区分解锁状态的数据"""
"""读取指定飞行日志文件中的数据，并根据指定的变量包名输出所有变量名"""
def search_var_package(in_file_name, package_name):
    with open(in_file_name) as file_object:
        lines = file_object.readlines()

    for line in lines:
        if line.startswith('FMT'):
            fmt_range = line.rstrip().split(', ')
            if fmt_range[3] == package_name:
                print('PackageName', end=':')
                print(fmt_range[3], end='; ')
                print(fmt_range[5:], sep=',')


"""根据指定文件，将所有变量包名和其对应的所有变量名提取到一个二维列表中"""
def packages_variables_list(in_file_name):
    with open(in_file_name) as file_object:
        lines = file_object.readlines()

    package_list = []  # package_list保存所有的变量包名和变量名，一个变量包占一行
    for line in lines:
        if line.startswith('FMT'):
            if line[3] != 'FMT' and line[3] != 'PARM' and line[3] != 'MSG' and line[3] != 'MODE':
                line = line.replace(',', ' ')
                fmt_range = line.rstrip().split()
                # print(fmt_range)
                package_variable = []  # package_variable保存单个变量包名及其对应的变量名
                package_name = fmt_range[3]  # package_name保存每行中读取到的变量包名
                package_variable.append(package_name)

                for variable in fmt_range[5:]:
                    package_variable.append(variable)

                package_list.append(package_variable)

    return package_list


"""根据指定的文件名和变量包名，找到对应所有变量的数据信息并输出"""
def search_data_variable(in_file_name, get_package_name):
    # package_list = packages_variables_list(in_file_name)

    with open(in_file_name) as file_object:
        lines = file_object.readlines()

    for line in lines:
        if line.startswith(get_package_name):
            print(line.rstrip())


"""根据指定的文件和变量包名，将其对应的所有变量的所有数据提取到一个二维列表中"""
def variables_data_list(in_file_name, package_name, package_list):
    with open(in_file_name) as file_object:
        lines = file_object.readlines()

    variables_list = []

    for i in package_list:
        if i[0] == package_name:
            variables_list.append(i)

    for line in lines:
        if line.startswith(package_name):
            line = line.replace(',', ' ')
            var_data_range = line.rstrip().split()
            variables_list.append(var_data_range)

    print(variables_list)
    return variables_list


"""导出变量包中变量的所有数据的二维列表到Excel表格中"""
def variables_data2_to_xls(out_file_name, variables_list):
    out_put = open(out_file_name, 'a', encoding='utf-8')

    for i in range(len(variables_list)):
        for j in range(len(variables_list[i])):
            out_put.write(variables_list[i][j])
            out_put.write('\t')
        out_put.write('\n')


"""导出变量包中变量的所有数据的一维列表到Excel表格中"""
def variables_data1_to_xls(out_file_name, variables_list):
    out_put = open(out_file_name, 'a', encoding='utf-8')

    for i in variables_list:
        out_put.write(i)
        out_put.write('\t')
    out_put.write('\n')



"""part2: 仅提取解锁状态的数据"""
"""去掉开头为'FMT'和'PARM'的行，仅提取含变量包名及对应变量数据的行，依次保存到一个二维列表中"""
def search_all_variables_data(in_file_name):
    with open(in_file_name) as file_object:
        lines = file_object.readlines()

    packages_variables_all = [] # packages_variables_locked保存全部变量包的变量数据
    for line in lines:
        # if line[:2] != 'FMT' and line[:3] != 'PARM' and line[:3] != 'MODE':
        if not line.startswith('FMT') and not line.startswith('PARM') and not line.startswith('MODE') and not line.startswith('RTK') and not line.startswith('MSG'):
            # print(line[:3])
            # 先把每行的元素以空格分开保存到一维列表variables_all_data中
            line = line.replace(',',' ')
            variables_all_data = line.rstrip().split()
            # 再把这个一维列表作为行添加到二维列表中
            packages_variables_all.append(variables_all_data)
    # print(packages_variables_locked)
    return packages_variables_all


"""根据读到的全部变量数据的二维列表，判断MKF1变量包中的ARD是否为高电平（1），若是则为解锁状态
   提取MKF1的ARD变为1后的所有变量数据到一个二维列表中"""
def search_variables_unlocked(packages_variables_all):
    packages_variables_unlocked = []  # 二维列表：保存所有解锁状态的变量数据
    # line_count = 0
    ard_status = 0
    for i in packages_variables_all:
        # print(i[0])
        # line_count = line_count + 1
        if i[0] == "MKF1" and i[10] == '1':
            ard_status = 1
        if i[0] == "MKF1" and i[10] == '0':
            ard_status = 0
        if ard_status == 1:
            # print("i[0]=", i[0])
            packages_variables_unlocked.append(i)
    return packages_variables_unlocked

"""
def search_varables_data_unlocked(packages_variables_all):
    pv_data = np.array(packages_variables_all)  # 将所有变量数据的二维列表转化为numpy数组
    unlocked_data_all = []  # 二维数组：保存所有解锁状态的数据
    seg_row = []  # 一维列表：存储需要分隔的行索引
    row1 = 0  # 标志位，标志第一个找到ARD为1的行
    row0 = 0  # 标志位，标志第一个找到ARD为0的行
    for i in pv_data:
        if row1 == 0 and i[0] == "MKF1" and i[10] == '1':
            seg_row.append(np.where(i))
            row0 = 0
        if row0 == 0 and i[0] == "MKF1" and i[10] == '0':
            seg_row.append(np.where(i))
            row1 = 0

    for row in range(0,len(seg_row),2):
        unlocked_data = pv_data[seg_row[row]:seg_row[row+1]]
    unlocked_data_all.append(unlocked_data)
    print(unlocked_data_all)
"""

"""已有解锁状态的所有变量数据的二维列表，根据指定的变量包名提取出仅含该变量包解锁数据的二维列表"""
def search_unlocked_data_by_package(packages_variables_unlocked, package_name, package_list):
    only_package_data_unlocked = []

    for name in package_list:
        if name[0] == package_name:
            only_package_data_unlocked.append(name)
            break

    for i in packages_variables_unlocked:
        # print("i=",i)
        if i[0] == package_name:
            only_package_data_unlocked.append(i)
    # print(only_package_data_unlocked)
    return only_package_data_unlocked


"""已有指定变量包的变量解锁状态数据的二维列表，统计各变量解锁状态下的最大值和最小值"""
def cal_variable_max_min(only_package_data_unlocked):
    only_data_unlocked = []  # 仅取出所有的数据
    for i in only_package_data_unlocked[1:]:
        only_data_unlocked.append(i[1:])

    a = np.array(only_data_unlocked)  # 将数据列表转化为numpy二维数组
    a = a.astype(np.float)

    # 统计每个变量的最大值，得到最大值一维数组
    v_max = a.max(0)
    print(v_max)
    single_variable_max = np.array(['max'])
    for i in v_max:
        single_variable_max = np.append(single_variable_max, i)
    single_variable_max.astype(str)
    single_variable_max = single_variable_max.tolist()
    print(single_variable_max)

    # 统计每个变量的最小值，得到最小值数组
    v_min = a.min(0)
    # print(v_min)
    single_variable_min = np.array(['min'])
    for i in v_min:
        single_variable_min = np.append(single_variable_min, i)
    single_variable_min.astype(str)
    single_variable_min = single_variable_min.tolist()
    print(single_variable_min)

    return single_variable_max, single_variable_min


"""已有指定变量包的变量解锁状态数据的二维列表，统计各变量解锁状态下的均值和方差"""
def cal_variable_means_var(only_package_data_unlocked):
    only_data_unlocked = []  # 仅取出所有的数据
    for i in only_package_data_unlocked[1:]:
        only_data_unlocked.append(i[1:])

    a = np.array(only_data_unlocked).astype(np.float)  # 将数据列表转化为numpy二维数组

    # 统计每个变量的均值
    v_mean = a.mean(0)
    single_variable_mean = np.array('mean')
    for i in v_mean:
        single_variable_mean = np.append(single_variable_mean, i)
    single_variable_mean = single_variable_mean.tolist()
    print(single_variable_mean)

    # 统计每个变量的方差
    v_var = a.var(0)
    single_variable_var = np.array('var')
    for i in v_var:
        single_variable_var = np.append(single_variable_var, i)
    single_variable_var = single_variable_var.tolist()
    print(single_variable_var)

    return single_variable_mean, single_variable_var


"""用csv的方式导出数据到Excel"""
def csv_data_to_excel(out_file_name, data_list):
    with open(out_file_name, 'w', newline='') as file_object:
        writer = csv.writer(file_object)
        writer.writerows(data_list)


"""给定路径，按要求（制定变量包名等）循环读出该路径下的所有解锁状态变量数据到Excel"""
def all_files_unlocked_data(package_name):
    path = 'E:\\工程实践\\AI+UVA\\工作安排\\第二次作业-python脚本取数据\\flightLog\\'
    files = os.listdir(path)

    out_file_name = input("请输入要输出的文件名：")

    for file in files:
        in_file_path = path + file
        package_list = packages_variables_list(in_file_path)  # 获得所有变量包及其变量名的二维列表
        packages_variables_all = search_all_variables_data(in_file_path)  # 获得所有变量及其数据的二维列表
        packages_variables_unlocked = search_variables_unlocked(packages_variables_all)  # 获得所有变量解锁状态数据的二维列表
        only_package_data_unlocked = search_unlocked_data_by_package(packages_variables_unlocked, package_name,
                                                                     package_list)  # 获得指定变量包的解锁数据的二维列表

        variables_data2_to_xls(out_file_name, only_package_data_unlocked)  # 将指定变量包的解锁数据输出到Excel

        # 将指定变量包的变量的最值输出到Excel
        single_variable_max, single_variable_min = cal_variable_max_min(only_package_data_unlocked)
        variables_data1_to_xls(out_file_name, single_variable_max)
        variables_data1_to_xls(out_file_name, single_variable_min)

        # 将指定变量包的变量的均值和方差输出到Excel
        single_variable_mean, single_variable_var = cal_variable_means_var(only_package_data_unlocked)
        variables_data1_to_xls(out_file_name,single_variable_mean)
        variables_data1_to_xls(out_file_name,single_variable_var)
        break


"""循环输出单个文件中所有的变量包的解锁数据到不同的文件中"""
def out_package_data_unlocked(in_file_name):
    package_list = packages_variables_list(in_file_name)
    packages_variables_all = search_all_variables_data(in_file_name)
    packages_variables_unlocked = search_variables_unlocked(packages_variables_all)

    path = 'E:\\工程实践\\AI+UVA\\工作安排\\第二次作业-python脚本取数据\\satistic_data\\'

    for i in packages_variables_all:
        package_name = i[0]
        only_package_data_unlocked = search_unlocked_data_by_package(packages_variables_unlocked, package_name,
                                                                     package_list)
        # print(only_package_data_unlocked)
        single_variable_max, single_variable_min = cal_variable_max_min(only_package_data_unlocked)
        single_variable_mean, single_variable_var = cal_variable_means_var(only_package_data_unlocked)
        print(package_name)
        out_file_name = path + package_name + '_' + in_file_name[6:8] +  r".xls"
        variables_data2_to_xls(out_file_name, only_package_data_unlocked)
        variables_data1_to_xls(out_file_name, single_variable_max)
        variables_data1_to_xls(out_file_name, single_variable_min)

        variables_data1_to_xls(out_file_name, single_variable_mean)
        variables_data1_to_xls(out_file_name, single_variable_var)

"""测试函数"""
def _main():
    in_file_name = input("请输入要读取的文件名：")
    # package_name = input("请输入要查找的变量包：")

    """part1"""
    # search_var_package(in_file_name, package_name)

    # package_list = packages_variables_list(in_file_name)
    # print(package_list)

    # search_data_variable(in_file_name, package_name)

    # variables_list = variables_data_list(in_file_name, package_name, package_list)

    # out_file_name = input("请输入要导出的Excel文件名：")
    # variables_data_to_xls(out_file_name, variables_list)

    """part2"""
    # packages_variables_all = search_all_variables_data(in_file_name)
    # packages_variables_unlocked = search_variables_unlocked(packages_variables_all)
    # only_package_data_unlocked = search_unlocked_data_by_package(packages_variables_unlocked, package_name,package_list)
    # single_variable_max, single_variable_min = cal_variable_max_min(only_package_data_unlocked)
    # cal_variable_max_min(only_package_data_unlocked)
    # single_variable_mean, single_variable_var = cal_variable_means_var(only_package_data_unlocked)
    # cal_variable_means_var(only_package_data_unlocked)


    # out_file_name = input("请输入要导出的Excel文件名：")
    # variables_data2_to_xls(out_file_name, only_package_data_unlocked)
    # variables_data1_to_xls(out_file_name, single_variable_max)
    # variables_data1_to_xls(out_file_name, single_variable_min)

    # variables_data1_to_xls(out_file_name, single_variable_mean)
    # variables_data1_to_xls(out_file_name, single_variable_var)
    # all_files_unlocked_data(package_name)
    out_package_data_unlocked(in_file_name)
_main()
