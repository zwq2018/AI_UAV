import pandas as pd
from pandas import DataFrame
import numpy as np
import os
import time

'''
实现功能：一些列表定义为全局变量，并初始化
'''
log_data = []  # 存放有效数据
log_index = {}  # 字典形式存放有效数据的列索引值，即变量包的各个变量名称
user_output = DataFrame()
'''
实现功能：对读取的log文件数据进行处理，只提取有效数据
备注：1.对数据进行分离提取，把每个元素提取处理存放到列表
        因为存在空格，换行和逗号三种分隔符，先把其他两种replace成逗号，然后以逗号为分隔符进行分离
      2.将log文件的数据分成三种类型:
            1.FMT代表各个变量包的索引名词及格式
            2.PARM、MSG不提供有效数据，可以忽略
            3.其他格式为变量包名称+数据
            先对数据进行三种情况的分类，分别存到list里面
      3.设置静态变量ard_status进行解锁状态的标志，用到python的装饰器
        注意python中没有像C++那种静态变量static  
'''


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(ard_status=False)
def separated_log_data(file_data):
    post_separation_line = [x.strip() for x in file_data.split(',')]  # 分离所有数据
    '''
    处理数据属于FMT时的情况,
    如果第三位是FMT，代表这个是FMT的变量包格式说明，不处理
    如果第三位是PARM和MSG，也视为无效变量包，不处理
    其他情况则进行FMT变量包的变量数据分离并存到log_fmt_index
    '''
    if post_separation_line[0] == 'FMT':
        if post_separation_line[3] != 'FMT':
            if post_separation_line[3] != 'PARM' and post_separation_line[3] != 'MSG':
                del post_separation_line[4]  # 无效数据，删去
                del post_separation_line[0:3]  # 删除前三个元素，因为无效数据
                if post_separation_line[0] not in log_index.keys():
                    log_index[post_separation_line[0]] = post_separation_line[1:]  # 字典的方式存放每个变量包的变量名称
    elif post_separation_line[0] != 'PARM' and post_separation_line[0] != 'MSG':  # 第三种情况的处理
        '''
        题目要求只要解锁状态下的数据，也就是MKF1的ARD值为1的情况下
        设置标志位ard_status来进行标志当前数据是否处于解锁状态
        '''
        if post_separation_line[0] == 'MKF1' and post_separation_line[10] == '1':
            separated_log_data.ard_status = True
        elif post_separation_line[0] == 'MKF1' and post_separation_line[10] == '0':
            separated_log_data.ard_status = False

        if separated_log_data.ard_status:  # ard_status=True代表解锁状态，只要解锁状态才对数据进行处理
            log_data.append(post_separation_line)


'''
实现功能：用GroupBy技术进行分组聚合
'''


def groupby_log_data():
    global data_grouped
    log_data_frame = DataFrame(log_data)  # 把变量包的变量数据的list转换成DataFrame类型
    data_grouped = log_data_frame.groupby([0])  # 用GroupBy对log_data_frame进行分组处理


'''
实现功能：把数据导到一个excel的多个sheet里面用到ExcelWriter
这里要十分小心，我在这里卡了很长时间，一开始的问题是不能存到一个excel里面的多个sheet下
最关键是别忘了最后一句保存，如果没有保存的话，程序可以执行，但是文件还不存在，所以在本机找不到这个文件
'''


def write_to_excel(out_name, file_name):
    out_writer = pd.ExcelWriter(out_name)
    # log_index_frame.to_excel(out_writer, 'Variable packet')  # 输出所有变量包
    for name, groups in data_grouped:
        del groups[0]  # 删除第一列，第一列全部位变量名称，注意DataFrame的行索引和列索引方式
        groups = DataFrame(groups, dtype=np.float)  # 把DataFrame的对象转换成float类型
        groups = groups.dropna(axis=1)  # 把全为NA的列删去
        group_data = groups.agg(['max', 'min', 'mean', 'std', np.var])
        group_data.columns = log_index[name]  # 修改列索引
        group_data.to_excel(out_writer, name)  # 把汇总信息输出到Excel表格里面
        if name == variable_package_name:  # 保存用户需要的信息
            user_output[file_name] = group_data[variable_name]
    out_writer.save()  # 千万别忘了写这句


'''
实现功能：主函数
'''


def main():
    global variable_package_name
    global variable_name
    path = input('请输入文件的路径：')
    variable_package_name = input('请输入要查询的变量包名称：')
    variable_name = input('请输入要查询的变量名称：')
    files = os.listdir(path)
    for file_name in files:
        if not os.path.isdir(file_name):  # 如果是文件的话才执行
            current_file = path + '\\' + file_name
            for file_data in open(current_file):
                separated_log_data(file_data)  # 分离数据并存到list里面

            groupby_log_data()  # 用GroupBy技术进行分组聚合

            path_name = current_file.replace('.log', '.xlsx')
            write_to_excel(path_name, file_name)  # 导出到excel里面
    print(user_output)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('运行时间：', end-start)
