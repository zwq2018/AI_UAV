import pandas as pd
from pandas import DataFrame
import numpy as np
import os

'''
实现功能：一些列表定义为全局变量，并初始化
'''
file_data = []  # 存放从log文件中读取的数据
log_data = []  # 存放有效数据
log_index = []  # 存放有效数据的列索引值，即变量包的各个变量名称

'''
实现功能：扫描文件夹，读取所有log文件并写入list里面
'''


def read_from_log(path):
    files = os.listdir(path)
    for f in files:
        if not os.path.isdir(f):
            filename = path + '\\' + f
            with open(filename) as read_log:
                data = read_log.readlines()
            file_data.append(data)


'''
实现功能：对读取的log文件数据进行处理，只提取有效数据
备注：1.对数据进行分离提取，把每个元素提取处理存放到列表
        因为存在空格，换行和逗号三种分隔符，先把其他两种replace成逗号，然后以逗号为分隔符进行分离
        第一次的代码如下：
            pre_separation_line = current_line
            pre_separation_line = pre_separation_line.replace(' ', '')
            pre_separation_line = pre_separation_line.replace('\n', '')
            post_separation_line = pre_separation_line.split(',')
        改进写法如下：
            post_separation_line = [x.strip() for x in current_line.split(',')]
      2.将log文件的数据分成三种类型:
            1.FMT代表各个变量包的索引名词及格式
            2.PARM、MSG不提供有效数据，可以忽略
            3.其他格式为变量包名词+数据
            先对数据进行三种情况的分类，分别存到list里面
'''


def separated_log_data():
    ard_status = False
    for current_list in file_data:
        for current_line in current_list:
            post_separation_line = [x.strip() for x in current_line.split(',')]  # 分离所有数据
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
                        if post_separation_line not in log_index:
                            log_index.append(post_separation_line)  # 存放每个变量包的变量名称
            elif post_separation_line[0] != 'PARM' and post_separation_line[0] != 'MSG':
                '''
                题目要求只要解锁状态下的数据，也就是MKF1的ARD值为1的情况下
                设置标志位flag来进行标志当前数据是否处于解锁状态
                '''
                if post_separation_line[0] == 'MKF1' and post_separation_line[10] == '1':
                    ard_status = True
                elif post_separation_line[0] == 'MKF1' and post_separation_line[10] == '0':
                    ard_status = False

                if ard_status:  # flag=True代表解锁状态，只要解锁状态才对数据进行处理
                    log_data.append(post_separation_line)


'''
实现功能：用GroupBy技术进行分组聚合
'''


def groupby_log_data():
    global data_grouped
    global log_index_frame
    log_data_frame = DataFrame(log_data)  # 把变量包的变量数据的list转换成DataFrame类型
    log_index_frame = DataFrame(log_index)  # 把变量包的变量数据的list转换成DataFrame类型
    log_index_frame = log_index_frame.sort_index()  # 排序
    data_grouped = log_data_frame.groupby([0])  # 用GroupBy对log_data_frame进行分组处理


'''
实现功能：把数据导到一个excel的多个sheet里面用到ExcelWriter
这里要十分小心，我在这里卡了很长时间，一开始的问题是不能存到一个ecel里面的多个sheet下
最关键是别忘了最后一句保存，如果没有保存的话，程序可以执行，但是文件还不存在，所以在本机找不到这个文件
'''


def write_to_excel(out_name):
    out_writer = pd.ExcelWriter(out_name)
    log_index_frame.to_excel(out_writer, 'Variable packet')  # 输出所有变量包
    for name, groups in data_grouped:
        del groups[0]  # 注意DataFrame的行索引和列索引方式
        for index_name in log_index:
            if index_name[0] == name:
                groups = DataFrame(groups, dtype=np.float)  # 把DataFrame的对象转换成float类型
                groups = groups.dropna(axis=1)
                group_by_name = groups.agg(['max', 'min', 'mean', 'std', np.var])
                group_by_name.columns = index_name[1:]
                group_by_name.to_excel(out_writer, name)
                break
    out_writer.save()  # 千万别忘了写这句


'''
实现功能：主函数
'''
if __name__ == '__main__':
    path = r'F:\项目架构(1)\作业\第二次作业\数据包'
    read_from_log(path)  # 从文件夹中扫描所有文件并读取数据到list

    separated_log_data()  # 分离log文件的数据，提取有效数据，存放到log_data里面
    groupby_log_data()  # 用GroupBy技术进行分组聚合

    path_name = r'C:\Users\zhangwenqi\Desktop\log_data.xlsx'
    write_to_excel(path_name)  # 导出到excel里面
