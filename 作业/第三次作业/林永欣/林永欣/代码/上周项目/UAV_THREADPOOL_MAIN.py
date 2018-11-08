from UAV_DATA import *
import time
import os
from concurrent.futures import ThreadPoolExecutor


def main():
    file_name_list = []
    global variable_package_name, variable_name

    '''输入路径、要查询的变量包和变量名称'''
    file_path = input('Please input your file path:')
    variable_package_name = input('Please input variable_package name:')
    variable_name = input('Please input variable name:')

    '''获取文件夹里所有文件，存到列表中'''
    files = os.listdir(file_path)
    for current_file in files:
        if not os.path.isdir(current_file):  # 如果是文件的话才执行
            current_file_name = file_path + '\\' + current_file
            file_name_list.append(current_file_name)  # 存放需要处理的文件名

    '''设置线程池最大容量是4个线程'''
    pool = ThreadPoolExecutor(4)

    '''创建读文件任务，让线程池进行处理'''
    for current_file in file_name_list:
        pool.submit(read_data_from_log, current_file)  # 从log中读取数据

    time.sleep(10)

    '''创建写文件任务，让线程池进行处理'''
    for file_name in file_name_list:
        pool.submit(write_data_to_excel, file_name, variable_package_name, variable_name)  # 汇总结果输出到Excel


if __name__ == '__main__':
    main()
