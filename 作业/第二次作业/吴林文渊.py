#该程序可以提取选定的log文件中指定的变量包的全部数据

from typing import List, Any
import xlwt
import pandas as pd
from pandas import DataFrame
import numpy as np

chache = []  # 把需要输入的数据放入chache  包括变量名 数据  以及最大最小值等  用chache 做整体的格式变换

def  findlog(f,logname):#找到logname的所有的变量名  只有前100多行读取，对速度影响不大
    lognameparameter = ['']
    print(logname)
    parameternum=0
    for eachline in f:
        eachline = eachline.strip('\n')#去除换行符
        if eachline[:3] == 'FMT':
            #print('eachline')
            alllogname = eachline.split(',')
            if alllogname[3] == ' '+logname:
                lognameparameter = alllogname[4:].copy()#' FMT', ' BBnNZ', ' Type', 'Length', 'Name', 'Format', 'Columns\n']
                parameternum = len(lognameparameter)
                break#找到就不需要向下再找了
    return lognameparameter,parameternum

def countandintnum(f,logname):#找到logname的所有的数据 用list保存起来
    flag=0
    intnum=[]
    count=0
    lognameparameter, parameternum = findlog(f, logname)
    f.seek(0)
    for eachline in f:
        if eachline[:4] == 'MKF1':
            temp = eachline.split(',')
            if temp[10] == ' 1':  # 解锁状态
                flag = 1
            else:
                flag = 0
            continue
        if flag == 1:
            temp = ['']
            temp = eachline.split(',')
            if temp[0] == logname:
                cop = temp[1:]  # 第一个是名字
                t = list(map(float, cop))
                intnum.append(t)
                chache.append(t)  #直接填入全局变量
                t = t * 0
                count = count + 1
    del intnum[0]
    return count-1,intnum

def dataanaly(intnum,parameternum):#找出最大值最小值以及平均数 方差
    bt=np.array(intnum).T
    max=[]
    min=[]
    ave=[]
    var=[]
    for i in range(0,parameternum-1):
        max.append(np.max(bt[i]))
    for i in range(0,parameternum-1):
        min.append(np.min(bt[i]))
    for i in range(0, parameternum - 1):
        ave.append(np.mean(bt[i]))
    for i in range(0, parameternum - 1):
        var.append(np.var(bt[i]))
    return max,min,ave,var

def main ():

    file = input('你所需要的文件在哪? : ')
    # path=input('你想存储在哪里： ')
    logname = input('你需要哪个包：')
    while logname == 'FMT' or logname == 'PARM' or logname == 'MODE' or logname == 'MSG':
        print('无法提取该变量包')
        logname = input('你需要哪个包：')

    f=open(file,'r')
    f.seek(0)

    lognameparameter, parameternum = findlog(f, logname)#找到所有的变量名称
    del lognameparameter[0]#第一个为无效的变量 QBIHBcLLefffB等

    path_save=logname+'.xls'
    #path_save=path+logname+'.xls'

    out_writer = pd.ExcelWriter(path_save)

    chache.append(lognameparameter)#变量名加入表头  领先于数据填入
    f.seek(0)
    count, intnum = countandintnum(f, logname)  # 找到所有的数据行数
    del chache[1]
    f.seek(0)

    max, min, ave, var=dataanaly(intnum, parameternum)

    #加入最大最小值
    max.append('MAX')
    min.append('MIN')
    ave.append('AVE')
    var.append('VAR')
    chache.append(max)
    chache.append(min)
    chache.append(ave)
    chache.append(var)

    logname_frame = pd.DataFrame(chache)#转换格式
    logname_frame.to_excel(out_writer, 'value')#整体写入
    f.close()
    out_writer.save()
main()
