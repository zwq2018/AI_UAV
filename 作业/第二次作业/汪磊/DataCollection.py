def ReadData(FlightLog):
    """从源文件中读取数据，建立“变量包：变量”和“变量包：数据”两个二维列表"""
    Variable = []   #“变量包：变量”
    Data = []       #“变量包：数据”
    ARD = 0         #表示无人机的解锁状态
    IllegalPackage = ["FMT", "PARM", "MODE", "MSG", "RTK"]  #不需要的变量包
    for line in FlightLog:
        line = line.replace(',', ' ')
        line = line.split()
        """建立“变量包：变量”列表"""
        if line[0] == 'FMT':
            if line[3] not in IllegalPackage:
                PackageVariable = []  # 存储单个变量包名及其变量
                PackageVariable.append(line[3])
                for key in line[5:]:
                    PackageVariable.append(key)
                Variable.append(PackageVariable)    #将单个变量包及其变量添加到列表中
        """建立“变量包：数据”列表"""
        if line[0] not in IllegalPackage:
            if line[0] == "MKF1":   #记录无人机的解锁状态
                ARD = int(line[10])
            if ARD:
                PackageData = []    #存储单个变量包名及其数据
                PackageData.append(line[0])
                line[1:] = map(float, line[1:]) #把变量数据从字符串转换成数字
                for value in line[1:]:
                    PackageData.append(value)
                Data.append(PackageData)    #将单个变量包名及其数据添加到列表中
    return Variable, Data

import numpy as np
def Caculation(PackageName, Data, Variable):
    """计算单个变量包的相关统计信息"""
    Package = []    #存放单个变量包的所有数据
    """将单个变量包的数据转存到numpy的数组中"""
    for line in Data:
        if PackageName in line:
            Package.append(line[1:])
    arr = np.array(Package)
    """最大值"""
    Maximun = arr.max(0)
    Maximun = Maximun.tolist()
    Maximun.insert(0, PackageName + '\Maximun')
    """最小值"""
    Minimun = arr.min(0)
    Minimun = Minimun.tolist()
    Minimun.insert(0, PackageName + '\Minimun')
    """最小值"""
    Mean = arr.mean(0)
    Mean = Mean.tolist()
    Mean.insert(0, PackageName + '\Mean')
    """方差"""
    Variance = arr.var(0)
    Variance = Variance.tolist()
    Variance.insert(0, PackageName + '\Variance')
    """将统计信息添加到存储变量包数据的列表中"""
    for line in Package:
        line.insert(0, PackageName)
    for line in Variable:
        if PackageName in line: #将变量包的所有变量名插入到列表的第一行
            Package.insert(0, line)
            break
    Package.append(Maximun)
    Package.append(Minimun)
    Package.append(Mean)
    Package.append(Variance)
    return Package

import csv
def ExportPackage(PackageName, Package):
    """输出单个变量包的数据"""
    FileName = PackageName + ".csv"
    with open(FileName, "w", newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(Package)
    print("The data of " + '"' + PackageName + '"' + " has been outputed to " + '"' + FileName + '" successfully.')