def ReadPackage(FlightLog):
    Packages = []   # “变量包”列表
    IllegalPackage = ["FMT", "PARM", "MODE", "MSG", "RTK"]  # 不需要的变量包
    for i in range(0, len(FlightLog)):
        FlightLog[i] = FlightLog[i].replace(',', ' ').split()
        if FlightLog[i][0] == "FMT":
            if FlightLog[i][3] not in IllegalPackage:
                Packages.append(FlightLog[i][3])
    return Packages

def ReadData(FlightLog, Packages):
    Variable = dict([(key, []) for key in Packages])    #"变量包：变量"字典
    Data = dict([(key, []) for key in Packages])        #“变量包：数据”字典
    ARD = 0         #表示无人机的解锁状态
    for line in FlightLog:
        """建立“变量包：变量”字典"""
        if line[0] == "FMT":
            if line[3] in Packages:
                Variable[line[3]].append(line[3])
                for VariableName in line[5:]:
                    Variable[line[3]].append(VariableName)
        """建立“变量包：数据”字典"""
        if line[0] in Data.keys():     # Data.keys()返回一个列表，占用时间和空间
            if line[0] == "MKF1":
                ARD = int(line[10])     #记录无人机的解锁状态
            if ARD:
                PackageData = []  # 存储单个变量包名及其数据
                line[1:] = map(float, line[1:])
                for value in line[1:]:
                    PackageData.append(value)
                Data[line[0]].append(PackageData)
    return Variable, Data

import numpy as np
def Caculation(Packages, Data):
    Feature = dict([(key, []) for key in Packages])     #“变量包：统计数据”字典
    for PackageName in Packages:
        if Data[PackageName]:   #有些变量包没有数据
            arr = np.array(Data[PackageName])
            """最大值"""
            Maximun = arr.max(0)
            Maximun = Maximun.tolist()
            Maximun.insert(0, 'Maximun')
            Feature[PackageName].append(Maximun)
            """最小值"""
            Minimun = arr.min(0)
            Minimun = Minimun.tolist()
            Minimun.insert(0, 'Minimun')
            Feature[PackageName].append(Minimun)
            """最小值"""
            Mean = arr.mean(0)
            Mean = Mean.tolist()
            Mean.insert(0, 'Mean')
            Feature[PackageName].append(Mean)
            """方差"""
            Variance = arr.var(0)
            Variance = Variance.tolist()
            Variance.insert(0, 'Variance')
            Feature[PackageName].append(Variance)
            # """调整输出格式：在原始数据每一行的前面加上变量包名"""
            # for line in Data[PackageName]:
            #     line.insert(0, PackageName)
    return Feature

import csv
def OutputToCSV(Variable, Data, Feature):
    #with open("AllPackages.csv", "w", newline = '') as csvfile:
    with open("AllPackages_none.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for PackageName in Variable:
            writer.writerow(Variable[PackageName])  #输出每个变量包包含的变量名
            # for values in Data[PackageName]:      #输出每个变量包的所有数据
            #     writer.writerow(values)
            for num in Feature[PackageName]:        #输出每个变量包的相关统计信息
                writer.writerow(num)
            writer.writerow('\n')