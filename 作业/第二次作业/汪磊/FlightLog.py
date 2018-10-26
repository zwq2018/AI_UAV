import DataCollection as dc
import os
filepath = r"F:\项目架构(1)\作业\第二次作业\数据包"
files = os.listdir(filepath)
OutputFile = input("Please input the file you want to search: ")
OutputPackage = input("Please input the variable package you want to output:\n" +
                        "(Please input 'ALL' if you want to output all the variable packages.)\n")
OutputPackage = OutputPackage.upper()   #无论用户输入大写还是小写，均转换成大写

if OutputFile in files:
    with open(filepath + "\\" + OutputFile) as sourcefile:
        FlightLog = sourcefile.readlines()
        Variable, Data = dc.ReadData(FlightLog)
        print(Variable)

        if OutputPackage == 'ALL':
            """输出所有变量包"""
            flag = 0    #在原始数据并不是每个变量包都有数据，flag标志变量包是否在Data中有数据
            count = 0   #统计每个文件中有效变量包的个数

            """循环输出单个变量包的所有数据"""
            for line in Variable:
                PackageName = line[0]
                """查找某个变量包是否有数据：若有数据则输出原始数据和统计信息；若没有数据则输出变量包名"""
                for i in range(0, len(Data)):
                    """变量包的数据存在，则输出原始数据和相关统计信息"""
                    if PackageName in Data[i]:
                        Package = dc.Caculation(PackageName, Data, Variable)
                        dc.ExportPackage(PackageName, Package)
                        flag = 1    #变量包的数据存在，则flag记为1
                        count += 1
                        break       #在Data中找到相应变量包，就不再继续遍历
                """变量包的数据不存在，则输出变量包名"""
                if flag == 0:
                    print("The " + '"' + PackageName + '"' + " has none data in " + '"' + OutputFile + '".')
                flag = 0    #为读取下一个变量包做准备
            print("There are " + str(count) + " variable packages'data has been collected in this file.")
        else:
            """输出单个变量包"""
            flag = 0    #标志变量包数据是否存在于待查找的文件中
            for line in Variable:
                if OutputPackage in line:
                    Package = dc.Caculation(OutputPackage, Data, Variable)
                    dc.ExportPackage(OutputPackage, Package)
                    flag = 1
                    break
            if flag == 0:
                print("The " + '"' + OutputPackage + '"' + " has none data in " + '"' + OutputFile + '".')
    print("The " + '"' + OutputFile + '"' + " has been read successfully!\n")
else:
    print("The " + '"' + OutputFile + '"' + " doesn't exist!")


