import DataCollection as dc
import time
import os
filepath = r"E:\USTC\AI+UAV\作业\第二周\作业数据"
if os.path.exists(filepath):
    files = os.listdir(filepath)
    OutputFile = input("Please input the file you want to search: ")
    start = time.clock()
    if OutputFile in files:
        with open(filepath + "\\" + OutputFile) as SourceFile:
            FlightLog = SourceFile.readlines()
            Packages = dc.ReadPackage(FlightLog)
            Variable, Data = dc.ReadData(FlightLog, Packages)
            Feature = dc.Caculation(Packages, Data)
            dc.OutputToCSV(Variable, Data, Feature)
    else:
        print("Please input the effective filename!")
else:
    print("Please input the effective path!")
end = time.clock()
print(end - start)