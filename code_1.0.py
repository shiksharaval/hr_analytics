# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn import tree
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
#from sklearn import cross_validation
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

#loading the dataser contained in a csv file
dataSet = pd.read_csv("HR_comma_sep.csv")

#%% Q4c
"""
#checking to  see missing values in dataset
print(dataSet.isnull().any())

#Data information
print(dataSet.info())

#display a sample of the dataset
print(dataSet.head())

#display data size
print(dataSet.shape)


#display data size
print(dataSet.describe())
"""





#%% Q4c

#satisfaction_level of Employees
satisfactionLevel = dataSet['satisfaction_level']

# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('coolwarm_r')

#Plotting Histogram
n, bins, patches = plt.hist(np.sort(satisfactionLevel), 30)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

plt.ylim(0, 1000)
plt.xlabel('Satisfaction Level')
plt.ylabel('Total Employees')
plt.title('Satisfaction Level of Employees')
plt.show()


#########################################################################################################



#Last Evaluation of Employees
lastEvaluation = dataSet['last_evaluation']
cm = plt.cm.get_cmap('PiYG_r')
n, bins, patches = plt.hist(np.sort(lastEvaluation),30)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)

for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))
plt.xlabel('Last Evaluation')
plt.ylabel('Total Employees')
plt.title('Last Evaluation of Employees')
plt.show()


#########################################################################################################


#number_project of Employees
projNum = dataSet['number_project'].value_counts().to_dict()
print(projNum)
projCounts = collections.OrderedDict(sorted(projNum.items()))
print(projCounts)
ProjKey = list(projCounts.keys())
print(ProjKey)
ProjValue = list(projCounts.values())
print(ProjValue)
df = pd.DataFrame(ProjValue, index = ProjKey)
ax = df.plot(kind='bar', legend = False, width = .5,rot = 0,color = "cadetblue", figsize = (6,5))
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,'%d' % int(height),ha='center', va='bottom')
plt.ylim(0, 5000)
plt.xlabel('Number of projects')
plt.ylabel('Total Employees')
plt.title('Number of Projects of Employees')
plt.show()


########################################################################################################


#Average_montly_hours of Employees
AvgMonthHours = dataSet['average_montly_hours']
cm = plt.cm.get_cmap('PRGn_r')
n, bins, patches = plt.hist(np.sort(AvgMonthHours),30)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)

for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))
plt.xlabel('Average Montly Hours')
plt.ylabel('Total Employees')
plt.title('Average Montly Hours of Employees')
plt.show()


#########################################################################################################


#time_spend_company of Employees
timeSpendComp = dataSet['time_spend_company']
yearsDict = dataSet['time_spend_company'].value_counts().to_dict()
print(yearsDict)
yearsCount = collections.OrderedDict(sorted(yearsDict.items()))
print(yearsCount)
YearsKey = list(yearsCount.keys())
print(YearsKey)
YearsValue = list(yearsCount.values())
print(YearsValue)
df = pd.DataFrame(YearsValue, index = YearsKey)
ax = df.plot(kind='bar', legend = False, width = .5,rot = 0,color = "lightcoral", figsize = (6,5))
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,'%d' % int(height),ha='center', va='bottom')
plt.ylim(0, 7000)
plt.xlabel('Years in Company')
plt.ylabel('Total Employees')
plt.title('Years in Company of Employees')
plt.show()


#########################################################################################################

#Work_accident of Employees
workAccident = dataSet['Work_accident']
workAcc1 = (workAccident == 0).sum()
workAcc2 = (workAccident == 1).sum()
a=[workAcc1,workAcc2]
print(a)
b=['Accident in Work','No Accident in Work']
df = pd.DataFrame(a, index = b)
ax = df.plot(kind='bar', legend = False, width = .5, rot = 0,color = "plum", figsize = (6,5))
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,'%d' % int(height),ha='center', va='bottom')
plt.ylim(0, 15000)
plt.xlabel('Work Accident')
plt.ylabel('Total Employees')
plt.title('Work Accident of Employees')
plt.show()  

#########################################################################################################


#left employment of Employees
leftCompany = dataSet['left']
Left = (leftCompany == 0).sum()
Working = (leftCompany == 1).sum()
a=[Left,Working]
print(a)
b=['Working','Left']
df = pd.DataFrame(a, index = b)
ax = df.plot(kind='bar', legend = False, width = .5, rot = 0,color = "darkkhaki", figsize = (6,5))
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,'%d' % int(height),ha='center', va='bottom')
plt.ylim(0, 15000)
plt.xlabel('Emplotmemt Status')
plt.ylabel('Total Employees')
plt.title('Employees emplotmemt status')
plt.show() 


#########################################################################################################

#Prmotion of Employees
Promotion = dataSet['promotion_last_5years']
noPromotion = (Promotion == 0).sum()
yesPromotion = (Promotion == 1).sum()
a=[noPromotion,yesPromotion]
print(a)
b=['No Promotion','Promotion']
df = pd.DataFrame(a, index = b)
ax = df.plot(kind='bar', legend = False, width = .5, rot = 0,color = "tan", figsize = (6,5))
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,'%d' % int(height),ha='center', va='bottom')
plt.ylim(0, 16000)
plt.xlabel('Employees Promotion Status')
plt.ylabel('Total Employees')
plt.title('Employees Promotion in last 5 years')
plt.show() 


#########################################################################################################

#Departments of Employees
counts = dataSet['sales'].value_counts().to_dict()
print(counts)
dept = list(counts.keys())
print(dept)
deptCount = list(counts.values())
print(deptCount)
df = pd.DataFrame(deptCount, index = dept)
ax = df.plot(kind='bar', legend = False, width = .5,color = "darkseagreen", figsize = (6,5))
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,'%d' % int(height),ha='center', va='bottom')
plt.xlabel('Departments')
plt.ylabel('Total Employees')
plt.title('Employees in Departments')
plt.show() 

#########################################################################################################

#Salary of Employees
salaryCounts = dataSet['salary'].value_counts().to_dict()
print(salaryCounts)
SalaryKey = list(salaryCounts.keys())
print(SalaryKey)
SalaryValue = list(salaryCounts.values())
print(SalaryValue)
df = pd.DataFrame(SalaryValue, index = SalaryKey)
ax = df.plot(kind='bar', legend = False, width = .5,rot = 0,color = "powderblue", figsize = (6,5))
for rect in ax.patches:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,'%d' % int(height),ha='center', va='bottom')
plt.ylim(0, 8000)
plt.xlabel('Salary')
plt.ylabel('Total Employees')
plt.title('Salary of Employees')
plt.show()


print("The Number of employee who left the company :",len(dataSet[dataSet['left']==1]))
print("The Number of employee who didn't left the company",len(dataSet[dataSet['left']==0]))
print("The proportion of employee who left",len(dataSet[dataSet['left']==1])/len(dataSet))

#########################################################################################################
#%% Q4c
"""


#Work_accident of Employees
xs = range(len(b))
plt.bar(xs, a,0.35)
plt.xticks(xs, b)
plt.show()
1.005
1.23, p.get_height() * 1.009))
for p in ax.patches:
   ax.annotate(str(p.get_height()), (p.get_x() * 1.23, p.get_height() * 1.009))
   

"""


"""
def zscore(numArray):
   return (numArray - np.mean(numArray))/np.std(numArray)

zleftCompany = zscore(leftCompany)
zPromotion = zscore(Promotion)
corPromoLeft = zPromotion.dot(zleftCompany) / len(zPromotion)
print("Correlation between Promotion vs Left: ",corPromoLeft)

"""
#########################################################################################################
#%% Q4c
print("--------------------------------------------------------------------------------------------")
print(dataSet.shape)
print(dataSet.iloc[:3,:2])
leftDataSet = dataSet[dataSet['left']==0]
WorkingDataSet = dataSet[dataSet['left']==1]


#Satisfaction Level vs left vs working
leftSatisFacLevel = leftDataSet['satisfaction_level']
workSatisfacLevel = WorkingDataSet['satisfaction_level']
ax = sbn.kdeplot(leftSatisFacLevel, color = 'b', shade = True, label = 'Left company')
ax = sbn.kdeplot(workSatisfacLevel, color = 'g', shade = True, label = 'Working')
ax.set_xlabel('Satisfaction Level')
ax.set_ylabel('Probability Density Function')
ax.set_title("Satisfaction Level Vs Left company Vs Working")
plt.show()

##############################################################################################


#last evaluation Level vs left vs working
leftLastEval = leftDataSet['last_evaluation']
workLastEval = WorkingDataSet['last_evaluation']
ax = sbn.kdeplot(leftLastEval, color = 'b', shade = True, label = 'Left company')
ax = sbn.kdeplot(workLastEval, color = 'g', shade = True, label = 'Working')
ax.set_xlabel('Last Evaluation')
ax.set_ylabel('Probability Density Function')
ax.set_title("Last Evaluation Vs Left company Vs Working")
plt.show()


##############################################################################################


#No of project vs left                         
leftProjDict = leftDataSet['number_project'].value_counts().to_dict()
print(leftProjDict)
leftProjDict = collections.OrderedDict(sorted(leftProjDict.items()))
print(leftProjDict)
leftProjList = list(leftProjDict.values())
leftProjList.extend([0])
print(leftProjList)


#No of project vs working vs working
workProjDict = WorkingDataSet['number_project'].value_counts().to_dict()
print(workProjDict)
workProjDict = collections.OrderedDict(sorted(workProjDict.items()))
print(workProjDict)
workProjList = list(workProjDict.values())
print(workProjList)

n_groups = 6
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.4

fig, ax = plt.subplots()
rects1 = ax.bar(index, leftProjList, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Left company')
rects2 = ax.bar(index + bar_width, workProjList, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Working')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('2', '3', '4', '5','6','7'))
#ax.xticks(index + (bar_width-0.18), ('2', '3', '4', '5','6','7'))
ax.set_xlabel('Number of Projects')
ax.set_ylabel('Employees Count')
ax.set_title("No. of Projects Vs Left company Vs Working")
ax.legend()
ax.set_ylim(0,5000)

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
plt.show()
#gArray[gArray[:, 2]<70,:]


##############################################################################################



#Avg Monthly hours vs left vs working
leftAvgMonthHours = leftDataSet['average_montly_hours']
workAvgMonthHours = WorkingDataSet['average_montly_hours']
ax = sbn.kdeplot(leftAvgMonthHours, color = 'b', shade = True, label = 'Left company')
ax = sbn.kdeplot(workAvgMonthHours, color = 'g', shade = True, label = 'Working')
ax.set_xlabel('Average Montly Hours')
ax.set_ylabel('Probability Density Function')
ax.set_title("Average Montly Hours Vs Left company Vs Working")
plt.show()


##############################################################################################

#No of Promotion vs left                         
leftPromoDict = leftDataSet['promotion_last_5years'].value_counts().to_dict()
print(leftPromoDict)
leftPromoDict = collections.OrderedDict(sorted(leftPromoDict.items()))
print(leftPromoDict)
leftPromoList = list(leftPromoDict.values())
print(leftPromoList)


#No of Promotion vs working 
workPromoDict = WorkingDataSet['promotion_last_5years'].value_counts().to_dict()
print(workPromoDict)
workPromoDict = collections.OrderedDict(sorted(workPromoDict.items()))
print(workPromoDict)
workPromoList = list(workPromoDict.values())
print(workPromoList)


n_groups = 2
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.4

fig, ax = plt.subplots()
rects1 = ax.bar(index, leftPromoList, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Left company')
rects2 = ax.bar(index + bar_width, workPromoList, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Working')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('No Promotion', 'Promotion'))
#ax.xticks(index + (bar_width-0.18), ('2', '3', '4', '5','6','7'))
ax.set_xlabel('Promotion in last 5 Years')
ax.set_ylabel('Employees Count')
ax.set_title("Promotion Vs Left company Vs Working")
ax.legend()
ax.set_ylim(0,13000)

def autolabel(rects):
   
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
plt.show()


############################################################################################3

#No of time_spend_company vs left                         
leftTimeDict = leftDataSet['time_spend_company'].value_counts().to_dict()
print(leftTimeDict)
leftTimeDict = collections.OrderedDict(sorted(leftTimeDict.items()))
print(leftTimeDict)
leftTimeList = list(leftTimeDict.values())
print(leftTimeList)


#No of Promotion vs working vs left
workTimeDict = WorkingDataSet['time_spend_company'].value_counts().to_dict()
print(workTimeDict)
workTimeDict = collections.OrderedDict(sorted(workTimeDict.items()))
print(workTimeDict)
workTimeList = list(workTimeDict.values())
workTimeList.extend([0,0,0])
print(workTimeList)


n_groups = 8
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.4

fig, ax = plt.subplots()
rects1 = ax.bar(index, leftTimeList, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Left company')
rects2 = ax.bar(index + bar_width, workTimeList, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Working')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('2', '3', '4', '5','6','7','8','10'))
#ax.xticks(index + (bar_width-0.18), ('2', '3', '4', '5','6','7'))
ax.set_xlabel('Time Spent in Company')
ax.set_ylabel('Employees Count')
ax.set_title("Time in company Vs Left company Vs Working")
ax.legend()
ax.set_ylim(0,7000)

def autolabel(rects):
   
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom',rotation='vertical')
autolabel(rects1)
autolabel(rects2)
plt.show()



############################################################################################3

#Sales vs left                         
leftSalesDict = leftDataSet['sales'].value_counts().to_dict()
print(leftSalesDict)
#leftSalesDict = collections.OrderedDict(sorted(leftSalesDict.items()))
#print(leftSalesDict)
leftSalesList = list(leftSalesDict.values())
print(leftSalesList)


#Sales vs working vs left
workSalesDict = WorkingDataSet['sales'].value_counts().to_dict()
print(workSalesDict)
#workSalesDict = collections.OrderedDict(sorted(workSalesDict.items()))
#print(workSalesDict)
workSalesList = list(workSalesDict.values())
print(workSalesList)


n_groups = 10
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.4

fig, ax = plt.subplots()
rects1 = ax.bar(index, leftSalesList, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Left company')
rects2 = ax.bar(index + bar_width, workSalesList, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Working')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('sales', 'accounting', 'hr', 'technical','support','management','IT','product_mgt','Marketing','RandD'),rotation=70)
#ax.xticks(index + (bar_width-0.18), ('2', '3', '4', '5','6','7'))
ax.set_xlabel('Departments')
ax.set_ylabel('Employees Count')
ax.set_title("Departments Vs Left company Vs Working")
ax.legend()
ax.set_ylim(0,5000)

def autolabel(rects):
   
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.15*height,
                '%d' % int(height),
                ha='center', va='bottom',rotation='vertical')
autolabel(rects1)
autolabel(rects2)
plt.show()



##################################################################################################

#No of time_spend_company vs left                         
leftSalaryDict = leftDataSet['salary'].value_counts().to_dict()
print(leftSalaryDict)
#leftSalaryDict = collections.OrderedDict(sorted(leftSalaryDict.items()))
#print(leftSalaryDict)
leftSalaryList = list(leftSalaryDict.values())
print(leftSalaryList)


#No of Promotion vs working vs left
workSalaryDict = WorkingDataSet['salary'].value_counts().to_dict()
print(workSalaryDict)
#workSalaryDict = collections.OrderedDict(sorted(workSalaryDict.items()))
#print(workSalaryDict)
workSalaryList = list(workSalaryDict.values())
print(workSalaryList)


n_groups = 3
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.4

fig, ax = plt.subplots()
rects1 = ax.bar(index, leftSalaryList, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Left company')
rects2 = ax.bar(index + bar_width, workSalaryList, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Working')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('Low', 'Medium', 'High'))
ax.set_xlabel('Salary')
ax.set_ylabel('Employees Count')
ax.set_title("Salary Vs Left company Vs Working")
ax.legend()
ax.set_ylim(0,7000)

def autolabel(rects):
   
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
plt.show()



##############################################################################################

#No of Promotion vs left                         
leftWorkAccDict = leftDataSet['Work_accident'].value_counts().to_dict()
print(leftWorkAccDict)
leftWorkAccDict = collections.OrderedDict(sorted(leftWorkAccDict.items()))
print(leftWorkAccDict)
leftWorkAccList = list(leftWorkAccDict.values())
print(leftWorkAccList)


#No of Promotion vs working vs left
workAccDict = WorkingDataSet['Work_accident'].value_counts().to_dict()
print(workAccDict)
workAccDict = collections.OrderedDict(sorted(workAccDict.items()))
print(workAccDict)
workAccList = list(workAccDict.values())
print(workAccList)


n_groups = 2
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.4

fig, ax = plt.subplots()
rects1 = ax.bar(index, leftWorkAccList, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Left company')
rects2 = ax.bar(index + bar_width, workAccList, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Working')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('No Accident', 'Accident'))
#ax.xticks(index + (bar_width-0.18), ('2', '3', '4', '5','6','7'))
ax.set_xlabel('Accident in work')
ax.set_ylabel('Employees Count')
ax.set_title("Accident in work Vs Left company Vs Working")
ax.legend()
ax.set_ylim(0,12000)

def autolabel(rects):
   
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
plt.show()
##############################################################################################
        


#%% Q4c

print("--------------------------------------------------------------------------------------------")

def zscore(numArray):
   return (numArray - np.mean(numArray))/np.std(numArray)

#modified columns to calculate corelation
modifiedDataSet = dataSet
modifiedDataSet['sales'].replace(['sales', 'accounting', 'hr', 'technical', 'support', 'management',
        'IT', 'product_mng', 'marketing', 'RandD'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace = True)
modifiedDataSet['salary'].replace(['low', 'medium', 'high'], [0, 1, 2], inplace = True)

#print(modifiedDataSet.head())



#satisfaction vs other columns
SatvsOthers = []
zSatisfaction = zscore(satisfactionLevel)
for i in range(10):
    colValues = modifiedDataSet.iloc[:,i]
    zOtherCol = zscore(colValues)
    corrValue = zOtherCol.dot(zSatisfaction) / len(zOtherCol)
    SatvsOthers.append(corrValue)
print("Correlation between Satisfaction vs Other Columns: ",SatvsOthers)


#####################################################################################################

#LastEval vs other columns
LastEvalvsOthers = []
zLastEval = zscore(lastEvaluation)
for i in range(10):
    colValues = modifiedDataSet.iloc[:,i]
    zOtherCol = zscore(colValues)
    corrValue = zOtherCol.dot(zLastEval) / len(zOtherCol)
    LastEvalvsOthers.append(corrValue)
print("\nCorrelation between Last Evaluation vs Other Columns: ",LastEvalvsOthers)


#####################################################################################################


#No. of Project vs other columns
ProjvsOthers = []
projNum = dataSet['number_project']
zProj = zscore(projNum)
for i in range(10):
    colValues = modifiedDataSet.iloc[:,i]
    zOtherCol = zscore(colValues)
    corrValue = zOtherCol.dot(zProj) / len(zOtherCol)
    ProjvsOthers.append(corrValue)
print("\nCorrelation between No. of Project vs Other Columns: ",ProjvsOthers)


#####################################################################################################

#AvgMonthHours vs other columns
AvgMonHourvsOthers = []
zAvgMonHours = zscore(AvgMonthHours)
for i in range(10):
    colValues = modifiedDataSet.iloc[:,i]
    zOtherCol = zscore(colValues)
    corrValue = zOtherCol.dot(zAvgMonHours) / len(zOtherCol)
    AvgMonHourvsOthers.append(corrValue)
print("\nCorrelation between Avg Monthly Hours vs Other Columns: ",AvgMonHourvsOthers)


#####################################################################################################


#Time Spent vs other columns
timeSpentvsOthers = []
zTimeSpent = zscore(timeSpendComp)
for i in range(10):
    colValues = modifiedDataSet.iloc[:,i]
    zOtherCol = zscore(colValues)
    corrValue = zOtherCol.dot(zTimeSpent) / len(zOtherCol)
    timeSpentvsOthers.append(corrValue)
print("\nCorrelation between Time Spent vs Other Columns: ",timeSpentvsOthers)


#####################################################################################################


#workAccident vs other columns
WorkAccvsOthers = []
zWorkAcc = zscore(workAccident)
for i in range(10):
    colValues = modifiedDataSet.iloc[:,i]
    zOtherCol = zscore(colValues)
    corrValue = zOtherCol.dot(zWorkAcc) / len(zOtherCol)
    WorkAccvsOthers.append(corrValue)
print("\nCorrelation between workAccident vs Other Columns: ",WorkAccvsOthers)


#####################################################################################################

#left vs other columns
LeftvsOthers = []
zleftCompany = zscore(leftCompany)
for i in range(10):
    colValues = modifiedDataSet.iloc[:,i]
    zOtherCol = zscore(colValues)
    corrValue = zOtherCol.dot(zleftCompany) / len(zOtherCol)
    LeftvsOthers.append(corrValue)
print("\nCorrelation between Left vs Other Columns: ",LeftvsOthers)


#####################################################################################################


#Promotion vs other columns
PromovsOthers = []
zPromotion = zscore(Promotion)
for i in range(10):
    colValues = modifiedDataSet.iloc[:,i]
    zOtherCol = zscore(colValues)
    corrValue = zOtherCol.dot(zPromotion) / len(zOtherCol)
    PromovsOthers.append(corrValue)
print("\nCorrelation between Promotion vs Other Columns: ",PromovsOthers)



#Sales vs other columns
SalesvsOthers = []
sales = modifiedDataSet['sales']
zSales = zscore(sales)
for i in range(10):
    colValues = modifiedDataSet.iloc[:,i]
    zOtherCol = zscore(colValues)
    corrValue = zOtherCol.dot(zSales) / len(zOtherCol)
    SalesvsOthers.append(corrValue)
print("\nCorrelation between Sales vs   Other Columns: ",SalesvsOthers)


#####################################################################################################

#Salary vs other columns
SalaryvsOthers = []
salary = modifiedDataSet['salary']
zSalary = zscore(salary)
for i in range(10):
    colValues = modifiedDataSet.iloc[:,i]
    zOtherCol = zscore(colValues)
    corrValue = zOtherCol.dot(zSalary) / len(zOtherCol)
    SalaryvsOthers.append(corrValue)
print("\nCorrelation between salary vs Other Columns: ",SalaryvsOthers)

"""
Spearman
salarySpear = []
salary_rank = np.argsort(np.argsort(salary))
for i in range(10):
    colValues = modifiedDataSet.iloc[:,i]
    col_rank = np.argsort(np.argsort(colValues))
    SpearCorr = np.corrcoef(salary_rank, col_rank)[0,1]
    salarySpear.append(SpearCorr)
print("\nSpearman Salary vs othe col",salarySpear)
"""



###################################################################################################
#%% Q4c
print("-----------------------------------------------------------------------------------------------")

def zscoreBoxplt(series): 
    return (series - series.mean(skipna=True)) / series.std(skipna=True);

dataSet2 = modifiedDataSet.apply(zscoreBoxplt)
dataSet2.boxplot()
plt.show()


def scaling(series): 
        return (series - series.min()) / (series.max() - series.min())

dataSet3 = modifiedDataSet.apply(scaling)
dataSet3.boxplot()
plt.show()

 
##############################################################################################
#%% Q4c

print("-----------------------------------------------------------------------------------------------")


# Convert these variables into categorical variables
dataSet["sales"] = dataSet["sales"].astype('category').cat.codes
dataSet["salary"] = dataSet["salary"].astype('category').cat.codes

#print(dataSet.head())

# Create train and test splits
X = dataSet.drop('left', axis=1)
#print(X.head())

Y=dataSet['left']
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.25, random_state=123, stratify=Y)

dectree = tree.DecisionTreeClassifier(min_samples_leaf=50)
dectree = dectree.fit(X, Y)
dot_data = tree.export_graphviz(dectree, None)
#print("\nDecison Tree  - dot data\n\n",dot_data)

featImportance = dectree.feature_importances_
featNames = X.columns
featImpList = pd.Series(featImportance,index=featNames).sort_values(ascending=False)
print("Decision Tree Features \n",featImpList)

pred = dectree.predict(X)
ConfusionMatrix = metrics.confusion_matrix(Y, pred)
print("\nConfusionMatrix\n",ConfusionMatrix)

precisionVal = metrics.precision_score(Y,pred)
recallVal = metrics.recall_score(Y,pred)
f1Val = metrics.f1_score(Y,pred)
KappaVal = metrics.cohen_kappa_score(Y, pred)
print("Precision = ",precisionVal)
print("Recall = ",recallVal)
print("F1 measure = ",f1Val)
print("Kappa = ", KappaVal)


###########################################################################################





