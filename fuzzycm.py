
from __future__ import division, print_function
import Tkinter
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import time

start_time = time.time()
colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
import csv
#----------------------------------------------------------------------
def csv_reader(file_obj):
    """
    Read a csv file
    """
    reader = csv.reader(file_obj)
    return reader
#----------------------------------------------------------------------
if __name__ == "__main__":
    csv_path = "food_corpus.csv"
    data=[]
    newdata=[]
    with open(csv_path, "rb") as f_obj:
        data=csv.reader(f_obj)
        i=0
        for row in data:
            if i!=0:
                energy=[float(row[3])-.15*float(row[3]),float(row[3]),float(row[3])+float(row[3])*.15]
                if(energy[2]>=900.0):                                ## 900 is the maximum calorie
                    energy[2]=900.0
                if(energy[0]<=0.0):                                 ##  o is the minimum calorie
                    energy[0]=0.0
                protein=[float(row[4])-.05*float(row[4]),float(row[4]),float(row[4])+float(row[4])*.05]
                fats=[float(row[5])-.05*float(row[5]),float(row[5]),float(row[5])+float(row[5])*.05]
                carbs=[float(row[7])-.05*float(row[7]),float(row[7]),float(row[7])+float(row[7])*.05]
                newdata.append([row[1],energy,protein,fats,carbs])
            i+=1
    #print(len(newdata))


def extract_foods(u_matrix):
    u = u_matrix.T.tolist()
    cluster_analysis = []
    #print(len(u))
    for i in range(0,len(u)):
        max_ele = 0.0
        for j in range(0,len(u[i])):
            max_ele =  max(max_ele,u[i][j])
        for j in range(0,len(u[i])):
            if(float(max_ele) <= float(u[i][j])):
                cluster_analysis.append(j)
    #print(cluster_analysis)
    return cluster_analysis



pts=[]
a = 400.0
b = 500.0
c = 900.0


for i in range(0,int(len(newdata))):
    if(newdata[i][1][1]<a):
        y1  =  float(newdata[i][1][0])/(1.0*newdata[i][1][0] - newdata[i][1][1] + a)
        x1 = y1*a
        y2  =  float(newdata[i][1][2])/(1.0*newdata[i][1][2] - newdata[i][1][1] + a)
        x2 = y2*a
        pts.append([y1*1,y2*1])
    elif(newdata[i][1][1]>=a and newdata[i][1][1]<=b):
        y1 = y2 = float(1)
        pts.append([y1*1,y2*1])
    else:
        y1  =  (c - newdata[i][1][0])/(c + newdata[i][1][1] - b - newdata[i][1][0])
        x1 = y1*(newdata[i][1][2] - newdata[i][1][1]) + newdata[i][1][1]
        y2  =  (c - newdata[i][1][2])/(c + newdata[i][1][1] - newdata[i][1][2] - b)
        x2 = y2*(newdata[i][1][1] - newdata[i][1][2]) + newdata[i][1][2]
        pts.append([y1*1,y2*1])

#print pts
pts = np.asarray(pts)
xpts = pts[:,0]
ypts = pts[:,1]
#print (pts)
#print( xpts)
#print(ypts)
#print (pts)
cluster_pts = []
clusters_objfn = []
iter_no = []
temp=[]
fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
alldata = np.vstack((xpts, ypts))
print
fpcs = []
#print (axes1)
for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
    # Store fpc values for later
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set

    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(xpts[cluster_membership == j],ypts[cluster_membership == j], '.', color=colors[j])

    cluster_pts.append(extract_foods(u))
    clusters_objfn.append(jm)
    iter_no.append(p)
        #for j in range(ncenters):
        #    #cluster_pts.append(x for x in range(xpts[cluster_membership == j]))
        #    print( xpts[cluster_membership == j])

    # Mark the center of each fuzzy cluster
    #print (jm)
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    ax.axis('off')

fig1.tight_layout()

print ("Done Clustering")
end_time = time.time()

print ("Time taken :  ", end_time - start_time)
plt.show()

"""
Bar-chart printing

"""
f,a = plt.subplots(3,3)
a = a.ravel()
for i,ax in enumerate(a):
    #plt.subplot(3,3,i)
    #print (i)
    x_labels = [ x for x in range(i+2)]
    y_labels =[0]*(i+2)
    for j in range(0,len(cluster_pts[i])):
        y_labels[cluster_pts[i][j]]+=1
    print(x_labels,y_labels)
    ax.set_title("No. of points Vs Cluster no.")
    ax.set_xlabel('Cluster no.')
    ax.set_ylabel('No. of points')
    ax.bar( x_labels,y_labels, align='center', alpha=0.5)
    #ax.hist(list(range(10)),y_labels,histtype='bar',rwidth=1)
plt.tight_layout()
plt.show()

"""
graph printing for objective function

"""
f,a = plt.subplots(3,3)
a = a.ravel()
for i,ax in enumerate(a):
    #plt.subplot(3,3,i)
    print(iter_no[i])
    x_labels = range(iter_no[i])
    y_labels =clusters_objfn[i]
    ax.set_title("Objective Function Vs Iteration")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Function')
    ax.scatter( x_labels,y_labels)
    ax.plot(y_labels)
    #ax.hist(list(range(10)),y_labels,histtype='bar',rwidth=1)
plt.tight_layout()
plt.show()
    #print (x_labels,y_labels)
    #plt.hist(x_labels,y_labels,histtype='bar',rwidth='.8')
#plt.show()


age = input("Enter your age : ")
weight = input("Enter weight in kgs : ")
height = input("Enter your height in cms : ")
gender = input("Gender (1 for MALE and 2 for FEMALE) : ")
activity_level = input(" Enter your activity level ( exercise sessions ): \n 1 for no exercise:\n 2 for 1-3 days per week:\n 3 for 3-5 days per week:\n 4 for 6-7 days per week:\n 5 for 2 times per day\n ")
nmeal = input("Enter no. of meal you wish to take( minimum 1 and maximum 5) : ")
bmr = 0

if(gender==1):
    bmr = 10*float(weight) + 6.25*float(height) + 5 - 5*float(age)
else:
    bmr = 10*float(weight) + 6.25*float(height) - 161 - 5*float(age)

al = int(activity_level)
cal = 0
if(al==1):
    cal = bmr * 1.2
elif(al==2):
    cal = bmr * 1.375
elif(al==3):
    cal = bmr * 1.55
elif(al==4):
    cal = bmr * 1.725
elif(al==5):
    cal = bmr * 1.9
else:
    print( "Entered wrong data\n")
print
print ("Calorie intake is (harris bennedict formula : )" + str(cal))
cal = int(cal)
print

print("Proposing your dietary plan from the optimal calorie food cluster")
print
calsum = 0
calfood=[]
error_in_optimal_food_cluster=0
total = 0
#print(cluster_pts[3])
for i in range(0,len(cluster_pts[8])):
    if(int(cluster_pts[8][i])==int(9)):
        calfood.append([newdata[i][0] ,int(newdata[i][1][1]),int(newdata[i][1][1])])

from random import shuffle
shuffle(calfood)



try:
    xrange
except:
    xrange = range

def totalvalue(comb):
    'Totalise a particular combination of items'
    totwt = totval = 0
    for item, wt, val in comb:
        totwt  += wt
        totval += val
    return (totval, -totwt) if totwt <= cal else (0, 0)

items = calfood

def knapsack01_dp(items, limit):
    table = [[0 for w in range(limit + 1)] for j in xrange(len(items) + 1)]

    for j in xrange(1, len(items) + 1):
        item, wt, val = items[j-1]
        for w in xrange(1, limit + 1):
            if wt > w:
                table[j][w] = table[j-1][w]
            else:
                table[j][w] = max(table[j-1][w],
                                  table[j-1][w-wt] + val)

    result = []
    w = limit
    for j in range(len(items), 0, -1):
        was_added = table[j][w] != table[j-1][w]

        if was_added:
            item, wt, val = items[j-1]
            result.append(items[j-1])
            w -= wt

    return result


bagged = knapsack01_dp(items,int(cal))
final_foods  =  [item for item,_,_ in bagged]
val, wt = totalvalue(bagged)
#print("for a total value of %i and a total weight of %i" % (val, -wt))




per_meal = int((len(final_foods))/nmeal)

food_plan = []


"""

"""
print ("Each serving consists of 100 gms/ml each")
i=0
k=0
if(len(final_foods)<=0):
    print ("No food found")
while(i<int(len(final_foods))):
    print ("Meal " + str(k+1))
    j=i
    while(j<=i+per_meal and j<len(final_foods)):
        print (str(final_foods[j]) + "  \t\t\t\t\t100 gms/ml each")
        j+=1
    i=j
    k+=1
