#Authors: Khashayar Shamsolketabi & Amir Hossein Alikhah Mishamandani
#Date & Time: 27/March/2020
#Description: First Library for Game Theory
#-----------------------------------------------------------------------------
#Required Libraries
from prettytable import PrettyTable
import numpy as np
from itertools import chain

print('Created by: Khashayar Shamsolkotabi & Amir Hossein Alikhah Mishamandani')
print('©PROPRIETARY SOFTWARE LICENSE AGREEMENT, All rights reserved by Khayyam™')
print('Version: v1.0.9.3.2020.a')
print('Version: v1.3.9')


#Variables, Vectors & Matrices
HeadingP1 = []
HeadingP2 = []
PayoffsP1 = []
PayoffsP2 = []
MatrixP1 = [[]]
MatrixP2 = [[]]
Gmin = []
Gmax = []
VecP1 = np.array([],dtype=np.float64)
NewVecP1 = np.array([],dtype=np.float64)
VecP2 = np.array([],dtype=np.float64)
NewVecP2 = np.array([],dtype=np.float64)
HistP1 = np.array([],dtype=np.float64)
HistP2 = np.array([],dtype=np.float64)

#parsing inputs
def str2Vec(InStr):
    string = InStr.split(",")
    out = [float(i) for i in string]
    return out

#Sample for user
def BuildSample():
    print('e.g.: 0,4,5,4,0,5,3,3,6')
    print('e.g.: 4,0,3,0,4,3,5,5,6')
    x = PrettyTable()
    x.field_names = ['','L','C','R']
    x.add_row(['U','0,4','4,0','5,3'])
    x.add_row(['M','4,0','0,4','5,3'])
    x.add_row(['D','3,5','3,5','6,6'])
    print(x)

#Generates the Table
def PayoffsTable(H1,H2,P1,P2,rows,columns):
    out = [[]*columns]*rows
    x = PrettyTable()
    x.title = 'Pay offs'
    x.field_names = H2
    Buff_Str = ''
    mid = []
    k = 0
    for i in range(0,rows):
        mid.append(H1[i+1])
        for j in range(0,columns):
            Buff_Str = str(str(P1[k])+str(',')+str(P2[k]))
            k = k + 1
            mid.append(Buff_Str)
        x.add_row(mid)
        mid = []
    print(x)

#Get Headings
def str2Head(InStr):
    out = InStr.split (",")
    out.insert(0,str(' '))
    return out

#Vec to Matrix
def to_matrix(l, n):

    return [l[i:i+n] for i in range(0, len(l), n)]

#Normalization Func
def Normalize(alpha, betta, x):
    
    return ((x-alpha)/(betta-alpha))

#Normalization - General
def Norm_Mat(V1,rows,columns):
    global Gmin
    global Gmax
    alpha = min(Gmin)
    betta = max(Gmax)
    np_V1 = np.asarray(V1)
    np_V1.astype(float)
    out = np.zeros(rows*columns)
    out.astype(float)
    for i in range(0,rows*columns):   
            out[i] = Normalize(alpha, betta, np_V1[i])
    return out.tolist()

#Update Gains
def UpdateGain(Iter,MP1,MP2,rows,columns):
    global HistP1
    global HistP2
    np_M1 = np.asarray(MP1)
    np_M1.astype(float)
    np_M2 = np.asarray(MP2)
    np_M2.astype(float)
    if(Iter == 0):
        global VecP1 
        VecP1 = np.ones(rows)
        global VecP2 
        VecP2 = np.ones(columns)
        global NewVecP1 
        NewVecP1 = np.zeros(rows)
        global NewVecP2 
        NewVecP2 = np.zeros(columns)
        VecP1 = VecP1 / rows
        VecP2 = VecP2 / columns
        HistP1 = np.append(HistP1,VecP1)
        HistP2 = np.append(HistP2,VecP2)
        print('Iteration ', Iter, ': P1 = ',VecP1,' P2 = ', VecP2)
        for i in range(0,rows):
            for j in range(0,columns):
                NewVecP1[i] = NewVecP1[i] + np_M1[i][j] * VecP2[j]
        for j in range(0,columns):
            for i in range(0,rows):
                NewVecP2[j] = NewVecP2[j] + np_M2[i][j] * VecP1[i]
        NewVecP1 = NewVecP1 / sum(NewVecP1)
        NewVecP2 = NewVecP2 / sum(NewVecP2)
        HistP1 = np.append(HistP1,NewVecP1)
        HistP2 = np.append(HistP2,NewVecP2)
        print('Iteration ', Iter, ': P1 = ',NewVecP1,' P2 = ',NewVecP2)
    if(Iter > 0):
        VecP1 = NewVecP1
        VecP2 = NewVecP2
        NewVecP1 = np.zeros(rows)
        NewVecP2 = np.zeros(columns)
        for i in range(0,rows):
            for j in range(0,columns):
                NewVecP1[i] = NewVecP1[i] + np_M1[i][j] * VecP2[j]
        for j in range(0,columns):
            for i in range(0,rows):
                NewVecP2[j] = NewVecP2[j] + np_M2[i][j] * VecP1[i]
        NewVecP1 = NewVecP1 / sum(NewVecP1)
        NewVecP2 = NewVecP2 / sum(NewVecP2)
        HistP1 = np.append(HistP1,NewVecP1)
        HistP2 = np.append(HistP2,NewVecP2)
        print('Iteration ', Iter, ': P1 = ',NewVecP1,' P2 = ',NewVecP2)
    
#Main Function
def main():

    global m 
    m = int(input('please, insert m for the rows: '))
    global n
    n = int(input('please, insert n for the columns: '))
    print('please, enter the pay-offs for player 1 & 2 as two string seperated with comma:')
    BuildSample()
    global PayoffsP1 
    global PayoffsP2 
    PayoffsP1 = str2Vec(input('Player 1 pay-offs: '))    
    PayoffsP2 = str2Vec(input('Player 2 pay-offs: '))

    if(min(PayoffsP1)<0 or min(PayoffsP2)<0):
        Gmin.append(min(PayoffsP1))
        Gmin.append(min(PayoffsP2))
        Gmax.append(max(PayoffsP1))
        Gmax.append(max(PayoffsP2))
        PayoffsP2 = Norm_Mat(PayoffsP2,m,n)
        PayoffsP1 = Norm_Mat(PayoffsP1,m,n)

    print('please, enter the heading for the P1 pay offs:')
    print('e.g.: U,M,D')
    print('e.g.: L,C,R')
    global HeadingP1
    HeadingP1 = str2Head(input('Player 1 headings: '))
    global HeadingP2 
    HeadingP2 = str2Head(input('Player 2 headings: '))
    PayoffsTable(HeadingP1,HeadingP2,PayoffsP1,PayoffsP2,m,n)
    global MatrixP1 
    MatrixP1 = to_matrix(PayoffsP1,n)
    global MatrixP2
    MatrixP2 = to_matrix(PayoffsP2,n)
    Iter = int(input('please, insert desired number of iterations: '))
    for i in range(0,Iter):
        UpdateGain(i,MatrixP1,MatrixP2,m,n)
     
main()
C = input('Do you want to continue (y/n): ')
if(C == 'y'):
    while(C == 'y'):
        main()
        C = input('Do you want to continue (y/n): ')


#----------------------------------------------------------------------------------------------------------
'''
    global HistP1
    global HistP2
    PlotRes(HistP1,Iter,m,HistP1)
    PlotRes(HistP2,Iter,n,HistP2)

    Out_Vec1 = list(chain.from_iterable(HeadingP1))
    Out_Vec2 = list(chain.from_iterable(HeadingP1))

    #Convergence
    def PlotRes(V1,Iter,RC,Leg):
        l = len(V1)
        x = np.arange(Iter+1)
        ID = 0
        for i in range(0,RC):
            y = np.array([],dtype=np.float64)
            for j in range(i,l,RC):
                y = np.append(y,V1[j])
            plt.plot(x,y)
        plt.title('Performance Function')
        plt.ylabel('Convergence')
        plt.xlabel('Iterations')
        plt.legend(Leg)
        plt.show()
'''





