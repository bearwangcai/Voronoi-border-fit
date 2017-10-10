import xlrd
from scipy.spatial import KDTree, Voronoi
import numpy as np
from math import cos, pi
import xlwt
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from copy import deepcopy

samplex = []
sampley = []
for x in range(-6,26):
    for y in range(-2,4):
        samplex.append(x)
        sampley.append(y)
for x in range(-6,-1):
    for y in range(3,10):
        samplex.append(x)
        sampley.append(y)
for x in range(-6,5):
    for y in range(10, 16):
        samplex.append(x)
        sampley.append(y)
for x in range(20,26):
    for y in range(-7, -1):
        samplex.append(x)
        sampley.append(y)
for x in range(15,26):
    for y in range(-13,-6):
        samplex.append(x)
        sampley.append(y)
bdtr = 10#bdtr大小决定距离基站距离
vdtr = 1#vdtr越小，则过拟合 vdtr越大，则欠拟合
cor = np.array(list(zip(samplex, sampley)))
#print(cor)


vor = Voronoi(cor)
plt.plot(cor[:, 0], cor[:, 1], 'o')
plt.plot(vor.vertices[:, 0], vor.vertices[:, 1], '*')

plt.xlim(min(samplex) - 20, max(samplex) + 20); plt.ylim(min(sampley) - 20, max(sampley) + 20)
#plt.xlim(min(lon) - 0.5, max(lon) + 0.5); plt.ylim(min(lat) - 0.5, max(lat) + 0.5)

'''
如果vor.vertices在vdtr范围内没有基站，同样认为是无效顶点，即为无穷远点
'''

corKD = KDTree(cor)
count1 = []

for index, item in enumerate(vor.vertices):
    corKD1 = corKD.query_ball_point(item,vdtr)
    if len(corKD1) < 1:
        count1.append(index)
#print(len(count1))
#print(len(vor.vertices))
#print(len(rvi))
plt.plot(vor.vertices[count1, 0], vor.vertices[count1, 1], 'r*')
#print(vor.ridge_vertices)
vorrv = deepcopy(vor.ridge_vertices)
for i in vorrv:
    for index, item in enumerate(i):
        if item in count1:
            i[index] = -1
#print(type(vorrv[0]))
#print(vor.ridge_vertices)


'''
寻找基站原始外围边界点
'''
boundaryoripoints = []
for index, item in enumerate(vorrv):
    #print(item)
    item = np.array(item)
    #print(item)
    if np.any(item < 0):
        if np.any(item > 0):
            boundaryoripoints.append(vor.ridge_points[index])
boundaryoripoints = np.array(boundaryoripoints)
boundaryoripointsfla = boundaryoripoints.flatten()
#boundaryoripoints = set(list(boundaryoripoints))
#boundaryoripointsid = list(boundaryoripoints)
#print(boundaryoripoints)
#print(cor[77],cor[287],cor[71],cor[83])



bop = list(deepcopy(boundaryoripointsfla))
dictbop = {}
for key in bop:
    dictbop[key] = dictbop.get(key, 0) + 1
#print(dictbop)



'''
寻找边界初始位置，以tempini表示该点
'''
for i in bop:
    if dictbop[i] == 2:
        tempini = i
        break
        
niu = []#边界
niutemp = []#临时边界
tempid = bop.index(tempini)

while(len(bop)):
    '''
    将两点相连
    '''

    if dictbop[tempini] <= 2:
        niutemp.append(tempini)
        #print("tempid is %d"%tempid)
        #print("length is %d"%len(bop))
        if tempid%2 == 0:
            tempini = bop[tempid + 1]
            del bop[tempid]
            del bop[tempid]
        else:
            tempini = bop[tempid - 1]
            del bop[tempid - 1]        
            del bop[tempid - 1]
        if len(bop) > 0:
            try:
                tempid = bop.index(tempini)
            except ValueError as e:
                niu.append(niutemp)
                niutemp = []
                tempini = bop[0]
                tempid = bop.index(tempini)
niu.append(niutemp)

t_oripoints = [cor[i] for i in boundaryoripointsfla]
tlen = len(t_oripoints)
t_oripoints = np.array(t_oripoints)
plt.plot(t_oripoints[:, 0], t_oripoints[:, 1], 'yo')

#print(niu)


for i in niu:
    #print(i)
    t_pl = [list(cor[j]) for j in i]
    iCD = KDTree(t_pl)
    t_plnp = np.array(t_pl)
    bp = []
    bpnum = []
    bbPath = mplPath.Path(t_plnp[:])
    for k in cor:
        if not bbPath.contains_point(k):
            bp.append(list(k))
    for k in bp:
        if not k in t_pl:
            bpnum.append(k)
    if len(bpnum) != 0:
        for l in bpnum:
            t_pl.append(iCD.query(l),l)
            iCD = KDTree(t_pl)
        t_plnp = np.array(t_pl)
    plt.plot(t_plnp[:, 0], t_plnp[:, 1], 'k-')
    plt.plot(t_plnp[[-1, 0], 0], t_plnp[[-1, 0], 1], 'k-')
plt.show()
#print(t_points)

