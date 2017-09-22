import xlrd
from scipy.spatial import KDTree, Voronoi
import numpy as np
from math import cos, pi
import xlwt
import matplotlib.pyplot as plt
import matplotlib.path as mplPath

def data():
    At = xlrd.open_workbook(r'E:\中国移动项目示例代码\base.xlsx')
    At_table = At.sheets()[0]
    id = At_table.col_values(0)
    atname = At_table.col_values(1)
    lon = At_table.col_values(2)
    lat = At_table.col_values(3)
    angle = At_table.col_values(4)
    hight = At_table.col_values(5)
    til = At_table.col_values(6)
    #print("lon1 is %f"%list(lon)[0])
    #print("lat1 is %f"%list(lat)[0])
    return id, atname, lon, lat, angle, hight, til
    
id, atname, lon, lat, angle, hight, til = data()
bdtr = 1.2#bdtr大小决定距离基站距离
vdtr = 1.4#vdtr越小，则过拟合 vdtr越大，则欠拟合
cor = list(zip(lon, lat))
#print(len(cor))
cor = np.array(list(set([tuple(i) for i in cor])))
#print(cor[:,0])
antx = cor[:,0].ravel()
#print(antx)
anty = cor[:,1].ravel()
#antminx = antx.mean()
#antminy = anty.mean()
antminx = min(antx)
antminy = max(anty)
samplex = (antx - antminx) * (111 * cos(antminy * pi/180))
#samplex为采样点距离中心基站距离，xx为采样点经度，cor[0]为基站经度
sampley = (anty - antminy) * 111
#sampley为采样点距离中心基站距离，yy为采样点纬度，cor[1]为基站纬度
cor = np.array(list(zip(samplex, sampley)))
#print(cor)

vor = Voronoi(cor)
plt.plot(cor[:, 0], cor[:, 1], 'o')
#plt.plot(vor.vertices[:, 0], vor.vertices[:, 1], '*')

plt.xlim(min(samplex) - 20, max(samplex) + 20); plt.ylim(min(sampley) - 20, max(sampley) + 20)
#plt.xlim(min(lon) - 0.5, max(lon) + 0.5); plt.ylim(min(lat) - 0.5, max(lat) + 0.5)

corKD = KDTree(cor)
count1 = []
for index, item in enumerate(vor.vertices):
    corKD1 = corKD.query_ball_point(item,vdtr)
    if len(corKD1) < 1:
        count1.append(index)
#print(len(count1))
#print(len(vor.vertices))
#print(len(rvi))
vorrv = vor.ridge_vertices
for i in vorrv:
    for index, item in enumerate(i):
        if item in count1:
            i[index] = -1
#print(vorrv)

#for simplex in vor.ridge_vertices:
'''
for simplex in vorrv:
    simplex = np.asarray(simplex)
    if np.all(simplex >= 0):
        plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-')
'''
#print("vor.ridge_points is %r"%vor.ridge_points)
#print("vor.ridge_vertices is %r"%vor.ridge_vertices)
#print("vor.vertices is \n%r"%vor.vertices)
center = cor.mean(axis=0)
thre_points = []
for pointidx, simplex in zip(vor.ridge_points, vorrv):
    simplex = np.asarray(simplex)
    if np.any(simplex < 0):
        if np.any(simplex > 0):
            i = simplex[simplex >= 0][0] # finite end Voronoi vertex
            t = cor[pointidx[1]] - cor[pointidx[0]]  # tangent
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]]) # normal
            midpoint = cor[pointidx].mean(axis=0)
            far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
            thre_point = midpoint + np.sign(np.dot(midpoint - center, n)) * n * bdtr
            #                       np.sign()计算parameter的正负
            while 1:
                corKD2 = corKD.query_ball_point(thre_point,bdtr)
                if len(corKD2) >= 1:
                    thre_point += np.sign(np.dot(midpoint - center, n)) * n * bdtr
                else:
                    break
            thre_points.append(thre_point)
            plt.plot(thre_point[0], thre_point[1], 'o', c = 'red')
            #plt.plot([vor.vertices[i,0], far_point[0]],
            #         [vor.vertices[i,1], far_point[1]], 'k--')
thre_points = np.array(thre_points)
#print(thre_points)

A=sorted(thre_points,key=lambda x: x[0])
p1=A[0]
pn=A[-1]

Lupper=[]
Llower=[]
#分成上包和下包两类
Lupper.append(p1)
for i in A:
    e=np.array([[i[0],i[1],1],[p1[0],p1[1],1],[pn[0],pn[1],1]])
    flag=np.linalg.det(e)#行列式右手螺旋法则

    if flag > 0 :
        Lupper.append(i)

    else :
        Llower.append(i)
Lupper.append(pn)  #上包      
Llower = sorted(Llower,key=lambda x: x[0],reverse = True)
Lupper.extend(Llower)
Lupper = np.array(Lupper)
#print(Lupper)
bbPath = mplPath.Path(Lupper[0:-2])
px = np.arange(min(thre_points[:, 0]), max(thre_points[:, 0]), 0.02)
py = np.arange(min(thre_points[:, 1]), max(thre_points[:, 1]), 0.02)
pxy = []
for x in px:
    for y in py:
        pxy.append((x,y))
pxy = np.array(pxy)
pxyreal = []
for i in pxy:
    if bbPath.contains_point(i):
        pxyreal.append(i)
pxyreal = np.array(pxyreal)
print(len(pxy))
print(len(pxyreal))
plt.plot(Lupper[:,0], Lupper[:,1], 'k-')
plt.plot(pxyreal[:, 0], pxyreal[:, 1], 'bo')
plt.show()

