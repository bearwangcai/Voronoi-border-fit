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
vorrv = vor.ridge_vertices
for i in vorrv:
    for index, item in enumerate(i):
        if item in count1:
            i[index] = -1
#print(vorrv)

'''
寻找基站外围边界点
'''
boundarypoints = []
for index, item in enumerate(vorrv):
    #print(item)
    item = np.array(item)
    #print(item)
    if np.any(item < 0):
        boundarypoints.append(vor.ridge_points[index])
boundarypoints = np.array(boundarypoints)
boundarypoints = boundarypoints.flatten()
boundarypoints = set(list(boundarypoints))
boundarypointsid = list(boundarypoints)
#print(boundarypointsid)

t_points = [cor[i] for i in boundarypointsid]
tlen = len(t_points)
t_points = np.array(t_points)
#plt.plot(t_points[:, 0], t_points[:, 1], 'ro')
#plt.show()
#print(t_points)
print(tlen)
def LUlo():
    A=sorted(t_points,key=lambda x: x[0])
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
    return Lupper
    
while 1:
    Lupper = LUlo()
    #print(Lupper)
    bbPath = mplPath.Path(Lupper[0:-2])
    pxyreal = []
    for i in cor:
        if not bbPath.contains_point(i):
            pxyreal.append(i)
    #print(pxyreal)
    t_points = list(t_points)
    t_points.extend(pxyreal)
    t_points = list(set(tuple(i) for i in t_points))
    t_points = np.array(t_points)
    tlen1 = len(t_points)
    if tlen1 ==tlen:
        break
    else:
        tlen = tlen1
    print(tlen)

plt.plot(Lupper[:, 0], Lupper[:, 1], 'ro')
#plt.show()


center = cor.mean(axis=0)
boundaryp = []
Lupper = np.array(Lupper)
for i in range(len(Lupper)):
    t = Lupper[i] - Lupper[i-1]  # tangent
    if abs(np.linalg.norm(t)) > 0.0001:
        t = t / np.linalg.norm(t)
        n = np.array([-t[1], t[0]]) # normal
        midpoint = Lupper[[i, i - 1]].mean(axis=0)
        thre_point = midpoint + np.sign(np.dot(midpoint - center, n)) * n * bdtr
        if not bbPath.contains_point(thre_point):
            boundaryp.append(thre_point)
            plt.plot(thre_point[0], thre_point[1], 'yo')

#plt.show()
boundaryp = np.array(boundaryp)
#寻找voronoi图边界点，组成list 将边界点连接，寻找是否有点在边界外，如果有，加进来
#找到基站中心，延长中心点到list中的点的线长0.5km
A=sorted(boundaryp,key=lambda x: x[0])
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
px = np.arange(min(boundaryp[:, 0]), max(boundaryp[:, 0]), 0.02)
py = np.arange(min(boundaryp[:, 1]), max(boundaryp[:, 1]), 0.02)
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