import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from functools import cmp_to_key


# 创建一个新的图
fig = plt.figure()

# 创建一个3D坐标轴
ax = fig.add_subplot(111, projection='3d')

x = []
y = []
z = []

with open("SiHits_cylinder_10000_15R50.txt",'r') as f:
# with open("test2.txt",'r') as f:
    # f.readline()
    # flag=0
    for i in range(1):
        # f.readline()
        # flag+=1
        # if flag%2 == 1:
        z00=float(f.readline().split()[-1])
        # x.append(0.0)
        # y.append(0.0)
        # z.append(z00)
        f.readline()
        xt, yt, zt, tt1, tt2 = [],[],[],[],[]
        for i in range(20):
            # print(f.readline().split())
            # print()
            x0,y0,z0 = list(map(float, f.readline().split()))
            xt.append(x0)
            yt.append(y0)
            zt.append(z0)
            tt1.append([x0,y0,z0])
        tt1.sort(key=lambda x: x[-1])
        f.readline()
        for i in range(20):
            # print(f.readline().split())
            # print()
            x0,y0,z0 = list(map(float, f.readline().split()))
            xt.append(x0)
            yt.append(y0)
            zt.append(z0)
            tt2.append([x0,y0,z0])
        tt2.sort(key=lambda x: x[-1])




# 定义数据点
# print(len(z))
 
# 使用scatter方法描绘点
# print(tt1)
# print(list(zip(*tt1)))
ps1=list(zip(*tt1))
ps2=list(zip(*tt2))
ax.scatter(*ps1,color='g')
ax.scatter(*ps2,color='b')
def drawln(n, ax=ax):#画线
    ax.plot(*zip([x[0],y[0],z[0]],[x[n],y[n],z[n]]),color='red')
def drawln3d(p1,p2,ax=ax):
    ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]], color='red')
def drawln2d(p1,p2,ax=ax):
    ax.plot([p1[0],p2[0]], [p1[1],p2[1]], color='red')
for i in range(20):
    drawln3d(tt1[i],tt2[i])
# 设置坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

#######
# r-z平面绘制
plt.figure()
# 使用plot函数绘制曲线
r1=[(tt1[i][0]**2+tt1[i][1]**2)**0.5 for i in range(20)]
r2=[(tt2[i][0]**2+tt2[i][1]**2)**0.5 for i in range(20)]
plt.plot(ps1[-1], r1, 'bo', ms=1)
plt.plot(ps2[-1], r2, 'go', ms=1)
for i in range(20):
    drawln2d([ps1[-1][i],r1[i]],[ps2[-1][i],r2[i]],ax=plt)
# 设置标题和轴标签
plt.title('r-z')
plt.xlabel('z')
plt.ylabel('r')
 
# 显示图形
plt.show()