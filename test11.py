import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# from functools import cmp_to_key
import math as m

# 极坐标变换
def c_to_p(x, y):
    r = m.sqrt(x**2 + y**2)
    theta = m.atan2(y, x)  # 以弧度表示
    theta_degrees = m.degrees(theta)  # 转换为度数
    if theta_degrees < 0:
        theta_degrees += 360
    return r, theta_degrees

def absequ(a,b,epsilon=2):
    if abs(a-b)<=epsilon:
        return True
    else: return False
rs = []
# 查找有效对
def SectorSearch(a_s,b_s,rs,zp,theta_p,epsilon_p):
    # flagp=0
    for i in range(len(a_s)):#a_s已按theta排好序
        thetax=a_s[i][-1]
        bij=[]
        for j in range(len(b_s)):
            delta_theta=b_s[j][-1] - thetax
            if delta_theta < 0: delta_theta+=360
            if delta_theta <= theta_p or 360-theta_p <= delta_theta:
            # if 0 <= b_s[j][-1]-(thetax-theta_p) <= 2*theta_p:
                # print(thetax,b_s[j][-1]-thetax)
                bij.append(b_s[j])
        zai=a_s[i][2]
        for k in range(len(bij)):
            zbi=bij[k][2]
            if absequ(2*zai, zp+zbi, epsilon=epsilon_p):##可训练参数
                rs.append([a_s[i], bij[k]])
                # print(rs)
            
def FindTargetPoint(pa,pb):
    # [z,r]
    zp = [0]*len(pa)
    zp[0] = pa[0] - (pb[0] - pa[0]) / (pb[1] - pa[1]) * pa[1] 
    zp[1]=0
    return zp[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

with open("SiHits_cylinder_10000_20_R50_v2.txt",'r') as f:
    for _ in range(1):
    # with open("test2.txt",'r') as f:
        # f.readline()
        # flag=0
        for i in range(1):
            # f.readline()
            # flag+=1
            # if flag%2 == 1:
            eventt=f.readline().split()
            z00=float(eventt[-1])
            event=eventt[1]
            # x.append(0.0)
            # y.append(0.0)
            # z.append(z00)
            n=int(f.readline()[:-1])
            a_s, b_s = [], []
            for i in range(n):
                # print(f.readline().split())
                # print()
                x0,y0,z0 = list(map(float, f.readline().split()))
                r0, theta0 = c_to_p(x0,y0)
                a_s.append([x0,y0,z0,r0,theta0])
            a_s.sort(key=lambda x: x[-1])
            n=int(f.readline()[:-1])
            for i in range(n):
                # print(f.readline().split())
                # print()
                x0,y0,z0 = list(map(float, f.readline().split()))
                r0, theta0 = c_to_p(x0,y0)
                b_s.append([x0,y0,z0,r0,theta0])
            b_s.sort(key=lambda x: x[-1])


        # 定义数据点
        # print(len(z))

        # 使用scatter方法描绘点
        # print(tt1)
        # print(list(zip(*tt1)))
        ps1=list(zip(*a_s))[:3]
        ps2=list(zip(*b_s))[:3]
        ax.scatter(*ps1,color='g')
        ax.scatter(*ps2,color='b')
        # def drawln(n, ax=ax):#画线
        # ax.plot(*zip([x[0],y[0],z[0]],[x[n],y[n],z[n]]),color='red')
        def drawln3d(p1,p2,ax=ax):
            ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]], color='red')
        def drawln2d(p1,p2,ax=ax):
            ax.plot([p1[0],p2[0]], [p1[1],p2[1]], color='red')

        # 设置坐标轴标签
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ###在这里启动
        abar=sum([i[2] for i in a_s])/n
        bbar=sum([i[2] for i in b_s])/n
        zp = FindTargetPoint([abar,2],[bbar,4])
        # print(abar,bbar,zp)
        SectorSearch(a_s,b_s,rs,zp,theta_p=2.5,epsilon_p=5)
        for t in range(len(rs)):
            drawln3d(rs[t][0][:3],rs[t][1][:3],ax=ax)
        arbar=sum([i[0][2] for i in rs])/n
        brbar=sum([i[1][2] for i in rs])/n
        maxn=len(rs)
        print(f"maxn={maxn}")
        ##结果分析
        ans=[]
        for i in range(maxn):
            ans.append(FindTargetPoint([rs[i][0][2],2],[rs[i][1][2],4]))



plt.figure()
plt.hist(ans)
plt.xlabel("the z coordinate of the vertex")
# plt.title("")
# print(zp)
# print(rs)
# print(ans)

# plt.show()
#######
# r-z平面绘制
plt.figure()
# 使用plot函数绘制曲线
r1=[rs[i][0][-2] for i in range(maxn)]
z1=[rs[i][0][2] for i in range(maxn)]
r2=[rs[i][1][-2] for i in range(maxn)]
z2=[rs[i][1][2] for i in range(maxn)]
plt.plot(z1, r1, 'bo')
plt.plot(z2, r2, 'go')
for i in range(maxn):
    print(rs[i][0][2:4],rs[i][1][2:4])
    drawln2d(rs[i][0][2:4],rs[i][1][2:4],ax=plt)
# 设置标题和轴标签
plt.title(rf'r-z for Event {event}\nn = {maxn}, 'r'$\bar{{z}}$'+rf' = {np.mean(ans)}, 'r'${{z_{{\text{{real}}}}}}$'+rf' = {z00}')
plt.title(f'r-z for Event {event}\nn = {maxn}, '+r'$\bar{z}$'+f' = {np.mean(ans)}, '+r'$z_{\text{real}}$'+f' = {z00}')
plt.xlabel('z')
plt.ylabel('r')
 
# 显示图形
plt.show()

