import random as r
import pandas as pd
import math as m
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import least_squares
from matplotlib.colors import to_rgba
from scipy.stats import linregress


# 极坐标变换
def c_to_p(x, y):
    r = m.sqrt(x**2 + y**2)
    theta = m.atan2(y, x)  # 以弧度表示
    theta_degrees = m.degrees(theta)  # 转换为度数
    if theta_degrees < 0:
        theta_degrees += 360
    return [r, theta_degrees]
# 极坐标变换
def c_to_p2(x, y):
    r = m.sqrt(x**2 + y**2)
    theta = m.atan2(y, x)  # 以弧度表示
    # theta_degrees = m.degrees(theta)  # 转换为度数
    if theta < 0:
        theta += 2*np.pi
    return [r, theta]
# 近似相等
def absequ(a,b,epsilon=2):
    if abs(a-b)<=epsilon:
        return True
    else: return False

# 任意极坐标变换
class PolarTransform:
    def __init__(self, origin, point):
        """
        初始化极坐标变换类
        :param origin: 极点O的坐标，元组形式 (x, y)
        :param point: 参考点P的坐标，元组形式 (x, y)
        """
        self.origin = origin
        self.ref_angle = self._calculate_reference_angle(point)

    def _calculate_reference_angle(self, point):
        """
        计算参考角度，即从极点到参考点的方向与正x轴之间的夹角
        :param point: 参考点P的坐标
        :return: 参考点P相对于极点O的角度，单位为弧度
        """
        dx = point[0] - self.origin[0]
        dy = point[1] - self.origin[1]
        angle = m.atan2(dy, dx)
        return angle

    def to_polar(self, point):
        """
        将直角坐标系中的点转换为极坐标系中的点
        :param point: 直角坐标系中的点，元组形式 (x, y)
        :return: 极坐标系中的点，元组形式 (r, theta) 其中r为半径 theta为角度
        单位为弧度，范围在 [0, 2*pi]
        """
        dx = point[0] - self.origin[0]
        dy = point[1] - self.origin[1]
        radius = m.sqrt(dx**2 + dy**2)
        angle = m.atan2(dy, dx) - self.ref_angle
        
        # 调整角度到 [0, 2*pi] 范围内
        if angle < 0:
            angle += 2 * m.pi
        elif angle >= 2 * m.pi:
            angle -= 2 * m.pi

        return [radius, angle]
        # 使用示例
        # origin = (0, 0)
        # point_p = (1, 1)
        # o_to_p = PolarTransform((0, 0), (1, 1))
        # point_q = (2, 2)
        # r,theta = o_to_p.to_polar(point_q)

# 直角坐标转换
class CoordinateTransformer:
    def __init__(self, origin, unit_vector_x, unit_vector_y):
        """
        初始化坐标转换器

        :param origin: 新坐标系的原点在旧坐标系中的坐标，形如 [x0, y0]
        :param unit_vector_x: 新坐标系x轴的单位向量在旧坐标系中的坐标，形如 [ux, uy]
        :param unit_vector_y: 新坐标系y轴的单位向量在旧坐标系中的坐标，形如 [vx, vy]
        """
        self.origin = np.array(origin)
        self.unit_vector_x = np.array(unit_vector_x)
        self.unit_vector_y = np.array(unit_vector_y)

    def new_to_old(self, new_coords):
        """
        将新坐标系下的点转换为旧坐标系下的点

        :param new_coords: 新坐标系下的点的坐标，形如 [x, y]
        :return: 旧坐标系下的点的坐标，形如 [X, Y]
        """
        new_coords = np.array(new_coords)
        old_coords = self.origin + new_coords[0] * self.unit_vector_x + new_coords[1] * self.unit_vector_y
        return old_coords

        # # 示例单位向量
        # new_x_unit = np.array([1,1])
        # new_y_unit = np.array([-1,1])
        # # 创建 CoordinateTransformer 实例
        # transformer = CoordinateTransformer(new_x_unit, new_y_unit)
        # # 测试转换函数
        # new_point = (1,1)  # 新坐标系中的点
        # old_point = transformer.convert_coords(*new_point)
        # print("Old coordinates:", old_point)

# 分段线性函数生成
class PiecewiseLinearFunction:

    def __init__(self, points):
        self.points = points
        self.slopes = []
        self.intercepts = []

        # 计算每段线的斜率和截距
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            self.slopes.append(slope)
            self.intercepts.append(intercept)

    def evaluate(self, x):
        # 查找x值所在的区间
        for i, point in enumerate(self.points):
            if x < point[0]:
                if i == 0:
                    # 如果x小于第一个点的x值，返回第一个点的y值
                    return point[1]
                # 使用对应的斜率和截距计算y值
                return self.slopes[i - 1] * x + self.intercepts[i - 1]
        # 如果x大于最后一个点的x值，返回最后一个点的y值
        return self.points[-1][1]
    
    #  示例
    # points = [(0, 0), (1, 1), (3, 4), (5, 6)]
    # pwlf = PiecewiseLinearFunction(points)
    # # 测试函数
    # print(pwlf.evaluate(0.5))  # 输出: 0.5

# 两个圆的交点坐标
def find_intersection_points(r0, P):
    """
    计算两个圆的交点坐标。
    r0 : float 第一个圆的半径。
    P : tuple 第二个圆的圆心坐标 (x, y)。
    返回: 包含两个交点坐标的列表，如果没有交点则返回空列表。
    """

    x1, y1 = P
    r2 = m.sqrt(x1**2 + y1**2)
    
    # 两圆心的距离
    d = m.sqrt(x1**2 + y1**2)
    
    if d > r0 + r2 or d < abs(r0 - r2) or (d == 0 and r0 == r2):
        # 如果两个圆没有交点或完全重合，则返回空列表
        return []

    # 两圆心连线的中点
    a = (r0**2 - r2**2 + d**2) / (2 * d)
    h = m.sqrt(r0**2 - a**2)
    
    # 中点坐标
    x2 = a * (x1 / d)
    y2 = a * (y1 / d)
    
    # 交点
    x3 = x2 + h * (y1 / d)
    y3 = y2 - h * (x1 / d)
    
    x4 = x2 - h * (y1 / d)
    y4 = y2 + h * (x1 / d)


    # 标识顺逆[顺 逆]
    cross_to_p=PolarTransform(P, (0, 0))
    if cross_to_p.to_polar((x3,y3))[1] < cross_to_p.to_polar((x4,y4))[1]:
        return [(x3,y3),(x4,y4)]
    else:
        return [(x4,y4),(x3,y3)]

    # # 返回结果
    # return [(x3, y3), (x4, y4)]

def Findln(A, B):
    x1, y1 = A
    x2, y2 = B
    # print(A,B)
    # 计算斜率 k
    # if y2 == y1:
    #     raise ValueError("两个点的 y 坐标相同，无法计算斜率。")
    k = (x2 - x1) / (y2 - y1)
    # 计算截距 b
    b = x1 - k * y1
    return k, b

# 生成随机颜色的函数
def random_color():
    r = np.random.rand()
    g = np.random.rand()
    b = np.random.rand()
    return to_rgba([r, g, b])

# 画线
def drawln3d(p1,p2,ax=plt,color='red'):
    ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]], color=color)
def drawln2d(p1,p2,ax=plt,color='red'):
    ax.plot([p1[0],p2[0]], [p1[1],p2[1]], color=color)

def Draw3Dscatter(a_s,ax=plt,color='g'):
    # 使用scatter方法描绘点
    ps1=list(zip(*a_s))
    ax.scatter(*ps1,color=color)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

# r-z平面绘制
def Drawrz(a_s,axisr,axisz,ax=plt):
    """
    绘制r-z平面
    a_s,axisr,axisz r,z的列号
    """
    ax.figure()
    # plt.title(fr'r-z for Event {event}''\n'fr'n = {maxn}, $\bar{{z}}$ = {np.mean(ans)}, $z_{{\text{{real}}}}$ = {z00}')
    ax.xlabel('z')
    ax.ylabel('r')
    # 使用plot函数绘制曲线 r-z的点集
    r1=[a_s[i][axisr] for i in range(100)]
    z1=[a_s[i][axisz] for i in range(100)]
    # r2=[b_s[i][-2] for i in range(n)]
    # z2=[b_s[i][2] for i in range(n)]
    ax.plot(z1, r1, 'bo')

# 点的聚类
def categorize_points(points, threshold):
    while True:
        categories = []
        current_category = [points[0]]  # 初始化第一个点为当前类别

        for i in range(1, len(points)):
            if points[i][-1] - points[i-1][-1] <= threshold:
                current_category.append(points[i])
            else:
                categories.append(current_category)
                current_category = [points[i]]

        categories.append(current_category)  # 添加最后一个类别
        categories2=[]
        for i in categories:
            i.append([0]*len(points[0])) # 添加起始点
            # i.sort(key=lambda x: x[2])
            if len(i) == 6: categories2.append(sorted(i,key=lambda x: x[7]))
            else:
                if len(i)>6: threshold += 0.0005
                else: threshold *=0.98
                continue
            # print('##########ctt',categories2[-1])
        break  
    return categories2


# 圆拟合
def FitCircle(arr,delta_r,draw=True,ax=plt):
    """
    delta_r=每层半径 [r0,r1,r2,r3,r4]
    return a, b, r, e0=[[n,e],...]
    """
    # 给定的五组点
    points = np.array(arr)

    def residuals(params, points):
        a, b, r = params
        return np.sqrt((points[:, 0] - a)**2 + (points[:, 1] - b)**2) - r

    # 初始猜测值
    x_mean, y_mean = np.mean(points[:, 0]), np.mean(points[:, 1])
    initial_guess = [x_mean, y_mean, np.mean(np.sqrt((points[:, 0] - x_mean)**2 + (points[:, 1] - y_mean)**2))]

    result = least_squares(residuals, initial_guess, args=(points,))
    a, b, r = result.x

    
    e0=[]

    points=arr

    # 点的排序
    c_polar=PolarTransform((a,b),(0,0))
    for i in range(len(points)):
        points[i].append((c_polar.to_polar((points[i][0],points[i][1])))[-1])
    points.sort(key=lambda x: x[-1])
    points.pop(0)
    points_mean=np.mean(points,axis=0)[-1]
    
    # 判断顺逆时针 层数必须由小到大
    # ap bp 数据给出点 | axx bxx 交点
    if points_mean<np.pi:#顺
        for i in range(5): #5个层layer
            ap,bp=points[i][:2]
            findx=find_intersection_points(delta_r[i], (a,b))[0]#0顺
            axx,bxx=findx
            # e0.append([i,np.sqrt((ap-axx)**2+(bp-bxx)**2)])
            e0.append(np.sqrt((ap-axx)**2+(bp-bxx)**2))
    else:
        for i in range(5): #5个层layer
            ap,bp=points[-i-1][:2]
            findx=find_intersection_points(delta_r[i], (a,b))[1]#
            axx,bxx=findx
            # e0.append([i,np.sqrt((ap-axx)**2+(bp-bxx)**2)])
            e0.append(np.sqrt((ap-axx)**2+(bp-bxx)**2))

    if draw:
        # 绘制结果
        # ax.figure()
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = a + r * np.cos(theta)
        y_circle = b + r * np.sin(theta)

        # plt.figure(figsize=(8, 8))
        ax.plot(x_circle, y_circle)
        ax.axis('equal')
        # plt.scatter(points[:, 0], points[:, 1], color='red', label='Data Points')
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.legend()
    return a, b, r, e0


# 圆拟合不带残差
def FitCircle_simp(arr,draw=True,ax=plt):
    """
    return (a, b), r
    """
    # 给定的n组点
    points = np.array(arr)

    def residuals(params, points):
        a, b, r = params
        return np.sqrt((points[:, 0] - a)**2 + (points[:, 1] - b)**2) - r

    # 初始猜测值
    x_mean, y_mean = np.mean(points[:, 0]), np.mean(points[:, 1])
    initial_guess = [x_mean, y_mean, np.mean(np.sqrt((points[:, 0] - x_mean)**2 + (points[:, 1] - y_mean)**2))]

    result = least_squares(residuals, initial_guess, args=(points,))
    a, b, r = result.x

    if draw:
        # 绘制结果
        # ax.figure()
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = a + r * np.cos(theta)
        y_circle = b + r * np.sin(theta)

        # plt.figure(figsize=(8, 8))
        ax.plot(x_circle, y_circle)
        ax.axis('equal')
        # plt.scatter(points[:, 0], points[:, 1], color='red', label='Data Points')
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.legend()
    return (a, b), r




# 画平面有限圆
def DrawOneCircle(theta, r,O):
    x_o,y_o=O
    unit_vector_x = np.array([-x_o,-y_o])/r
    unit_vector_y = np.array([y_o,-x_o])/r

    # print(unit_vector_x,unit_vector_y)

    transformer = CoordinateTransformer(O, unit_vector_x, unit_vector_y)
    max_theta=len(theta)
    x=np.cos(theta)*r
    y=np.sin(theta)*r
    x0,y0=[],[]
    for i in range(max_theta):
        x0_,y0_=transformer.new_to_old([x[i],y[i]])
        x0.append(x0_)
        y0.append(y0_)
    
    return [x0,y0]
    # plt.plot(x0,y0,color=color)


# 绘制等距螺旋线
def DrawEllipse(center0, cc, r, ax=plt, color='purple'):
    # [0x,1y,2z,3r,4theta,5k,6theta_o]
    # print(len(cc[0]))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    # 参数设置
    maxn=200
    # center = np.array(center0)  # 圆心
    cx,cy=center0
    # alpha=c_to_p2(cx,cy)[1]

    # 极坐标变换
    to_p = PolarTransform((cx,cy), (0,0))

    # r = cc[3]
    # radius = np.sqrt(center[0]**2 + center[1]**2)  # 半径
    # start_point = np.array([0, 0, 0])  # 起点
    # k = 0.5  # z 与投影经过的圆弧之比

    # 生成数据
    # theta = np.linspace(0, 1 * np.pi,maxn)  # 角度2*pi*r ~ 360
    # 2pir/360*cc[4]
    # rthetar=[]
    # rthetad=[]
    for i in range(len(cc)):
        cc[i].append(to_p.to_polar((cc[i][4],cc[i][5]))[-1])#R
        # rthetar[i][-1]-=alpha
        # rthetad.append(c_to_p(cc[i][0]-cx,cc[i][1]-cy)-alpha)#Degree
    # r=np.mean(rthetar,axis=0)[0]

    cc.sort(key=lambda x:x[10])
    # rthetar=list(zip(*cc))[-1]
    # print(cc)
    # rthetar.sort(key=lambda x:x[0])
    # rthetad.sort(key=lambda x:x[0])
    
    # z=np.array([func.evaluate(theta[i]*r) for i in range(len(theta))])
    if 0<=cc[1][10]<=np.pi:
        theta = np.linspace(0, cc[-1][-1], maxn)#Radian
    else:
        cc[0]=[0]*4+[0,0,0,0,0,0,np.pi*2]
        cc.sort(key=lambda x:x[10])
        theta = np.linspace(cc[0][-1],np.pi*2, maxn)
    # phi = (alpha + np.pi + theta) % (2*np.pi)

    # to_z=PiecewiseLinearFunction([[rthetar[i][-1]*r,cc[i][2]] for i in range(len(rthetar))])
    # z=np.array([to_z.evaluate(theta[i]*r) for i in range(len(theta))])
    # xl=np.array([rthetar[i][-1] for i in range(len(rthetar))])*r
    xl=np.array(list(zip(*cc))[-1])*r
    yl=np.array(list(zip(*cc))[6])
    # print(r)
    k, b, *_ = linregress(xl, yl)

    # k=np.mean([cc[i][-2] for i in range(len(cc))])
    z=k*theta*r+b
    # print(z)
    # phi = theta + (c_to_p2(cx,cy)[1]+np.pi) % (2*np.pi)
    
    x,y=DrawOneCircle(theta, r, center0)

    # print(len(x),len(y),len(z))
    # 调整起点
    # x += start_point[0] - x[0]
    # y += start_point[1] - y[0]
    # z += start_point[2] - z[0]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # 绘制螺旋线
    ax.plot(x, y, z, color=color)
    # ax.legend()

