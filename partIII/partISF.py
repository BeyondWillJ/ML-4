import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
import numpy as np
# from functools import cmp_to_key
import math as m

# 功能函数定义

# 极坐标变换
def c_to_p(x, y):
    r = m.sqrt(x**2 + y**2)
    theta = m.atan2(y, x)  # 以弧度表示
    theta_degrees = m.degrees(theta)  # 转换为度数
    if theta_degrees < 0:
        theta_degrees += 360
    return r, theta_degrees
# 近似相等
def absequ(a,b,epsilon=2):
    if abs(a-b)<=epsilon:
        return True
    else: return False

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

# 查找延长线
def FindTargetPoint(A,B):
    # [z,r]
    # zp = [0]*len(pa)
    # zp[0] = pa[0] - (pb[0] - pa[0]) / (pb[1] - pa[1]) * pa[1] 
    # zp[1]=0
    # return zp[0]
    # 计算通过 A 和 B 的直线的斜率 (k) 和 y 截距 (b)
    k = (B[0] - A[0]) / (B[1] - A[1])
    b = A[0] - k * A[1]
    # 直线方程是 x = ky + b，我们需要找到与 y = 0 的交点
    # 在直线方程中设 y = 0，求 x 坐标的交点 P
    # P = np.array([x_p, 0])
    return b

# 绘制延长线
def DrawlnxZ(A,B,ax=plt):
    # 选择点 A 和 B，使它们在 y=0 的一侧
    A = np.array([-3, 4])
    B = np.array([-1, 2])

    # 计算通过 A 和 B 的直线的斜率 (k) 和 y 截距 (b)
    k = (B[0] - A[0]) / (B[1] - A[1])
    b = A[0] - k * A[1]

    # 直线方程是 x = ky + b，我们需要找到与 y = 0 的交点
    # 在直线方程中设 y = 0，求 x 坐标的交点 P
    x_p = b
    P = np.array([x_p, 0])

    # 计算 P 到 A 和 B 的距离
    dist_PA = np.linalg.norm(P - A)
    dist_PB = np.linalg.norm(P - B)
    # 确定较远的点
    farther_point = A if dist_PA > dist_PB else B
    # 绘制点、直线和线段
    ax.axhline(0, color='gray', linewidth=1)  # y=0 直线
    ax.plot([A[0], B[0]], [A[1], B[1]],)
    ax.scatter([A[0], B[0], P[0]], [A[1], B[1], P[1]], color=['blue', 'green', 'red'], zorder=5)
    ax.scatter(P[0], P[1], color='black', s=50, zorder=10)
    ax.plot([P[0], farther_point[0]], [P[1], farther_point[1]])

    return x_p

# 初始化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 画线
def drawln3d(p1,p2,ax=ax,color='red'):
    ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]], color=color)
def drawln2d(p1,p2,ax=ax,color='red'):
    ax.plot([p1[0],p2[0]], [p1[1],p2[1]], color=color)


def ReadOneEvent(f):
    """
    ReadOneEvent按顺序读取一个事件
    """
    eventt=f.readline().split()
    z00=float(eventt[-1])
    event=eventt[1]
    n=int(f.readline()[:-1])
    a_s, b_s = [], []
    for i in range(n):
        x0,y0,z0 = list(map(float, f.readline().split()))
        r0, theta0 = c_to_p(x0,y0)
        a_s.append([x0,y0,z0,r0,theta0])
    a_s.sort(key=lambda x: x[-1])
    n=int(f.readline()[:-1])
    for i in range(n):
        x0,y0,z0 = list(map(float, f.readline().split()))
        r0, theta0 = c_to_p(x0,y0)
        b_s.append([x0,y0,z0,r0,theta0])
    b_s.sort(key=lambda x: x[-1])

    return a_s,b_s,n,z00,event


def Draw3Dscatter(a_s,b_s,rs,ax=ax):
    # 使用scatter方法描绘点
    ps1=list(zip(*a_s))[:3]
    ps2=list(zip(*b_s))[:3]
    ax.scatter(*ps1,color='g')
    ax.scatter(*ps2,color='b')

    for t in range(len(rs)):
        drawln3d(rs[t][0][:3],rs[t][1][:3],ax=ax)

    # 设置坐标轴标签
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

def Findzp(n,a_s,b_s,ax):
    """
    找到延长线交点z的分布
    """
    # 寻找zp
    bmax, bmin = max(list(zip(*b_s))[2]), min(list(zip(*b_s))[2])
    mm=abs(bmax-bmin)
    bmin-=mm/2
    bmax+=mm/2
    zp_s = []
    for i in range(n):
        for j in range(n):
            # [x,y,z,r,theta]
            zp=FindTargetPoint(a_s[i][2:4],b_s[j][2:4])
            # print(zp)
            if zp<bmin or zp>bmax:
                continue
            else:
                zp_s.append(zp)
                # DrawlnxZ(a_s[i][2:4],b_s[j][2:4],ax=ax)
    return zp_s,bmin,bmax


def FindPeak(data,bins=150,threshold=2):
    """
    寻找直方图峰值
    """
    # 计算直方图
    hist, bin_edges = np.histogram(data, bins=bins)
    # 找到峰值
    peaks, _ = find_peaks(hist)
    if len(peaks) == 0:
        print("没有找到峰值")
    else:
        # 获取峰值处的坐标
        peak_index = peaks[np.argmax(hist[peaks])]  # 选择最大峰值
        peak_bin_center = (bin_edges[peak_index] + bin_edges[peak_index + 1]) / 2
        
        # 检查相邻的 bins
        # threshold = 0.1  # 阈值可以调整
        adjacent_bins = [peak_index]
        
        for i in range(peak_index - 1, -1, -1):
            if abs(hist[i] - hist[peak_index]) < threshold:
                adjacent_bins.append(i)
            else:
                break
        for i in range(peak_index + 1, len(hist)):
            if abs(hist[i] - hist[peak_index]) < threshold:
                adjacent_bins.append(i)
            else:
                break
            
        # 计算相邻 bins 的中点
        if adjacent_bins:
            bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in adjacent_bins]
            peak_center = np.mean(bin_centers)
        else:
            peak_center = peak_bin_center

        # 绘制直方图
        # plt.hist(data, bins=30, edgecolor='black')
        # plt.axvline(x=peak_center, color='red', linestyle='--', label=f'Peak at {peak_center:.2f}')
        # plt.xlabel('Value')
        # plt.ylabel('Frequency')
        # plt.legend()
        # plt.show()
        # print(f'峰值所在坐标: {peak_center:.2f}')
    return peak_center

def SlideWindow(z_s, window_size=5):
    """
    找到一维数据中最密集的区域及其质心 其中窗口大小基于数值距离
    z_s (list): 点的 x 坐标列表。
    window_size (float): 滑动窗口的数值大小。
    """
    sorted_z_s = sorted(z_s)
    
    max_density = 0
    start_index = 0
    end_index = 0
    
    # 初始化窗口
    left = 0
    right = 0
    while right < len(sorted_z_s) and sorted_z_s[right] - sorted_z_s[left] < window_size:
        right += 1
    
    if right == len(sorted_z_s):
        centroid = sum(sorted_z_s[left:right]) / (right - left)
        return (left, right - 1, centroid)
    
    max_density = right - left
    start_index = left
    end_index = right - 1
    
    # 移动窗口
    while right < len(sorted_z_s):
        right += 1
        while right < len(sorted_z_s) and sorted_z_s[right] - sorted_z_s[left] < window_size:
            right += 1
        current_density = right - left
        if current_density > max_density:
            max_density = current_density
            start_index = left
            end_index = right - 1
        left += 1
    
    # 计算质心
    centroid = sum(sorted_z_s[start_index:end_index+1]) / (end_index - start_index + 1)
    return centroid, start_index, end_index



# 在这里启动
def start(a_s,b_s,n,zp=None,theta_p0=3,epsilon_p0=5,bins=80,threshold=2):
    rs = []
    # 寻找估计的zp
    if zp == None:
        # 获取zp
        zp_s,bmin,bmax = Findzp(n,a_s,b_s,ax)
        # zp = FindPeak(zp_s,bins=bins,threshold=threshold)
        zp = SlideWindow(zp_s, 0.2)[0]

    SectorSearch(a_s,b_s,rs,zp,theta_p0,epsilon_p0)

    arbar=sum([i[0][2] for i in rs])/n
    brbar=sum([i[1][2] for i in rs])/n
    maxn=len(rs)
    # print(f"maxn={maxn}")
    ##结果分析
    ans=[]
    for i in range(maxn):
        ans.append(FindTargetPoint([rs[i][0][2],2],[rs[i][1][2],4]))
    
    return rs,maxn,ans,arbar,brbar

# 最终函数封装
def StartOne(tp,ep,tpp,epp):
    ans_deltaz=[]

    a_s,b_s,n,z00,event = ReadOneEvent(f)


    ####
    theta_p0=tp
    epsilon_p0=ep
    # 初始粗测
    _,maxn,ans,_,_ = start(a_s,b_s,n,theta_p0=theta_p0,epsilon_p0=epsilon_p0)
    # 收缩迭代
    # for _ in range(70):
    while True:
        theta_p0*=tpp
        epsilon_p0*=epp
        _,maxn,ans,_,_ = start(a_s,b_s,n,zp=np.mean(ans),theta_p0=theta_p0,epsilon_p0=epsilon_p0)
        # print(maxn)
        if maxn<15: break
    # 输出结果
    rs,maxn,ans,arbar,brbar = start(a_s,b_s,n,zp=np.mean(ans),theta_p0=theta_p0,epsilon_p0=epsilon_p0)
    # print(theta_p0,epsilon_p0)

    ###
    # Draw3Dscatter(a_s,b_s,rs)
    # for ii in range(n):
        # drawln3d(a_s[ii][:3],[0,0,z00],color='red')
        # drawln3d(b_s[ii][:3],[0,0,z00],color='red')
    ans_deltaz.append(z00-np.mean(ans))

    # print(ans_deltaz)


    return ans_deltaz,np.std(ans_deltaz)


