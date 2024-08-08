from itertools import combinations
import numpy as np


# 查找延长线
def FindTargetPoint(A,B):
    # 计算通过 A 和 B 的直线的斜率 (k) 和 y 截距 (b)
    k = (B[0] - A[0]) / (B[1] - A[1])
    b = A[0] - k * A[1]
    # 直线方程是 x = ky + b，我们需要找到与 y = 0 的交点
    # 在直线方程中设 y = 0，求 x 坐标的交点 P
    # P = np.array([x_p, 0])
    return b

def SlideWindow(z_s, window_size=0.1):
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


def Findzp(a_s):
    """
    找到延长线交点z的分布
    """
    # 寻找zp
    # 先分层
    a_s_layers=[[] for _ in range(5)]
    for i in range(len(a_s)):
        a_s_layers[int(a_s[i][3])].append(a_s[i])
    # 再组合
    zp_s = []
    ranges=combinations(range(5), 2)
    for i in ranges:
        layer1=a_s_layers[i[0]]
        layer2=a_s_layers[i[1]]
    
    
        for i in range(len(layer1)):
            for j in range(len(layer2)):
                # [x,y,z,r,theta]
                zp=FindTargetPoint(layer1[i][9:11],layer2[j][9:11])
                # print(zp)
                zp_s.append(zp)
                # DrawlnxZ(a_s[i][2:4],b_s[j][2:4],ax=ax)
    return zp_s



def calc_circle(p1, p2, p3):
    """
    计算通过三个点的圆的圆心和半径
    :param : p1, p2, p3 三个点
    :return: 圆心坐标 (h, k) 和半径 r
    """
    # 提取坐标
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # 构造方程组的系数矩阵
    A = np.array([
        [2*(x2 - x1), 2*(y2 - y1)],
        [2*(x3 - x1), 2*(y3 - y1)]
    ])
    B = np.array([
        [x2**2 - x1**2 + y2**2 - y1**2],
        [x3**2 - x1**2 + y3**2 - y1**2]
    ])
    
    # 解方程组
    h, k = np.linalg.solve(A, B)
    
    # 计算半径
    r = np.sqrt((x1 - h)**2 + (y1 - k)**2)
    
    return (h, k), r




