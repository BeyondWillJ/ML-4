from SFlib_I import *
from SFlib_II import *
from itertools import combinations,product
import copy

data=r'data\SiHits_3D_pvar_zvar_0.03_0.50_28_0.04_0.06_2000_v1.txt'
# event0=915 #9989

# 初始化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# 读取文件
# [0event number, 1track number, track 2px, 3py, 4pz, hit 5x, 6y,7z]
df = pd.read_csv(data, header=None)
# 显示前几行数据以确认读取正确
# print(df.head(50))

#     0     1  2  3       4       5       6       7       8       9
# 0   0 -1.53  0  0 -0.0360 -0.0759 -0.0061 -0.3728 -0.9229 -1.5978
# 1   0 -1.53  0  1 -0.0360 -0.0759 -0.0061 -0.6580 -1.8834 -1.6705
# 2   0 -1.53  0  2 -0.0360 -0.0759 -0.0061 -0.8436 -2.8738 -1.7432
# 3   0 -1.53  0  2 -0.0360 -0.0759 -0.0061 -2.4877 -1.6766 -4.1410


# =============================================
# =============================================
# =============================================
# 在这里启动
event0=1
eventi=df[df.iloc[:, 0] == event0]
a_s=[]
for i in range(len(eventi)):
    a_s_p=eventi.iloc[i, :].tolist()
    a_s.append(a_s_p+c_to_p(a_s_p[7],a_s_p[8]))
    # print(a_s[-1])

realz0=a_s[0][1]

# ===================================
# 顶点寻找
zp_s = Findzp(a_s)
zp = SlideWindow(zp_s, 0.001)[0]
print('zp',zp)
# ===================================

# 绘制r-z
# plt.figure()
# # plt.title(fr'r-z for Event {event}''\n'fr'n = {mpltn}, $\bar{{z}}$ = {np.mean(ans)}, $z_{{\text{{real}}}}$ = {z00}')
# plt.xlabel('z')
# plt.ylabel('r')
# plt.title(fr'r-z for Event {event0}')
# # 使用plot函数绘制曲线 r-z的点集
# r1=[a_s[i][10] for i in range(len(a_s))]
# z1=[a_s[i][9] for i in range(len(a_s))]
# plt.plot(z1, r1, 'bo')

rz_theta_s=[]
for i in range(len(a_s)):
    # rz_theta_s.append(c_to_p2(a_s[i][9]-realz0,a_s[i][10]))
    a_s[i].append(c_to_p2(a_s[i][9]-realz0,a_s[i][10])[1])

    # 绘制r-z连线
    # drawln2d([realz0,0],a_s[i][9:11],ax=plt)

# plt.figure()
# plt.title(r"$\theta - r$ | (r-z) mapping")
# plt.scatter([rz_theta_s[i][1] for i in range(len(rz_theta_s))],[rz_theta_s[i][0] for i in range(len(rz_theta_s))])
# rz_theta_s_t=[m.log(rz_theta_s[i][1]+20,2)*50 for i in range(len(rz_theta_s))]
rz_theta_s_t=[a_s[i][12]*100 for i in range(len(a_s))]
# plt.scatter(rz_theta_s_t,[a_s[i][12] for i in range(len(a_s))])
rz_theta_s_delta=[]



a_s.sort(key=lambda x:x[12])

# 求差分数组
for i in range(len(a_s)-1):
    rz_theta_s_delta.append(a_s[i+1][12]-a_s[i][12])

# 绘制数组图像~i
# plt.figure()
# plt.title(r"$\Delta (\theta - r)$ | (r-z) mapping")
# plt.scatter(range(len(rz_theta_s_delta)),rz_theta_s_delta)
# plt.axhline(y=0, color='black', linewidth=1)  # 水平线作为X轴
# # 在X轴上每隔1个单位绘制一条垂直的参考线
# for x in range(0, len(rz_theta_s_delta)+1):
#     plt.axvline(x=x, color='gray', linestyle='-', linewidth=0.5)
# # 添加网格
# plt.grid(axis='x')  # 只显示X轴上的网格


# 点的选择性聚类
flag=0
ct_s=[]
tempct=[]
for i in range(len(rz_theta_s_delta)):
    if rz_theta_s_delta[i]<0.01:
        flag+=1
        tempct.append(a_s[i])
        # print(rz_theta_s_t[i])
    else:
        if flag!=0:
        # if flag!=0 and flag>3:
            flag=0
            tempct.append(a_s[i])
            if len(tempct)>3: ct_s.append(tempct)
            tempct=[]
        else: pass


# 绘制选点
# plt.figure()
# plt.title(r"$\theta$ | (r-z) mapping")
# plt.scatter(rz_theta_s_t,[0 for i in range(len(a_s))],color='lightgrey')
# for j in range(len(ct_s)):
#     color=random_color()
#     plt.scatter([ct_s[j][i][12]*100 for i in range(len(ct_s[j]))],[0 for i in range(len(ct_s[j]))],color=color,s=10)
# plt.savefig('figure_partIII.png',bbox_inches="tight", dpi=800)

# 最小二乘遍历
ct_II=[]
sum_residuals=[]
c_s=[[] for _ in range(len(ct_s))]
c_s_points=[]


###填写一个具体的ci
# ci=3


# 对每一个I类
for i in range(len(ct_s)):
# for i in [ci]:
    circles=[]
    ct_II_i=[[] for i in range(5)]
    # if len(ct_s[i])>8: continue
    for ii in range(len(ct_s[i])):
        ct_II_i[int(ct_s[i][ii][3])].append(copy.deepcopy(ct_s[i][ii]))
    ct_II_i_len=[len(ct_II_i[i]) for i in range(5)]
    # 加入顶点
    ver=copy.deepcopy(ct_s[i][-1])
    ver[7:13]=[0,0,realz0,0,0,0]

    l12=list(product(*[range(n) for n in ct_II_i_len]))
    print(len(l12))
    for j in range(len(l12)):
        ct_temp=[ct_II_i[k][l12[j][k]] for k in range(len(l12[j]))]
        # print(ct_temp)
        (xc,yc),rc = FitCircle_simp([ct_temp[i][7:9] for i in range(len(ct_temp))],False,ax=plt)
        # print(xc,yc,rc)
        sum_residual=0
        for k in range(len(ct_temp)):
            sum_residual+=abs(distance(ct_temp[k][7:9],(xc,yc))-rc)
        print(sum_residual)
        # circles.append([xc,yc,rc])
        if sum_residual<=0.1:
            pltf=PolarTransform((xc,yc),(0,0))
            a_or_b=[0,0]
            for k in range(len(ct_temp)):
                r_t,theta_t = pltf.to_polar(ct_temp[k][7:9])
                if 0<=theta_t<=np.pi: a_or_b[0]+=1
                else: a_or_b[1]+=1
            # 满足只占一侧的条件
            if a_or_b[0]==0 or a_or_b[1]==0:
                circles.append([xc,yc,rc])
                ct_II.append([[copy.deepcopy(ver)]+copy.deepcopy(ct_temp),[xc,yc,rc]])
    ct_s[i].append(copy.deepcopy(ver))


# 绘制3d
Draw3Dscatter([a_s[i][7:10] for i in range(len(a_s))],ax=ax,color='b')
ax.scatter(0,0,realz0, c='r', marker='o')
fig.subplots_adjust(top=0.85)
ax.set_title(f"event {event0}\ndata: {data}\ntracks: {len(ct_s)} | $z_\\mathrm{{real}}$ = {realz0} $z_\\mathrm{{pred}}$ = {zp:.5f}")
# for i in range(len(ct_II)):#画线
#     cl = random_color()
#     ct_II[i].sort(key=lambda x:x[9])
#     for j in range(len(ct_II[i])-1):
#         drawln3d(ct_II[i][j][7:10], ct_II[i][j+1][7:10], ax=ax, color=cl)

# 拟合
for i in range(len(ct_II)):#拟合
    color=random_color()
    xx,yy,rr=ct_II[i][1]
    cc=copy.deepcopy(ct_II[i][0])
    DrawEllipse([xx,yy], cc, rr, ax=ax, color=color)

plt.show()




