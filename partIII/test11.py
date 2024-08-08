from partIISFlib import *
from partISFlib import *

data='SiHits_3D_pvar_zvar_0.03_0.50_28_0.04_0.06_2000_v1.txt'
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
event0=21
eventi=df[df.iloc[:, 0] == event0]
a_s=[]
for i in range(len(eventi)):
    a_s_p=eventi.iloc[i, :].tolist()
    a_s.append(a_s_p+c_to_p(a_s_p[7],a_s_p[8]))
    print(a_s[-1])

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
    if rz_theta_s_delta[i]<0.003:
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


# 去除离群点


# 绘制3d
Draw3Dscatter([a_s[i][7:10] for i in range(len(a_s))],ax=ax,color='b')
ax.scatter(0,0,realz0, c='r', marker='o')
fig.subplots_adjust(top=0.85)
ax.set_title(f"event {event0}\ndata: {data}\ntracks: {len(ct_s)} | $z_\\text{{real}}$ = {realz0} $z_\\text{{pred}}$ = {zp:.5f}")
for i in range(len(ct_s)):#画线
    cl = random_color()
    ct_s[i].sort(key=lambda x:x[9])
    for j in range(len(ct_s[i])-1):
        drawln3d(ct_s[i][j][7:10], ct_s[i][j+1][7:10], ax=ax, color=cl)

c_s=[]
# 三点共圆遍历
for i in range(len(ct_s)):
    c_s.append([])
    l12=list(combinations(range(len(ct_s[i])),3))
    for j in range(len(l12)):
        x1,x2,x3=ct_s[i][l12[j][0]][7:9],ct_s[i][l12[j][1]][7:9],ct_s[i][l12[j][2]][7:9]
        (xc,yc),rc = calc_circle(x1,x2,x3)
        c_s[i].append([xc,yc])


plt.figure()
# 最小二乘遍历
for i in range(len(ct_s)):
    c_s.append([])
    l12=list(combinations(range(len(ct_s[i])),4))
    for j in range(len(l12)):
        x1,x2,x3,x4=ct_s[i][l12[j][0]][7:9],ct_s[i][l12[j][1]][7:9],ct_s[i][l12[j][2]][7:9],ct_s[i][l12[j][3]][7:9]
        (xc,yc),rc = FitCircle_simp([x1,x2,x3,x4],True,ax=plt)
        c_s[i].append([xc,yc])

ci=2
plt.title(f"centre of circle {ci}")
plt.scatter([ct_s[ci][i][7] for i in range(len(ct_s[ci]))],[ct_s[ci][i][8] for i in range(len(ct_s[ci]))],color='b')
plt.scatter([c_s[ci][i][0] for i in range(len(c_s[ci]))],[c_s[ci][i][1] for i in range(len(c_s[ci]))],color='r')




plt.show()









#     0  1  2       3       4       5       6       7       8
# 0   0  0  0  0.0495  0.0522 -0.0526  0.7370  0.6745 -0.7309
# 1   0  0  1  0.0495  0.0522 -0.0526  1.5798  1.2243 -1.4622

# event0=346
# print(f"event {event0}")

# # 单个event
# # i0=0+event0*100
# totalevent=7000
# for i0 in range(9800,10000):
#     nowevent=i0
#     # i0=0+event0*100
#     if not i0%50:
#         print(f"event {i0}")
#     a_s=[]
#     for i in range(20):
#         for j in range(5):
#             a_s_p=df.iloc[i0, :].tolist()
#             a_s.append([a_s_p[1]]+a_s_p[3:]+c_to_p(a_s_p[6],a_s_p[7]))
#             # a_ss.append(a_s_p)
#             i0+=1



#     # print(a_s)

#     ###############
#     # 总分析绘图

#         # plt.plot(z2, r2, 'go')
#         # r-z线段
#         # for i in range(len(rs)):
#         #     # print(rs[i][0][2:4],rs[i][1][2:4])
#         #     drawln2d(rs[i][0][2:4],rs[i][1][2:4],ax=plt)
#             # drawln2d([z00,0],b_s[i][2:4],ax=plt)
#             # drawln2d([-1.53792346,0],b_s[i][2:4],ax=plt)
#         #####

#     # 绘制rz
#     # Drawrz(a_s)

#     k_s=[]
#     for i in range(len(a_s)):
#         k,b = Findln([0,0],a_s[i][6:8])
#         a_s[i].append(k)
#         k_s.append(k)
#         # print(k,b)

#     a_s.sort(key=lambda x: x[-1])

#     cp=categorize_points(a_s,0.001)
#     # print(cp)

#     # 绘制3d
#     # Draw3Dscatter([a_s[i][4:7] for i in range(len(a_s))],ax=ax,color='b')
#     # ax.scatter(0,0,0, c='r', marker='o')
#     # ax.set_title(f"event {event0}\ndata: {data}")

#     # for i in range(len(cp)):#画线
#     #     cl = random_color()
#     #     if len(cp[i])>5: print(i)
#     #     for j in range(len(cp[i])-1):
#     #         drawln3d(cp[i][j][:3], cp[i][j+1][:3], ax=ax, color=cl)

#     # plt.show()

#     # 绘制斜率分布
#     # plt.figure()
#     # plt.scatter(k_s,[0]*len(k_s))

#     # 绘制xOy投影
#     # plt.figure()
#     # plt.scatter([a_s[i][4] for i in range(len(a_s))],[a_s[i][5] for i in range(len(a_s))])
#     # # for i in range(len(cp)):#画线
#     # #     cl = random_color()
#     # #     if len(cp[i])>5: print(i)
#     # #     for j in range(len(cp[i])-1):
#     # #         drawln2d(cp[i][j][:2], cp[i][j+1][:2], ax=plt, color=cl)
#     # plt.axis('equal') # 设置等比例

#     # plt.figure()
#     # print('cp',cp)



#     for j in range(len(cp)):#拟合
#         color=random_color()
#         # print(cp[j][1])
#         # print([cp[j][i][4:6] for i in range(len(cp[j]))])
#         ccx,ccy,r0,e0 = FitCircle([cp[j][i][4:6] for i in range(len(cp[j]))],delta_r=[1,2,3,5,7],draw=False,ax=plt)
#         r0_s.append([ccx,ccy])
#         cc_s.append(np.sqrt(cp[j][2][1]**2+cp[j][2][2]**2)/r0)
#         pt_s.append(np.sqrt(cp[j][2][1]**2+cp[j][2][2]**2))
#         cc_s_p.append([np.sqrt(cp[j][2][1]**2+cp[j][2][2]**2),cp[j][2][1],cp[j][2][2]])
#         cc_s_r.append(r0)
#         t.append([r0]+e0)
#         p_T.append([cp[j][1][1],cp[j][1][2]])
#         # DrawEllipse([ccx,ccy], cp[j], r0, ax=ax, color=color)

# print(f"event {nowevent} | finished")


# # plt.figure()
# # plt.hist(cc_s,bins=550)
# # print(cc_s_r)
# # print(cc_s_p)
# # plt.title(r"Using $p_{xOy,\text{real}}/r_\text{predicted}$")
# # print(cc_s)
# # plt.show()
# # t



# print(p_T)

# with open('p.txt', 'w') as f:
#     for i in range(len(p_T)):
#         f.write(f"{p_T[i]}\n")