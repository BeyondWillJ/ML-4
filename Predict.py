from SFlib_II import *

data=r'data\SiHits_3D_pvar_0.02_10000_v2.txt'
# event0=915 #9989

# 初始化
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


# 读取文件
# [0event number, 1track number, track 2px, 3py, 4pz, hit 5x, 6y,7z]
df = pd.read_csv(data, header=None)
# 显示前几行数据以确认读取正确
# print(df.head(50))




cc_s=[]
cc_s_p=[]
cc_s_r=[]

# 装载 r p_T 预备训练
# r特征为 t=[[r0,e0],...]
r0_s=[]
t=[]
p_T=[]
pt_s=[]
checkout=[]

#     0  1  2       3       4       5       6       7       8
# 0   0  0  0  0.0495  0.0522 -0.0526  0.7370  0.6745 -0.7309
# 1   0  0  1  0.0495  0.0522 -0.0526  1.5798  1.2243 -1.4622

# event0=346
# print(f"event {event0}")

# 单个event
# i0=0+event0*100
totalevent=10000
for i0 in range(totalevent):
    nowevent=i0
    # i0=0+event0*100
    if not i0%50:
        print(f"event {i0}")
    a_s=[]
    for i in range(20):
        for j in range(5):
            a_s_p=df.iloc[i0, :].tolist()
            # a_s.append([a_s_p[1]]+a_s_p[3:]+c_to_p(a_s_p[6],a_s_p[7]))##这里改动
            a_s.append([a_s_p[0:3]]+a_s_p[3:]+c_to_p(a_s_p[6],a_s_p[7]))##这里改动
            # a_ss.append(a_s_p)
            i0+=1



    # print(a_s)

    ###############
    # 总分析绘图

        # plt.plot(z2, r2, 'go')
        # r-z线段
        # for i in range(len(rs)):
        #     # print(rs[i][0][2:4],rs[i][1][2:4])
        #     drawln2d(rs[i][0][2:4],rs[i][1][2:4],ax=plt)
            # drawln2d([z00,0],b_s[i][2:4],ax=plt)
            # drawln2d([-1.53792346,0],b_s[i][2:4],ax=plt)
        #####

    # 绘制rz
    # Drawrz(a_s)

    k_s=[]
    for i in range(len(a_s)):
        k,b = Findln([0,0],a_s[i][6:8])
        a_s[i].append(k)
        k_s.append(k)
        # print(k,b)

    a_s.sort(key=lambda x: x[-1])

    cp=categorize_points(a_s,0.001)
    # print(cp)

    # 绘制3d
    # Draw3Dscatter([a_s[i][4:7] for i in range(len(a_s))],ax=ax,color='b')
    # ax.scatter(0,0,0, c='r', marker='o')
    # ax.title(f"event {event0}\ndata: {data}")

    # for i in range(len(cp)):#画线
    #     cl = random_color()
    #     if len(cp[i])>5: print(i)
    #     for j in range(len(cp[i])-1):
    #         drawln3d(cp[i][j][:3], cp[i][j+1][:3], ax=ax, color=cl)

    # plt.show()

    # 绘制斜率分布
    # plt.figure()
    # plt.scatter(k_s,[0]*len(k_s))

    # 绘制xOy投影
    # plt.figure()
    # plt.scatter([a_s[i][4] for i in range(len(a_s))],[a_s[i][5] for i in range(len(a_s))])
    # # for i in range(len(cp)):#画线
    # #     cl = random_color()
    # #     if len(cp[i])>5: print(i)
    # #     for j in range(len(cp[i])-1):
    # #         drawln2d(cp[i][j][:2], cp[i][j+1][:2], ax=plt, color=cl)
    # plt.axis('equal') # 设置等比例

    # plt.figure()
    # print('cp',cp)


    # 统计量放在这里
    for j in range(len(cp)):#拟合
        color=random_color()
        # print(cp[j][1])
        # print([cp[j][i][4:6] for i in range(len(cp[j]))])
        ccx,ccy,r0,e0 = FitCircle([cp[j][i][4:6] for i in range(len(cp[j]))],delta_r=[1,2,3,5,7],draw=False,ax=plt)
        r0_s.append([ccx,ccy])
        cc_s.append(np.sqrt(cp[j][2][1]**2+cp[j][2][2]**2)/r0)
        pt_s.append(np.sqrt(cp[j][2][1]**2+cp[j][2][2]**2))
        cc_s_p.append([np.sqrt(cp[j][2][1]**2+cp[j][2][2]**2),cp[j][2][1],cp[j][2][2]])
        cc_s_r.append(r0)
        t.append([r0]+e0)
        p_T.append([cp[j][1][1],cp[j][1][2]])
        checkout.append(cp[j][0])
        # DrawEllipse([ccx,ccy], cp[j], r0, ax=ax, color=color)

print(f"event {nowevent} | finished")

r=r0_s

p=p_T

n=len(r)

r=np.array(r)
t0=np.array(t)
t0[:,1:]*=1000
t=t0[:,1:]
p=np.array(p)
p*=100

# 训练模型

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# 生成示例数据
X1,X2 = r,t
y=p
# 将输入数据展平为二维
X_flattened=np.concatenate((X1, X2), axis=1)
# X_flattened = X.reshape(100, -1)

# 划分训练集和测试集
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=test_size)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 创建 MLP 回归模型
mlp = MLPRegressor(hidden_layer_sizes=(90, 90),activation='tanh',
                   solver='adam', max_iter=1200)

# 训练模型
print("Training the model...")
mlp.fit(X_train, y_train)
print("Model trained!")

# 预测
y_pred = mlp.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 可视化结果
plt.figure(figsize=(12, 6))

# 绘制第一个目标值的实际值和预测值
# plt.subplot(1, 2, 1)
plt.scatter(y_test[:, 0],y_test[:, 1], color='blue', label='Actual')
plt.scatter(y_pred[:, 0],y_pred[:, 1], color='red', label='Predicted',s=15)
plt.title(f'vectors matching scatter plot | total: {totalevent}\ndataset: {data}\nMSE: {mse:.6f} | test_size: {test_size}')
plt.xlabel(r'$p_x$')
plt.ylabel(r'$p_y$')
plt.legend()
plt.savefig('figure_predict_MSE.png',bbox_inches="tight", dpi=800)


# 生成直方图
# 目标
trains=np.array(pt_s)
# 再训练
tests=mlp.predict(scaler.transform(X_flattened))/100
# 求模长
testsr0=[np.sqrt(tests[i][0]**2+tests[i][1]**2) for i in range(len(tests))]
want=testsr0/trains
notwant=np.array([np.sqrt(i[0]**2+i[1]**2) for i in r0_s])/testsr0/100
with open('testsr0.txt','w') as f:
    f.write("pT_actual\n")
    f.write(str(list(trains)))
    f.write("\npT_predicted\n")
    f.write(str(testsr0))

print(sorted(testsr0))

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# # matplotlib.rcParams['mathtext.default'] = 'regular'
# fig.suptitle(r"$r_{\mathrm{fitted}}/p_{T,\mathrm{true}}$ vs $p_{T,\mathrm{predicted}}/p_{T,\mathrm{true}}$"f'\ndata: {data}', y=1.16)
# fig.subplots_adjust(wspace=0.4)  # 增加 wspace 的值来加大子图间的距离
# ax1.ylabel("Frequency")
# ax2.set_ylabel("Frequency")

# ax1.hist(notwant,bins=60,edgecolor='black',color='purple',range=(0.8,1.2))
# ax1.xlabel(r"$r_{\mathrm{fitted}}/p_{T,\mathrm{true}}$")
# ax1.title(r"Using $r_{\mathrm{fitted}}/p_{T,\mathrm{true}}$"f"\nstd = {np.std(notwant):.8f}")

# ax2.hist(want,bins=60,edgecolor='black',color='purple',range=(0.9,1.1))
# ax2.set_xlabel(r"${p_{T,\mathrm{predicted}}} / {p_{T,\mathrm{real}}}$")
# ax2.set_title(r"Using $p_{T,\mathrm{predicted}}/p_{T,\mathrm{true}}$"f"\nstd = {np.std(want):.8f}")

# 四子图
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle(r"$r_{\mathrm{fitted}}/p_{T,\mathrm{true}}$ vs $p_{T,\mathrm{predicted}}/p_{T,\mathrm{true}}$"f'\ndata: {data}', y=1.05)
fig.subplots_adjust(wspace=0.4, hspace=0.5)  # 调整子图之间的水平和垂直间距

# 第一行保持不变
ax1.hist(notwant, bins=60, edgecolor='black', color='purple',range=(0.8,1.2))
ax1.set_xlabel(r"$\frac{r_{\mathrm{fitted}}}{100}/p_{T,\mathrm{true}}$")
ax1.set_ylabel("Frequency")
ax1.set_title(r"Using $\frac{r_{\mathrm{fitted}}}{100}/p_{T,\mathrm{true}}$"f"\nstd = {np.std(notwant):.8f}")

ax2.hist(want, bins=60, edgecolor='black', color='purple',range=(0.8,1.2))
ax2.set_xlabel(r"${p_{T,\mathrm{predicted}}} / {p_{T,\mathrm{real}}}$")
ax2.set_ylabel("Frequency")
ax2.set_title(r"Using $p_{T,\mathrm{predicted}}/p_{T,\mathrm{true}}$"f"\nstd = {np.std(want):.8f}")

# 新增第二行子图
ax3.hist(notwant, bins=60, edgecolor='black', color='purple',range=(0.8,1.2))
ax3.set_xlabel(r"$\frac{r_{\mathrm{fitted}}}{100}/p_{T,\mathrm{true}}$")
ax3.set_ylabel("Frequency~log")
ax3.set_yscale('log')  # 设置y轴为对数比例
ax3.set_title(r"Log-scale $\frac{r_{\mathrm{fitted}}}{100}/p_{T,\mathrm{true}}$")

ax4.hist(want, bins=60, edgecolor='black', color='purple',range=(0.8,1.2))
ax4.set_xlabel(r"${p_{T,\mathrm{predicted}}} / {p_{T,\mathrm{real}}}$")
ax4.set_ylabel("Frequency~log")
ax4.set_yscale('log')  # 设置y轴为对数比例
ax4.set_title(r"Log-scale $p_{T,\mathrm{predicted}}/p_{T,\mathrm{true}}$")




plt.savefig('figure_predict_partII.png',bbox_inches="tight", dpi=800)


plt.show()