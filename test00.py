import numpy as np

# 常量
num_points = 20
z_min, z_max = -5, 5

# 随机生成z坐标
z_coords = np.random.uniform(z_min, z_max, num_points)

# 小柱面上的点 (x^2 + y^2 = 4, 因此r=2)
angles = np.random.uniform(0, 2 * np.pi, num_points)
x_coords_small = 2 * np.cos(angles)
y_coords_small = 2 * np.sin(angles)

# 生成小柱面上的20个点
points_small_cylinder = np.vstack((x_coords_small, y_coords_small, z_coords)).T

# 点 p0
p0 = np.array([0, 0, 0])

# 计算与大柱面 (x^2 + y^2 = 16, 因此r=4) 的交点
points_large_cylinder = []

for point in points_small_cylinder:
    x0, y0, z0 = point
    t = 4 / np.sqrt(x0**2 + y0**2)  # 缩放因子
    x1, y1, z1 = t * x0, t * y0, t * z0
    points_large_cylinder.append([x1, y1, z1])

points_large_cylinder = np.array(points_large_cylinder)

# 输出小柱面上的点
print("小柱面上的点坐标:")
for point in points_small_cylinder:
    print(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}")

# 输出大柱面上的交点
print("\n大柱面上的交点坐标:")
for point in points_large_cylinder:
    print(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}")
