import math as m

# 极坐标变换
def c_to_p(x, y):
    r = m.sqrt(x**2 + y**2)
    theta = m.atan2(y, x)  # 以弧度表示
    theta_degrees = m.degrees(theta)  # 转换为度数
    if theta_degrees < 0:
        theta_degrees += 360
    return r, theta_degrees

# 示例使用
x = 2
y = 0
r, theta_degrees = c_to_p(x, y)
print(f"极坐标：(r, θ) = ({r}, {theta_degrees}°)")
