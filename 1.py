import numpy as np
import matplotlib.pyplot as plt

# 参数设置
L = 0.6       # 轴距 (m)
kx = 2.0      # 位置控制增益
ky = 2.0
k_theta = 1.5 # 航向控制增益
dt = 0.1      # 采样时间
sim_time = 30 # 仿真时间

# 参考轨迹（圆形）
t = np.arange(0, sim_time, dt)
x_ref = 3 * np.cos(0.5 * t)
y_ref = 3 * np.sin(0.5 * t)
dx_ref = -1.5 * np.sin(0.5 * t)
dy_ref = 1.5 * np.cos(0.5 * t)
ddx_ref = -0.75 * np.cos(0.5 * t)
ddy_ref = -0.75 * np.sin(0.5 * t)

# 初始化AGV状态
x, y, theta, v = 3.0, 0.0, 0.0, 0.0
x_hist, y_hist = [], []

for i in range(len(t)):
    # 跟踪误差
    ex = x - x_ref[i]
    ey = y - y_ref[i]
    
    # 外环位置控制 (计算期望速度和航向)
    v_des = np.sqrt( (dx_ref[i] - kx*ex)**2 + (dy_ref[i] - ky*ey)**2 )
    theta_des = np.arctan2(dy_ref[i] - ky*ey, dx_ref[i] - kx*ex)
    
    # 内环航向控制 (计算转向角phi)
    phi = np.arctan( L / v_des * (k_theta * (theta_des - theta)) ) if v_des > 0.1 else 0.0
    phi = np.clip(phi, -np.pi/4, np.pi/4)  # 转向角限幅
    
    # 更新速度（假设速度可瞬时调节）
    v = v_des
    
    # 更新状态（欧拉积分）
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += (v / L) * np.tan(phi) * dt
    
    # 记录轨迹
    x_hist.append(x)
    y_hist.append(y)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(x_ref, y_ref, 'r--', label='Reference')
plt.plot(x_hist, y_hist, 'b-', label='AGV Trajectory')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.title('全驱AGV轨迹跟踪仿真（修正模型）')
plt.grid(True)
plt.show()