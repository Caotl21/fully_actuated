import numpy as np
import matplotlib.pyplot as plt
import time

# ================== AGV参数 ==================
L = 1.0                  # 轴距(m)
Ts = 0.1                 # 控制周期(s)
v_max = 2.0              # 最大速度(m/s)
delta_max = np.deg2rad(30)  # 最大转向角(rad)

# 初始状态 [x, y, theta, v, delta]
state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# ================== 控制器参数 ==================
kp = [3.0, 3.0]          # 比例增益
kd = [2.0, 2.0]          # 微分增益
I_max = 5.0              # 积分项限幅

# ================== 轨迹类型枚举 ==================
class TrajectoryType:
    STRAIGHT = 0
    CIRCLE = 1
    SINE = 2

# ================== 轨迹生成器 ==================
def generate_reference(traj_type, t):
    x_ref = np.zeros_like(t)
    y_ref = np.zeros_like(t)
    dx_ref = np.zeros_like(t)
    dy_ref = np.zeros_like(t)
    ddx_ref = np.zeros_like(t)
    ddy_ref = np.zeros_like(t)
    
    for i, ti in enumerate(t):
        if traj_type == TrajectoryType.STRAIGHT:
            # 直线轨迹
            x_ref[i] = ti
            dx_ref[i] = 1.0
        elif traj_type == TrajectoryType.CIRCLE:
            # 圆形轨迹 (半径2m，角速度0.5rad/s)
            r, omega = 2.0, 0.5
            x_ref[i] = r * np.cos(omega*ti)
            y_ref[i] = r * np.sin(omega*ti)
            dx_ref[i] = -r*omega*np.sin(omega*ti)
            dy_ref[i] = r*omega*np.cos(omega*ti)
            ddx_ref[i] = -r*omega**2*np.cos(omega*ti)
            ddy_ref[i] = -r*omega**2*np.sin(omega*ti)
        elif traj_type == TrajectoryType.SINE:
            # 正弦轨迹
            A, omega = 2.0, 0.5
            x_ref[i] = ti
            y_ref[i] = A*np.sin(omega*ti)
            dx_ref[i] = 1.0
            dy_ref[i] = A*omega*np.cos(omega*ti)
            ddx_ref[i] = 0.0
            ddy_ref[i] = -A*omega**2*np.sin(omega*ti)
    return x_ref, y_ref, dx_ref, dy_ref, ddx_ref, ddy_ref

# ================== 全驱控制器 ==================
class FullDriveController:
    def __init__(self):
        self.u_prev = np.zeros(2)
        self.d_hat = np.zeros(2)  # 扰动估计
        
    def compute_control(self, state, xd, yd, dxd, dyd, ddxd, ddyd):
        x, y, theta, v, delta = state
        
        # 跟踪误差
        ex = x - xd
        ey = y - yd
        dex = v * np.cos(theta) - dxd
        dey = v * np.sin(theta) - dyd
        
        # 虚拟控制量
        u1_prime = ddxd - kp[0]*ex - kd[0]*dex
        u2_prime = ddyd - kp[1]*ey - kd[1]*dey
        
        # 计算雅可比矩阵
        J = np.array([
            [np.cos(theta), -v**2/L * np.sin(theta)*np.tan(delta)],
            [np.sin(theta),  v**2/L * np.cos(theta)*np.tan(delta)]
        ])
        
        # 扰动估计（示例：简单低通滤波）
        self.d_hat = 0.9*self.d_hat + 0.1*np.array([ex, ey])
        
        # 计算实际控制量
        try:
            u_actual = np.linalg.solve(J, [u1_prime, u2_prime])
        except np.linalg.LinAlgError:
            u_actual = np.zeros(2)
        
        # 控制量限幅
        dv = np.clip(u_actual[0], -0.5, 0.5)
        d_delta = np.clip(u_actual[1], -delta_max, delta_max)
        
        return np.array([dv, d_delta])

# ================== 主程序 ==================
if __name__ == "__main__":
    # 仿真参数
    sim_time = 30.0
    t = np.arange(0, sim_time, Ts)
    selected_traj = TrajectoryType.STRAIGHT
    
    # 生成参考轨迹
    x_ref, y_ref, dx_ref, dy_ref, ddx_ref, ddy_ref = generate_reference(selected_traj, t)
    
    # 初始化控制器
    controller = FullDriveController()
    
    # 存储数据
    x_hist, y_hist = [], []
    u_hist = []
    e_hist = []
    d_hat_hist = []
    
    # 实时绘图设置
    plt.ion()
    fig = plt.figure(figsize=(15, 10))
    
    # 主循环
    for k in range(len(t)):
        # 获取当前参考轨迹
        xd, yd = x_ref[k], y_ref[k]
        dxd, dyd = dx_ref[k], dy_ref[k]
        ddxd, ddyd = ddx_ref[k], ddy_ref[k]
        
        # 计算控制输入
        u = controller.compute_control(state, xd, yd, dxd, dyd, ddxd, ddyd)
        
        # 施加控制输入 (欧拉积分)
        dv, d_delta = u
        state[3] += dv * Ts          # 更新速度 v
        state[4] += d_delta * Ts     # 更新转向角 delta
        state[4] = np.clip(state[4], -delta_max, delta_max)
        
        # 更新位置和航向
        state[0] += state[3] * np.cos(state[2]) * Ts  # x
        state[1] += state[3] * np.sin(state[2]) * Ts  # y
        state[2] += (state[3]/L) * np.tan(state[4]) * Ts  # theta
        
        # 记录数据
        x_hist.append(state[0])
        y_hist.append(state[1])
        u_hist.append(u)
        e_hist.append(np.linalg.norm([state[0]-xd, state[1]-yd]))
        d_hat_hist.append(controller.d_hat.copy())
        
        # 实时绘图更新 (每10步更新一次)
        if k % 10 == 0:
            plt.clf()
            
            # 轨迹跟踪
            plt.subplot(2, 2, 1)
            plt.plot(x_ref, y_ref, 'r--', label='Reference')
            plt.plot(x_hist, y_hist, 'b-', label='Actual')
            plt.plot(state[0], state[1], 'go', markersize=8, label='Current')
            plt.title(f'Trajectory Tracking (t={t[k]:.1f}s)')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.axis('equal')
            plt.legend()
            plt.grid(True)
            
            # 跟踪误差
            plt.subplot(2, 2, 2)
            plt.plot(t[:k+1], e_hist, 'r-', label='Position Error')
            plt.title('Tracking Error')
            plt.xlabel('Time (s)')
            plt.ylabel('Error Norm (m)')
            plt.legend()
            plt.grid(True)
            
            # 控制输入
            plt.subplot(2, 2, 3)
            u_array = np.array(u_hist)
            plt.plot(t[:k+1], u_array[:,0], 'b-', label='dv')
            plt.plot(t[:k+1], u_array[:,1], 'g-', label='d_delta')
            plt.title('Control Inputs')
            plt.xlabel('Time (s)')
            plt.ylabel('Input Value')
            plt.legend()
            plt.grid(True)
            
            # 扰动估计
            plt.subplot(2, 2, 4)
            d_hat_array = np.array(d_hat_hist)
            plt.plot(t[:k+1], d_hat_array[:,0], 'm-', label='d_x')
            plt.plot(t[:k+1], d_hat_array[:,1], 'c-', label='d_y')
            plt.title('Disturbance Estimation')
            plt.xlabel('Time (s)')
            plt.ylabel('Estimation')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.pause(0.001)
    
    plt.ioff()
    plt.show()