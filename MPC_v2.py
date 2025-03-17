import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm
from enum import Enum

# ================== 轨迹类型定义 ==================
class TrajectoryType(Enum):
    STRAIGHT = 1     # 直线轨迹
    CIRCLE = 2       # 圆形轨迹
    DOUBLE_LANE = 3  # 双移线轨迹

# ================== 参数设置 ==================
# 车辆参数
L = 0.6          # 舵轮轴距 (m)
phi_max = np.deg2rad(30)  # 最大转向角
Ts = 0.05        # 缩短采样时间

# MPC参数
N = 15           # 延长预测时域
M = 8            # 增加控制时域
Q = np.diag([50, 5, 50, 5, 0.1])  # 调整后的状态权重
R = np.diag([0.01, 0.01])          # 降低输入权重
S = np.diag([0.1, 0.1])            # 输入增量权重

# 约束条件
w_max = np.array([3.0, 2.0])        # 放宽输入限制
delta_w_max = 1.0                   # 放宽输入增量
e_max = 0.3                        
I_max = 1.0                        

# 仿真参数
sim_time = 30.0                    
t = np.arange(0, sim_time, Ts)
selected_traj = TrajectoryType.STRAIGHT 

# ================== 轨迹生成器 ==================  
def generate_reference(traj_type, t):
    """
    生成参考轨迹
    返回: x_ref, y_ref, dx_ref, dy_ref, ddx_ref, ddy_ref
    """
    if traj_type == TrajectoryType.STRAIGHT:
        # 直线轨迹 (沿x轴匀速运动)
        v = 2.0  # m/s
        x = v * t
        y = np.zeros_like(t)
        dx = v * np.ones_like(t)
        dy = np.zeros_like(t)
        ddx = np.zeros_like(t)
        ddy = np.zeros_like(t)
    
    elif traj_type == TrajectoryType.CIRCLE:
        # 圆形轨迹 (半径3m，周期8s)
        R = 3.0
        omega = 2*np.pi/8
        x = R * np.cos(omega * t)
        y = R * np.sin(omega * t)
        dx = -R*omega*np.sin(omega*t)
        dy = R*omega*np.cos(omega*t)
        ddx = -R*omega**2*np.cos(omega*t)
        ddy = -R*omega**2*np.sin(omega*t)
    
    elif traj_type == TrajectoryType.DOUBLE_LANE:
        # 双移线轨迹 (两次换道)
        A = 2.5  # 换道幅度
        T = 5.0   # 换道周期
        x = np.linspace(0, 50, len(t))
        y = A * np.sin(2*np.pi*t/T) + A * np.sin(4*np.pi*t/T)
        
        dx = np.gradient(x, t)
        dy = np.gradient(y, t)
        ddx = np.gradient(dx, t)
        ddy = np.gradient(dy, t)
    
    else:
        raise ValueError("未知轨迹类型")
    
    # 曲率检查
    #curvature = np.abs(ddx*dy - ddy*dx) / (dx**2 + dy**2)**1.5
    #if np.any(1/curvature < L/np.tan(phi_max)):
    #    print("警告：轨迹包含不可达点！")

    return x, y, dx, dy, ddx, ddy

# 生成选定轨迹
x_ref, y_ref, dx_ref, dy_ref, ddx_ref, ddy_ref = generate_reference(selected_traj, t)

# ================== 模型离散化 ==================
def discretize(A, B, dt):
    n = A.shape[0]
    M = np.block([
        [A, B], 
        [np.zeros((B.shape[1], n+B.shape[1]))]
        ])
    M_exp = expm(M*dt)
    Ad = M_exp[:n, :n]
    Bd = M_exp[:n, n:]
    return Ad, Bd

A_c = np.array([
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [Ts, 0, Ts, 0, 1]
])
B_c = np.array([
    [0, 0],
    [-1, 0],
    [0, 0],
    [0, -1],
    [0, 0]
])

A_delta, B_delta = discretize(A_c, B_c, Ts)
D_delta = np.array([    [0, 0],
    [Ts, 0],
    [0, 0],
    [0, Ts],
    [0, 0]
])

# ================== 扰动观测器 ==================
class DisturbanceObserver:
    def __init__(self):
        self.d_hat = np.zeros(2)
        self.prev_error = np.zeros(2)
    
    def update(self, error, u, dt=Ts):
        # 简单的一阶观测器
        alpha = 0.9  # 滤波系数
        self.d_hat = alpha * self.d_hat + (1-alpha) * (error - self.prev_error)/dt
        self.prev_error = error
        return self.d_hat

# ================== MPC控制器 ==================
class RobustMPC:
    def __init__(self):
        self.obs = DisturbanceObserver()
        self.u_prev = np.zeros(2)
        
        # 优化变量
        self.U = cp.Variable((M, 2))
        self.xi = cp.Parameter(5)
        self.xi_ref = cp.Parameter((N, 5))
        
        # 构建优化问题
        cost = 0
        constraints = []
        xi_pred = self.xi
        
        for i in range(N):
            if i < M:
                u = self.U[i]
            else:
                u = self.U[-1]
            
            # 状态预测 + 前馈补偿
            xi_pred = A_delta @ xi_pred + B_delta @ u + D_delta @ self.xi_ref[i, 2:4]
            
            # 代价函数
            cost += cp.quad_form(xi_pred - self.xi_ref[i], Q)
            if i < M:
                cost += cp.quad_form(u, R)
                if i > 0:
                    cost += cp.quad_form(u - self.U[i-1], S)
            
            # 约束
            constraints += [
                cp.abs(xi_pred[0]) <= e_max,
                cp.abs(xi_pred[2]) <= e_max,
                cp.abs(xi_pred[4]) <= I_max,
                cp.abs(u) <= w_max
            ]
            if i > 0 and i < M:
                constraints += [cp.abs(u - self.U[i-1]) <= delta_w_max]
        
        self.problem = cp.Problem(cp.Minimize(cost), constraints)
    
    def solve(self, xi_current, xi_ref, d_hat):
        self.xi.value = xi_current
        self.xi_ref.value = xi_ref
        
        try:
            self.problem.solve(solver=cp.OSQP, verbose=False)
            if self.problem.status == cp.OPTIMAL:
                u_opt = self.U.value[0] - d_hat  # 扰动补偿
                return np.clip(u_opt, -w_max, w_max)
            return np.zeros(2)
        except:
            return np.zeros(2)

# ================== 仿真初始化 ==================
mpc = RobustMPC()
xi = np.zeros(5)  # [e_x, Δe_x, e_y, Δe_y, I_e]

# 存储数据
x_hist, y_hist = [], []
u_hist, e_hist = [], []

# ================== 可视化 ==================
def plot_trajectory_comparison():
    plt.figure(figsize=(12, 6))
    
    # 绘制所有轨迹类型示例
    plt.subplot(1, 2, 1)
    t_example = np.arange(0, 20, 0.1)
    
    # 直线
    x, y, *_ = generate_reference(TrajectoryType.STRAIGHT, t_example)
    plt.plot(x, y, label='Straight')
    
    # 圆形
    x, y, *_ = generate_reference(TrajectoryType.CIRCLE, t_example)
    plt.plot(x, y, label='Circle')
    
    # 双移线
    x, y, *_ = generate_reference(TrajectoryType.DOUBLE_LANE, t_example)
    plt.plot(x, y, label='Double Lane')
    
    plt.axis('equal')
    plt.title('Trajectory Examples')
    plt.legend()
    
    # 绘制实际跟踪效果
    plt.subplot(1, 2, 2)
    plt.plot(x_ref, y_ref, 'r--', label='Reference')
    plt.plot(x_hist, y_hist, 'b-', label='Actual')
    plt.title(f'Tracking Result: {selected_traj.name}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


# ================== 主循环 ==================
for k in range(len(t)):
    # 参考轨迹生成
    xi_ref = np.zeros((N, 5))
    for i in range(N):
        if k+i < len(t):
            # 积分项计算
            if k+i > 0 and len(x_hist) >= k+i:
                I_e = np.sum(x_ref[:k+i] - x_hist[:k+i]) + np.sum(y_ref[:k+i] - y_hist[:k+i])
                I_e = np.clip(I_e, -I_max, I_max)
            else:
                I_e = 0
            
            xi_ref[i] = [0, 0, 0, 0, I_e]
    
    # 求解MPC
    current_error = np.array([xi[0], xi[2]])
    d_hat = mpc.obs.update(current_error, mpc.u_prev)
    u_opt = mpc.solve(xi, xi_ref, d_hat)
    
    # 施加扰动 (5-6秒)
    if 5.0 <= t[k] < 6.0:
        u_opt += np.array([0, 0.8])
    
    # 前馈补偿
    u_ff = np.array([ddx_ref[k], ddy_ref[k]])
    u_total = u_opt + 0.5*u_ff  # 前馈增益
    
    # 状态更新
    xi_next = A_delta @ xi + B_delta @ u_total + D_delta @ np.array([ddx_ref[k], ddy_ref[k]])
    
    # 积分抗饱和
    if np.linalg.norm([xi[0], xi[2]]) < 0.2:
        xi_next[4] = np.clip(xi[4] + Ts*(xi[0]+xi[2]), -I_max, I_max)
    else:
        xi_next[4] = 0
    
    # 记录数据
    x_hist.append(x_ref[k] + xi[0])
    y_hist.append(y_ref[k] + xi[2])
    u_hist.append(u_total)
    e_hist.append(xi[0]**2 + xi[2]**2)
    
    # 更新状态
    xi = xi_next
    mpc.u_prev = u_total
    

# 在仿真结束后调用
plot_trajectory_comparison()