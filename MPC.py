import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import expm
from enum import Enum

# ================== 轨迹类型定义 ==================
class TrajectoryType(Enum):
    STRAIGHT = 1     # 直线轨迹
    CIRCLE = 2       # 圆形轨迹
    DOUBLE_LANE = 3  # 双移线轨迹

# ================== 参数设置 ==================
# 车辆参数
L = 0.6          # 舵轮轴距 (m)
phi_max = np.deg2rad(45)  # 最大转向角
Ts = 0.1         # 采样时间

# MPC参数
N = 10           # 预测时域
M = 5            # 控制时域
Q = np.diag([50, 50])    # 状态权重 [x, y]
R = np.diag([0.1, 0.1])  # 输入权重

# 约束条件
w_max = np.array([3.0, 2.0])  # 最大加速度

# ================== 轨迹生成器 ==================  
def generate_reference(traj_type, t):
    """生成参考轨迹"""
    if traj_type == TrajectoryType.STRAIGHT:
        v = 2.0  # m/s
        x = v * t
        y = v * t
    elif traj_type == TrajectoryType.CIRCLE:
        R = 3.0
        omega = 2*np.pi/10
        x = R * np.cos(omega * t)
        y = R * np.sin(omega * t)
    elif traj_type == TrajectoryType.DOUBLE_LANE:
        A = 2.0
        T = 8.0
        x = np.linspace(0, 50, len(t))
        y = A * np.sin(2*np.pi*t/T)
    else:
        raise ValueError("未知轨迹类型")
    
    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    return x, y, dx, dy

# ================== 全驱模型定义 ==================
def dynamics(x, u):
    """全驱系统动力学"""
    A = np.array([[0, 0], [0, 0]])
    B = np.array([[1, 0], [0, 1]])
    return A @ x + B @ u

def discretize_model(dt):
    """模型离散化（欧拉近似）"""
    A_d = np.eye(2)
    B_d = dt * np.eye(2)
    return A_d, B_d

A_d, B_d = discretize_model(Ts)

# ================== MPC控制器 ==================
class FullDriveMPC:
    def __init__(self):
        self.U = cp.Variable((M, 2))  # 控制输入序列
        self.x0 = cp.Parameter(2)     # 初始状态
        self.x_ref = cp.Parameter((N, 2))  # 参考轨迹
        
        # 构建优化问题
        cost = 0
        self.states = [self.x0]
        for t in range(N):
            u = self.U[t] if t < M else self.U[-1]
            next_state = A_d @ self.states[-1] + B_d @ u
            self.states.append(next_state)
            cost += cp.quad_form(next_state - self.x_ref[t], Q)
            if t < M:
                cost += cp.quad_form(u, R)
        
        # 输入约束
        constraints = [cp.abs(self.U) <= w_max]
        
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, current_state, x_ref_traj):
        self.x0.value = current_state
        self.x_ref.value = x_ref_traj
        self.problem.solve(solver=cp.OSQP)
        return self.U.value[0] if self.problem.status == cp.OPTIMAL else np.zeros(2)

# ================== 主程序 ==================
def main():
    # 轨迹参数
    traj_type = TrajectoryType.CIRCLE
    sim_time = 50.0
    t = np.arange(0, sim_time, Ts)
    
    # 生成参考轨迹
    x_ref, y_ref, dx_ref, dy_ref = generate_reference(traj_type, t)
    
    # 初始化控制器
    mpc = FullDriveMPC()
    
    # 初始状态
    current_state = np.array([x_ref[0], y_ref[0]])
    
    # 存储数据
    x_hist, y_hist = [], []
    
    plt.figure(figsize=(12, 6))
    
    for k in range(len(t)-N):
        # 获取参考轨迹窗口
        x_ref_window = np.column_stack((x_ref[k:k+N], y_ref[k:k+N]))
        
        # 求解MPC
        u_opt = mpc.solve(current_state, x_ref_window)
        
        # 状态更新
        current_state = A_d @ current_state + B_d @ u_opt
        
        # 记录数据
        x_hist.append(current_state[0])
        y_hist.append(current_state[1])
        
        # 实时绘图
        if k % 10 == 0:
            plt.clf()
            plt.plot(x_ref, y_ref, 'r--', label='Reference')
            plt.plot(x_hist, y_hist, 'b-', label='Actual')
            plt.title(f'Tracking @ t={t[k]:.1f}s')
            plt.legend()
            plt.pause(0.01)
            
            
            
           
    
    plt.show()

if __name__ == "__main__":
    main()