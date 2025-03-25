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
Ts = 0.01        # 采样时间
v_max = 5.0      # 最大速度

# MPC参数
N = 15           # 预测时域
M = 8            # 控制时域
Q = np.diag([100, 10, 100, 10])  # 状态权重 [x, vx, y, vy]
R = np.diag([0.01, 0.01])          # 输入权重 [ax, ay]
S = np.diag([0.1, 0.1])          # 输入增量权重

# 约束条件
a_max = 3.0                      # 最大加速度
delta_a_max = 1.0                 # 加速度增量限制

# ================== 轨迹生成器 ==================  
def generate_reference(traj_type, t):
    """
    生成参考轨迹
    返回: x_ref, y_ref, dx_ref, dy_ref, ddx_ref, ddy_ref
    """
    if traj_type == TrajectoryType.STRAIGHT:
        # 直线轨迹 (沿x轴匀速运动)
        v = 1.0  # m/s
        x = v * t
        #y = np.zeros_like(t)
        y = v * t
        dx = v * np.ones_like(t)
        dy = np.ones_like(t)
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
        A = 2.0  # 换道幅度
        T = 9.0   # 换道周期
        x = np.linspace(0, 50, len(t))
        y = A * np.sin(2*np.pi*t/T) + A * np.sin(4*np.pi*t/T)
        
        dx = np.gradient(x, t)
        dy = np.gradient(y, t)
        ddx = np.gradient(dx, t)
        ddy = np.gradient(dy, t)
    
    else:
        raise ValueError("未知轨迹类型")
    
    # 曲率检查
    curvature = np.abs(ddx*dy - ddy*dx) / (dx**2 + dy**2)**1.5
    if np.any(1/curvature < L/np.tan(phi_max)):
        print("警告：轨迹包含不可达点！")

    return x, y, dx, dy, ddx, ddy

# ================== 反馈线性化模型 ==================
class LinearizedModel:
    def __init__(self):
        # 连续时间系统矩阵 (双积分器)
        self.A_c = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        self.B_c = np.array([
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 1]
        ])
        
        # 离散化
        self.Ad, self.Bd = self.discretize(Ts)
    
    def discretize(self, dt):
        """零阶保持离散化"""
        n = self.A_c.shape[0]
        M = np.block([[self.A_c, self.B_c],
                     [np.zeros((self.B_c.shape[1], n + self.B_c.shape[1]))]])
        M_exp = expm(M*dt)
        Ad = M_exp[:n, :n]
        Bd = M_exp[:n, n:]
        return Ad, Bd
    
    def convert_to_physical(self, state, u):
        """将虚拟控制量转换为物理量"""
        x, vx, y, vy = state
        theta = np.arctan2(vy, vx) if (vx**2 + vy**2) > 1e-6 else 0.0
        v = np.sqrt(vx**2 + vy**2)
        
        # 计算实际控制输入
        ax, ay = u
        phi = np.arctan(L * (ay*vx - ax*vy) / (v**3 + 1e-6))
        #phi = np.clip(phi, -phi_max, phi_max)
        
        return phi, ax, ay

# ================== MPC控制器 ==================
class HOFAMPC:
    def __init__(self, model):
        self.model = model
        #预测域为N，控制域为M
        self.U = cp.Variable((M, 2))  # 确保维度为(M,2)
        self.x0 = cp.Parameter(4)
        self.x_ref = cp.Parameter((N, 4))
        
        cost = 0
        constraints = []
        x_pred = [self.x0]
        
        for i in range(N):
            # 控制输入选择逻辑修正
            if i < M:
                u = self.U[i]
            else:
                u = self.U[-1]  # 超过控制时域使用最后一个输入
                #u = (0 , 0)         # 超过控制时域令u = 0
            
            # 状态预测
            x_next = model.Ad @ x_pred[-1] + model.Bd @ u
            x_pred.append(x_next)
            
            # 代价函数
            cost += cp.quad_form(x_next - self.x_ref[i], Q)
            if i < M:
                cost += cp.quad_form(u, R)
                #cost += cp.quad_form(u - self.U[i-1], R)
                if i > 0:  # 仅对i < M的输入施加增量约束
                    constraints += [
                        cp.norm(u - self.U[i-1], 'inf') <= delta_a_max
                    ]
            
            # 输入幅值约束（对所有i有效）
            constraints += [
                cp.abs(u[0]) <= a_max,
                cp.abs(u[1]) <= a_max
            ]
        
        self.problem = cp.Problem(cp.Minimize(cost), constraints)
    
    def solve(self, current_state, ref_traj):
        
        
        self.x0.value = current_state
        self.x_ref.value = ref_traj
        self.problem.solve(solver=cp.OSQP, verbose=False)
        
        if self.problem.status == cp.OPTIMAL:
            #print(f"MPC status: {self.problem.status}")
            return self.U.value[0]
        else:
            return np.zeros(2)

# ================== 主程序 ==================
if __name__ == "__main__":
    # 初始化模型和控制器
    model = LinearizedModel()
    mpc = HOFAMPC(model)
    
    # 生成参考轨迹
    traj_type = TrajectoryType.CIRCLE
    sim_time = 30.0
    t = np.arange(0, sim_time, Ts)
    x_ref, y_ref, dx_ref, dy_ref, ddx_ref, ddy_ref = generate_reference(traj_type, t)
    
    # 初始化车辆状态
    state = np.array([x_ref[0], dx_ref[0], y_ref[0], dy_ref[0]])  # [x, vx, y, vy]
    #state = np.array([0, 0, 10, 0])  # [x, vx, y, vy]
    x_hist, y_hist, error_hist, theta_hist, u_hist = [], [], [], [], []
    v = np.sqrt(dx_ref[0]**2+dy_ref[0]**2)
    theta =  np.arctan2(dy_ref[0], dx_ref[0])
    MSE_post = 0
    MSE_theta = 0
    num = len(t)-N
    
    # 实时绘图设置
    plt.ion()
    fig = plt.figure(figsize=(15, 10))
    
    for k in range(len(t)-N):
        # 生成参考轨迹窗口
        ref_window = np.column_stack([
            x_ref[k:k+N], 
            dx_ref[k:k+N], 
            y_ref[k:k+N], 
            dy_ref[k:k+N]
        ])
        
        # 求解MPC
        u_opt = mpc.solve(state, ref_window)
        
        # 转换为物理控制量
        phi, ax, ay = model.convert_to_physical(state, u_opt)
        delta_v = np.sqrt(ax**2 + ay**2)
        u = [delta_v, phi]
        
        # 状态更新 (物理模型)
        vx = state[1] + ax * Ts
        vy = state[3] + ay * Ts
        x = state[0] + state[1]*Ts + 0.5*ax*Ts**2
        y = state[2] + state[3]*Ts + 0.5*ay*Ts**2
        theta = np.arctan2(vy, vx)
        error_post = np.sqrt((x - x_ref[k])**2 + (y - y_ref[k])**2)
        error_theta = (theta - (np.arctan2(dx_ref[k+1], dy_ref[k+1])))**2
        
        # 记录状态
        state = np.array([x, vx, y, vy])
        x_hist.append(x)
        y_hist.append(y)
        theta_hist.append(theta)
        u_hist.append(u)
        error_hist.append(error_post)
        
        #计算算法性能指标
        MSE_post += (error_post**2)
        MSE_theta += error_theta
        #MSCI +=
        
        # 实时绘图
        if k % 10 == 0:
            plt.clf()
            
            # 轨迹跟踪
            plt.subplot(2, 2, 1)
            plt.plot(x_ref, y_ref, 'r--', label='Reference')
            plt.plot(x_hist, y_hist, 'b-', label='Actual')
            plt.title(f'Tracking @ t={t[k]:.1f}s')
            plt.legend()
            
            # 控制输入
            plt.subplot(2, 2, 2)
            plt.plot(t[:k+1], np.array(u_hist)[:,0], label='a')
            plt.plot(t[:k+1], np.array(u_hist)[:,1], label='phi')
            plt.title('Control Inputs')
            plt.legend()
            
            # 航向角
            plt.subplot(2, 2, 3)
            plt.plot(t[:k+1], theta_hist, label='Heading Angle')
            plt.title('Vehicle Orientation')
            plt.legend()
            
            # 速度曲线
            plt.subplot(2, 2, 4)
            plt.plot(t[:k+1], error_hist, label='Error')
            plt.title('Tracking Error')
            plt.legend()
            
            plt.tight_layout()
            plt.pause(0.01)
    
    MSE_post_out = 'MSE of post is ' + repr(MSE_post / num)
    print(MSE_post_out)
    MSE_theta_out = 'MSE of theta is ' + repr(MSE_theta / num)
    print(MSE_theta_out)
    
    
    plt.ioff()
    plt.show()