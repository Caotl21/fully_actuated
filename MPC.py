import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ================== 配置参数 ==================
class Config:
    L = 0.6
    Ts = 0.1
    N = 15
    M = 8
    Q = np.diag([50, 5, 50, 5, 0.1])
    R = np.diag([0.05, 0.05])
    S = np.diag([1, 1])
    w_max = np.array([2.0, 1.5])
    delta_w_max = 0.5
    e_max = 0.3
    I_max = 1.0
    sim_time = 20.0
    traj_scale = 5.0

config = Config()

# ================== 轨迹生成 ==================
def generate_ref_traj(t):
    omega = 2 * np.pi / config.sim_time
    x_ref = config.traj_scale * np.sin(omega * t)
    y_ref = config.traj_scale * np.sin(2 * omega * t)
    dx_ref = config.traj_scale * omega * np.cos(omega * t)
    dy_ref = 2 * config.traj_scale * omega * np.cos(2 * omega * t)
    ddx_ref = -config.traj_scale * omega**2 * np.sin(omega * t)
    ddy_ref = -4 * config.traj_scale * omega**2 * np.sin(2 * omega * t)
    return x_ref, y_ref, dx_ref, dy_ref, ddx_ref, ddy_ref

t = np.arange(0, config.sim_time, config.Ts)
x_ref, y_ref, dx_ref, dy_ref, ddx_ref, ddy_ref = generate_ref_traj(t)

# ================== 系统模型 ==================
class EnhancedIncrementalModel:
    def __init__(self):
        self.A = np.array([
            [1, config.Ts, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, config.Ts, 0],
            [0, 0, 0, 1, 0],
            [config.Ts, 0, config.Ts, 0, 1]
        ])
        self.B = np.array([
            [0, 0],
            [-config.Ts, 0],
            [0, 0],
            [0, -config.Ts],
            [0, 0]
        ])
        self.D = np.array([
            [0, 0],
            [config.Ts, 0],
            [0, 0],
            [0, config.Ts],
            [0, 0]
        ])
    
    def predict(self, xi, u, ddx, ddy):
        return self.A @ xi + self.B @ u + self.D @ np.array([ddx, ddy])

model = EnhancedIncrementalModel()

# ================== 控制器 ==================
class RobustMPC:
    def __init__(self):
        self.u_prev = np.zeros(2)
        self._init_optimization()
    
    def _init_optimization(self):
        self.U = cp.Variable((config.M, 2))
        self.xi = cp.Parameter(5)
        self.xi_ref = cp.Parameter((config.N, 5))
        
        cost = 0
        constraints = []
        xi_pred = self.xi
        
        for i in range(config.N):
            if i < config.M:
                u = self.U[i]
            else:
                u = 0.8 * self.U[-1]
            
            xi_pred = model.A @ xi_pred + model.B @ u + model.D @ self.xi_ref[i, 2:4]
            cost += cp.quad_form(xi_pred - self.xi_ref[i], config.Q)
            
            if i < config.M:
                cost += cp.quad_form(u, config.R)
                if i > 0:
                    cost += cp.quad_form(u - self.U[i-1], config.S)
            
            constraints += [
                cp.abs(xi_pred[0]) <= config.e_max,
                cp.abs(xi_pred[2]) <= config.e_max,
                cp.abs(xi_pred[4]) <= config.I_max,
                cp.norm(u, 'inf') <= np.linalg.norm(config.w_max)
            ]
            
            if i < config.M:
                constraints += [
                    cp.abs(u[0]) <= config.w_max[0],
                    cp.abs(u[1]) <= config.w_max[1],
                    cp.abs(u - self.U[i-1]) <= config.delta_w_max if i > 0 else None
                ]
        
        self.problem = cp.Problem(cp.Minimize(cost), [c for c in constraints if c is not None])
    
    def solve(self, xi_current, xi_ref):
        self.xi.value = xi_current
        self.xi_ref.value = xi_ref
        
        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True)
            return self.U.value[0] if self.problem.status == cp.OPTIMAL else np.clip(self.u_prev, -config.w_max, config.w_max)
        except:
            return np.clip(self.u_prev, -config.w_max, config.w_max)

# ================== 仿真循环 ==================
def run_simulation():
    mpc = RobustMPC()
    xi = np.zeros(5)
    x_hist, y_hist = [], []
    
    for k in range(len(t)):
        # 参考轨迹生成
        xi_ref = np.zeros((config.N, 5))
        for i in range(config.N):
            idx = min(k + i, len(t) - 1)
            valid_len = min(idx, len(x_hist))
            
            # 安全积分计算
            if valid_len > 0:
                ref_slice_x = x_ref[:valid_len]
                hist_slice_x = x_hist[:valid_len]
                I_e_x = np.trapezoid(ref_slice_x - hist_slice_x, dx=config.Ts)
                
                ref_slice_y = y_ref[:valid_len]
                hist_slice_y = y_hist[:valid_len]
                I_e_y = np.trapezoid(ref_slice_y - hist_slice_y, dx=config.Ts)
            else:
                I_e_x = I_e_y = 0.0
            
            xi_ref[i, 4] = I_e_x + I_e_y
        
        # 求解控制量
        u_opt = mpc.solve(xi, xi_ref)
        
        # 状态更新
        xi = model.predict(xi, u_opt, ddx_ref[k], ddy_ref[k])
        xi[4] += config.Ts * (xi[0] + xi[2])
        
        # 记录轨迹
        x_hist.append(x_ref[k] + xi[0])
        y_hist.append(y_ref[k] + xi[2])
    
    return x_hist, y_hist

# ================== 执行与可视化 ==================
if __name__ == "__main__":
    x_traj, y_traj = run_simulation()
    
    plt.figure(figsize=(12, 8))
    plt.plot(x_ref, y_ref, 'r--', label='Reference')
    plt.plot(x_traj, y_traj, 'b-', label='MPC Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Enhanced MPC Trajectory Tracking')
    plt.legend()
    plt.grid(True)
    plt.show()
