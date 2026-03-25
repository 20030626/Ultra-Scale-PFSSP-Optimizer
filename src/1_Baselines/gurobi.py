import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

# 1. 准备数据 (取前 15 个工件和 15 台机器进行演示)
df = pd.read_csv('../../data/extracted_JSSP_data.csv', header=None)
n_full, m_full = 20, 20 # 建议先跑小规模，1000x1000会内存溢出
p = df.iloc[:n_full, :m_full].values

n = n_full
m = m_full

# 2. 创建模型
model = gp.Model("FlowShop_MILP")

# 3. 定义变量
# x[i, k] = 1 表示工件 i 放在第 k 个顺序
x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
# C[k, j] 表示第 k 个位置的工件在机器 j 的完成时间
c = model.addVars(n, m, vtype=GRB.CONTINUOUS, name="c")

# 4. 目标函数: 最小化最后一个工件在最后一台机器的完成时间
model.setObjective(c[n-1, m-1], GRB.MINIMIZE)

# 5. 约束条件
# (1) 每个工件分配到一个位置，每个位置一个工件
model.addConstrs((x.sum(i, '*') == 1 for i in range(n)), "JobAssign")
model.addConstrs((x.sum('*', k) == 1 for k in range(n)), "PosAssign")

# (2) 计算第一台机器上的完成时间
model.addConstr(c[0, 0] == gp.quicksum(p[i][0] * x[i, 0] for i in range(n)))
for k in range(1, n):
    model.addConstr(c[k, 0] == c[k-1, 0] + gp.quicksum(p[i][0] * x[i, k] for i in range(n)))

# (3) 计算第一个工件在所有机器上的完成时间
for j in range(1, m):
    model.addConstr(c[0, j] == c[0, j-1] + gp.quicksum(p[i][j] * x[i, 0] for i in range(n)))

# (4) 核心流转约束
for k in range(1, n):
    for j in range(1, m):
        # 必须等到前一台机器完工，且等到该机器处理完前一个工件
        model.addConstr(c[k, j] >= c[k, j-1] + gp.quicksum(p[i][j] * x[i, k] for i in range(n)))
        model.addConstr(c[k, j] >= c[k-1, j] + gp.quicksum(p[i][j] * x[i, k] for i in range(n)))

# 6. 求解
#model.Params.TimeLimit = 60 # 限制运行时间
model.optimize()

# 7. 输出结果
if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
    print(f"\n最优完工时间 (Makespan): {model.ObjVal}")
    sequence = []
    for k in range(n):
        for i in range(n):
            if x[i, k].X > 0.5:
                sequence.append(i)
    print(f"最优入场顺序: {sequence}")

#最优完工时间 (Makespan): 3807.7332999972577
#最优入场顺序: [13, 28, 10, 6, 17, 21, 3, 29, 15, 18, 26, 25, 2, 7, 9, 8, 14, 1, 11, 5, 20, 22, 24, 12, 0, 27, 16, 23, 19, 4]