import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 原始数据
data = [0.049623049795627594, 0.09512066841125488, 0.10126751661300659, 0.089586541056633, 0.07831014692783356, 0.07442305982112885, 0.06772354245185852, 0.07494684308767319, 0.0688847154378891, 0.07116088271141052, 0.07067650556564331, 0.06580938398838043, 0.054801151156425476, 0.06388674676418304, 0.06659337878227234, 0.06656873226165771, 0.05926899611949921, 0.07063260674476624, 0.0572952926158905, 0.06307245045900345, 0.06881016492843628, 0.05730108916759491, 0.0621102899312973, 0.058112043887376785, 0.06491939723491669, 0.06621656566858292, 0.06293752789497375, 0.058002546429634094, 0.06336171925067902, 0.060144200921058655, 0.05802318826317787, 0.06018126755952835, 0.05208631604909897, 0.05494976043701172, 0.05818801373243332, 0.05650031194090843, 0.06131990998983383, 0.056562989950180054, 0.056990209966897964, 0.06230000779032707, 0.06324170529842377, 0.05577147752046585, 0.05714786797761917, 0.06097839027643204, 0.06309647858142853, 0.06045302748680115, 0.06688480079174042, 0.06263639777898788, 0.06013137102127075, 0.0558600015938282, 0.04945925250649452]
data_np = np.array(data)

# 步骤2：用IQR法计算极值阈值
Q1 = np.percentile(data_np, 25)  # 第一四分位数
Q3 = np.percentile(data_np, 75)  # 第三四分位数
IQR = Q3 - Q1                    # 四分位距
lower_bound = Q1 - 1.5 * IQR     # 下限（小于此值为极值）
upper_bound = Q3 + 1.5 * IQR     # 上限（大于此值为极值）

# 步骤3：过滤极值
data_filtered = data_np[(data_np >= lower_bound) & (data_np <= upper_bound)]

# 步骤4：绘图（原始数据+去极值后数据对比）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# 子图1：原始数据
ax1.plot(data_np, 'o-', color='#FF6B6B', label='原始数据', markersize=6)
ax1.axhline(y=upper_bound, color='#4ECDC4', linestyle='--', label=f'上限阈值({upper_bound:.3f})')
ax1.axhline(y=lower_bound, color='#45B7D1', linestyle='--', label=f'下限阈值({lower_bound:.3f})')
ax1.set_title('原始数据（含极值）', fontsize=14, fontweight='bold')
ax1.set_ylabel('数值', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(alpha=0.3)

# 子图2：去极值后的数据
ax2.plot(data_filtered, 'o-', color='#96CEB4', label='去极值后数据', markersize=6)
ax2.set_title('去极值后的数据', fontsize=14, fontweight='bold')
ax2.set_xlabel('数据索引', fontsize=12)
ax2.set_ylabel('数值', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(alpha=0.3)

# 调整子图间距
plt.tight_layout()
# 显示图表
plt.show()

# 打印关键信息
print(f"原始数据数量：{len(data_np)}")
print(f"去极值后数据数量：{len(data_filtered)}")
print(f"极值阈值：下限={lower_bound:.3f}，上限={upper_bound:.3f}")
print(f"被剔除的极值：{data_np[(data_np < lower_bound) | (data_np > upper_bound)]}")