import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 原始数据
data = [0.08094336837530136, 0.12438517063856125, 0.12608057260513306, 0.10877136886119843, 0.09932462871074677, 0.09995779395103455, 0.09062739461660385, 0.08403785526752472, 0.07875233888626099, 0.0901733785867691, 0.06277641654014587, 0.07050827145576477, 0.06395956128835678, 0.06721535325050354, 0.07643276453018188, 0.07405060529708862, 0.06631718575954437, 0.07351143658161163, 0.07177743315696716, 0.06825192272663116, 0.06901618838310242, 0.064633809030056, 0.07467871159315109, 0.05871152877807617, 0.06832051277160645, 0.06716254353523254, 0.06484197825193405, 0.07073399424552917, 0.05728663504123688, 0.06616344302892685, 0.06529663503170013, 0.06861187517642975, 0.06889680027961731, 0.06418157368898392, 0.07132333517074585, 0.06243663281202316, 0.06778975576162338, 0.0625293180346489, 0.061135634779930115, 0.07090194523334503, 0.06486787647008896, 0.061263032257556915, 0.06979192793369293, 0.061911024153232574, 0.05763857811689377, 0.061356015503406525, 0.06736651062965393, 0.05529142916202545, 0.06740020215511322, 0.07439769804477692, 0.0595957487821579]
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