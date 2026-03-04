import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 原始数据
data = [0.05739152804017067, 0.11231356859207153, 0.12260384857654572, 0.11244191229343414, 0.09079279005527496, 0.09406615793704987, 0.08373397588729858, 0.08086839318275452, 0.0788075178861618, 0.0730326920747757, 0.07758738100528717, 0.07688233256340027, 0.07617346942424774, 0.07275666296482086, 0.05779995769262314, 0.07633092999458313, 0.06942710280418396, 0.0675400048494339, 0.07347936183214188, 0.06934400647878647, 0.07364413142204285, 0.0657622367143631, 0.07762379944324493, 0.07597830891609192, 0.07802975922822952, 0.07047611474990845, 0.06375401467084885, 0.07489439845085144, 0.0720389187335968, 0.07388786226511002, 0.06808657944202423, 0.0725783258676529, 0.07032620161771774, 0.06890357285737991, 0.061908915638923645, 0.0665888637304306, 0.06968425214290619, 0.05801139399409294, 0.0666886419057846, 0.07271669805049896, 0.0650145411491394, 0.06665222346782684, 0.06832212209701538, 0.06114747375249863, 0.0638105720281601, 0.06343704462051392, 0.07151389867067337, 0.06721161305904388, 0.07445476949214935, 0.06372490525245667, 0.04978176951408386]
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