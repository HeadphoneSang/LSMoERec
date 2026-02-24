import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 原始数据
data = [0.07965417206287384, 0.1632845401763916, 0.1360042840242386, 0.10515492409467697, 0.10048624873161316, 0.10450183600187302, 0.11328581720590591, 0.1010008156299591, 0.09111810475587845, 0.08597470819950104, 0.08756228536367416, 0.07260246574878693, 0.07524842023849487, 0.06982345879077911, 0.08619961142539978, 0.07721703499555588, 0.07341054081916809, 0.07066486030817032, 0.06316246837377548, 0.07984936237335205, 0.06943264603614807, 0.06938906759023666, 0.0778060257434845, 0.06458279490470886, 0.06783682107925415, 0.07349739968776703, 0.07079089432954788, 0.07568613439798355, 0.0705488994717598, 0.08328356593847275, 0.08851160109043121, 0.08490195125341415, 0.08764263987541199, 0.07753044366836548, 0.08095259964466095, 0.08604003489017487, 0.08752238750457764, 0.09381090104579926, 0.09225640445947647, 0.07569807767868042, 0.07970361411571503, 0.08408595621585846, 0.07981221377849579, 0.08259308338165283, 0.08742226660251617, 0.09309100359678268, 0.09202618896961212, 0.09286625683307648, 0.09397642314434052, 0.08074751496315002, 0.07314662635326385]
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