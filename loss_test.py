import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 原始数据
data = [0.057596150785684586, 0.005097541026771069, 0.05913544073700905, 0.05971388518810272, 0.004439576994627714, 0.004628846421837807, 0.0037640444934368134, 0.1429443657398224, 0.06041453778743744, 0.0035609733313322067, 0.003478490747511387, 0.05537313222885132, 0.0034090871922671795, 0.05983360856771469, 0.0574994795024395, 0.058099858462810516, 0.05515201762318611, 0.05813039839267731, 0.05886431783437729, 0.05750345066189766, 0.3092748522758484, 0.004400553181767464, 0.0040206266567111015, 0.005687425844371319, 0.05829665809869766, 0.003917123191058636, 0.004444073885679245, 0.004162726923823357, 0.004641673993319273, 0.05492990463972092, 0.0034279865212738514, 0.05727560818195343, 0.0026471447199583054, 0.06930883228778839, 0.06148754060268402, 0.005315789952874184, 0.05734659731388092, 0.06026734039187431, 0.005738887470215559, 0.0027295686304569244, 0.0038343211635947227, 0.0582156702876091, 0.06291962414979935, 0.0011488058371469378, 0.06060066074132919, 0.0005054603097960353, 0.05729864910244942, 0.002720089629292488, 0.004883970133960247, 0.06078476086258888, 0.0016600544331595302]
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