import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 原始数据
data = [0.19011139869689941, 0.21995487809181213, 0.2116008996963501, 0.20335936546325684, 0.18563802540302277, 0.184248149394989, 0.1806657910346985, 0.17047153413295746, 0.1705503761768341, 0.16544455289840698, 0.1593887358903885, 0.16110292077064514, 0.15800538659095764, 0.1560528427362442, 0.1599225103855133, 0.15214133262634277, 0.1609732061624527, 0.15515784919261932, 0.1503400355577469, 0.15552590787410736, 0.14920629560947418, 0.15425659716129303, 0.15268144011497498, 0.1549234390258789, 0.15758740901947021, 0.15712320804595947, 0.1491696685552597, 0.15231868624687195, 0.15456938743591309, 0.15431109070777893, 0.1543445885181427, 0.15438668429851532, 0.15484577417373657, 0.15507739782333374, 0.15745124220848083, 0.15491516888141632, 0.1523597538471222, 0.15303999185562134, 0.15008217096328735, 0.15393030643463135, 0.15255525708198547, 0.15694889426231384, 0.15947920083999634, 0.16928385198116302, 0.161640465259552, 0.154263436794281, 0.16581399738788605, 0.1624535173177719, 0.16876336932182312, 0.14910888671875, 0.1414300799369812]
# 步骤1：转换为numpy数组，方便计算
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