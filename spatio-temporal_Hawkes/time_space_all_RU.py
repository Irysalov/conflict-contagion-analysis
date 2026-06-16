from scipy.optimize import minimize_scalar
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.special import gammaln
import shapefile
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("乌克兰全局时空 Hawkes 假设检验")
print("H0: α = 0 (无空间自激发) vs H1: α > 0 (存在空间自激发)")
print("="*70)

# ========== 参数设置 ==========
ALPHA_MAX = 0.8
SIGMA_MIN = 10.0
SIGMA_MAX = 1000.0
EARTH_RADIUS = 6371

# ========== 1. 加载数据 ==========
df = pd.read_csv('../data/Russia_Ukraine.csv')
selected_types = ['Shelling/artillery/missile attack', 'Air/drone strike']
df = df[df['sub_event_type'].isin(selected_types)].copy()

df['event_date'] = pd.to_datetime(df['event_date'])
df = df.sort_values('event_date').reset_index(drop=True)
start_date = df['event_date'].min()
df['t'] = (df['event_date'] - start_date).dt.days
T = df['t'].max() + 1
original_len = len(df)

print(f"时间跨度: {T} 天, 事件总数: {original_len}")

# ========== 2. 加载行政区划 ==========
shp_path_adm2 = r'gadm_UKR\gadm41_UKR_2.shp'
shp_path_adm1 = r'gadm_UKR\gadm41_UKR_1.shp'
shp_path_adm0 = r'gadm_UKR\gadm41_UKR_0.shp'

sf_adm2 = shapefile.Reader(shp_path_adm2)
print(f"加载区级边界，共 {len(sf_adm2.shapes())} 个区")

field_names = [field[0] for field in sf_adm2.fields[1:]]
raion_field = 'NAME_2' if 'NAME_2' in field_names else field_names[0]
oblast_field = 'NAME_1' if 'NAME_1' in field_names else None
print(f"使用区名称字段: {raion_field}")

# ========== 3. 提取行政区 ==========
raions = []
for idx, shape in enumerate(sf_adm2.shapes()):
    parts_coords = []
    parts_idx = list(shape.parts) + [len(shape.points)]
    for i in range(len(parts_idx)-1):
        lons = []
        lats = []
        for j in range(parts_idx[i], parts_idx[i+1]):
            lons.append(shape.points[j][0])
            lats.append(shape.points[j][1])
        parts_coords.append((lons, lats))

    rec = sf_adm2.record(idx)
    name_2 = rec[field_names.index(
        raion_field)] if raion_field else f"Raion_{idx}"
    name_1 = rec[field_names.index(oblast_field)] if oblast_field else ""

    all_lons = []
    all_lats = []
    for plon, plat in parts_coords:
        all_lons.extend(plon)
        all_lats.extend(plat)

    if all_lons:
        center_lon = (min(all_lons) + max(all_lons)) / 2
        center_lat = (min(all_lats) + max(all_lats)) / 2
        bbox = (min(all_lons), max(all_lons), min(all_lats), max(all_lats))
    else:
        center_lon, center_lat = 0.0, 0.0
        bbox = (0.0, 0.0, 0.0, 0.0)

    raions.append({
        'id': idx,
        'name_2': name_2,
        'name_1': name_1,
        'parts': parts_coords,
        'center': (center_lon, center_lat),
        'bbox': bbox
    })

n_raions = len(raions)
print(f"共 {n_raions} 个区")

# 预计算中心点矩阵
centers = np.array([r['center'] for r in raions])

# ========== 4. 点匹配函数 ==========


def point_in_polygon(lon, lat, poly_lons, poly_lats):
    if lon < min(poly_lons) or lon > max(poly_lons) or lat < min(poly_lats) or lat > max(poly_lats):
        return False
    inside = False
    n = len(poly_lons)
    for i in range(n):
        x1, y1 = poly_lons[i], poly_lats[i]
        x2, y2 = poly_lons[(i+1) % n], poly_lats[(i+1) % n]
        if ((y1 > lat) != (y2 > lat)) and (lon < (x2 - x1)*(lat - y1)/(y2 - y1) + x1):
            inside = not inside
    return inside


# ========== 5. 匹配事件到区 ==========
print("\n匹配事件到行政区...")
df['raion_id'] = -1

for i, row in df.iterrows():
    lon, lat = row['longitude'], row['latitude']
    for rid, r in enumerate(raions):
        bminx, bmaxx, bminy, bmaxy = r['bbox']
        if not (bminx <= lon <= bmaxx and bminy <= lat <= bmaxy):
            continue
        matched = False
        for plon, plat in r['parts']:
            if point_in_polygon(lon, lat, plon, plat):
                df.at[i, 'raion_id'] = rid
                matched = True
                break
        if matched:
            break
    if (i+1) % 20000 == 0:
        print(f"  已处理 {i+1}/{original_len}")

df = df[df['raion_id'] >= 0].copy()
print(f"成功匹配: {len(df)}/{original_len} ({len(df)/original_len*100:.1f}%)")

if len(df) == 0:
    raise ValueError("没有事件匹配到任何行政区！")

# ========== 6. 构建时间序列 ==========
Y_raion = np.zeros((T, n_raions), dtype=np.float64)
for _, row in df.iterrows():
    Y_raion[int(row['t']), int(row['raion_id'])] += 1

active_raions = np.where(Y_raion.sum(axis=0) > 0)[0]
print(f"有事件的区: {len(active_raions)}/{n_raions}")

# ========== 7. 预计算距离矩阵 (哈弗辛距离) ==========


def haversine_distance_matrix(centers, R=EARTH_RADIUS):
    """向量化计算哈弗辛距离矩阵 (单位: km)"""
    lon_rad = np.radians(centers[:, 0])
    lat_rad = np.radians(centers[:, 1])
    dlon = lon_rad[:, np.newaxis] - lon_rad[np.newaxis, :]
    dlat = lat_rad[:, np.newaxis] - lat_rad[np.newaxis, :]
    a = np.sin(dlat/2)**2 + np.cos(lat_rad[:, np.newaxis]) * \
        np.cos(lat_rad[np.newaxis, :]) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


dist_matrix = haversine_distance_matrix(centers)
print(f"哈弗辛距离矩阵计算完成，范围: [{dist_matrix.min():.1f}, {dist_matrix.max():.1f}] km")

# ========== 8. 全局时空 Hawkes 负对数似然 ==========


def neg_log_lik_national(params, Y_matrix, dist_matrix):
    mu0, alpha, beta, sigma = params

    if mu0 > 10 or mu0 < -10:
        return 1e10
    if alpha < 0 or alpha > ALPHA_MAX:
        return 1e10
    if beta < 0 or beta >= 1:
        return 1e10
    if sigma < SIGMA_MIN or sigma > SIGMA_MAX:
        return 1e10

    lam0 = np.exp(mu0)
    T_len, N_raions = Y_matrix.shape

    # 空间权重矩阵
    W = np.exp(-dist_matrix / (2 * sigma**2))
    sum_W = W.sum(axis=1, keepdims=True)
    W = np.where(sum_W > 0, W / sum_W, 0)

    # 空间影响
    S_total = Y_matrix @ W.T
    S_shifted = np.zeros_like(S_total)
    S_shifted[1:, :] = S_total[:-1, :]

    # 时间递推
    trigger = np.zeros_like(Y_matrix)
    for t in range(1, T_len):
        trigger[t, :] = beta * trigger[t-1, :] + S_shifted[t, :]

    lam = lam0 + alpha * trigger
    lam = np.clip(lam, 1e-8, None)

    log_lik_matrix = Y_matrix * np.log(lam) - lam - gammaln(Y_matrix + 1)
    total_log_lik = np.sum(log_lik_matrix)

    return -total_log_lik


# ========== 9. 全局拟合 ==========
print("\n" + "="*70)
print("开始拟合全局时空 Hawkes 模型...")
print("="*70)

mean_daily_events = np.mean(Y_raion)
init_mu0 = np.log(max(mean_daily_events, 0.01))
init_params = [init_mu0, 0.1, 0.5, 150.0]

bounds = [
    (-10, 10),              # mu0
    (0, ALPHA_MAX),         # alpha
    (0.01, 0.99),           # beta
    (SIGMA_MIN, SIGMA_MAX)  # sigma (km)
]

print(
    f"初始参数: mu0={init_params[0]:.3f}, alpha={init_params[1]}, beta={init_params[2]}, sigma={init_params[3]:.0f}km")

try:
    result_nat = minimize(
        neg_log_lik_national,
        x0=init_params,
        args=(Y_raion, dist_matrix),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 300, 'ftol': 1e-6, 'disp': True}
    )

    if result_nat.success:
        mu0_opt, alpha_opt, beta_opt, sigma_opt = result_nat.x
        ll_full = -result_nat.fun

        print("\n✅ 全局拟合成功！最优参数：")
        print(
            f"  μ₀ (对数背景率) = {mu0_opt:.4f} → 背景率 = {np.exp(mu0_opt):.6f} 次/天/区")
        print(f"  α (自激发系数)  = {alpha_opt:.4f}")
        print(f"  β (时间衰减)    = {beta_opt:.4f}")
        print(f"  σ (空间带宽)    = {sigma_opt:.2f} km")
        print(f"  全局对数似然    = {ll_full:.2f}")
    else:
        print("\n❌ 优化未成功收敛:", result_nat.message)
        exit()
except Exception as e:
    print(f"拟合报错: {e}")
    exit()

# ========== 10. 零模型（独立泊松） ==========


def neg_log_lik_null(mu0):
    if mu0 > 10 or mu0 < -10:
        return 1e10
    lam0 = np.exp(mu0)
    log_lik = np.sum(Y_raion * np.log(lam0) - lam0 - gammaln(Y_raion + 1))
    return -log_lik


res_null = minimize_scalar(
    neg_log_lik_null, bounds=(-10, 10), method='bounded')
mu0_null = res_null.x
ll_null = -res_null.fun

print(f"\n零模型 (α=0):")
print(f"  μ₀ = {mu0_null:.4f} → 背景率 = {np.exp(mu0_null):.6f} 次/天/区")
print(f"  对数似然 = {ll_null:.2f}")

# ========== 11. 似然比检验 ==========
LR = 2 * (ll_full - ll_null)
# 注意：原假设 α=0，β 和 σ 在 H0 下不可识别，有效自由度 = 1
p_value = 0.5 * (1 - chi2.cdf(LR, df=1))

print("\n" + "="*70)
print("似然比检验结果")
print("="*70)
print(f"似然比统计量 LR = {LR:.4f}")
print(f"p-value = {p_value:.6e}")

if p_value < 0.001:
    print("\n✓ 在 0.001 显著性水平下拒绝 H₀")
    print("  乌克兰战区存在极显著的空间自激发/传染效应！")
elif p_value < 0.05:
    print("\n✓ 在 0.05 显著性水平下拒绝 H₀")
    print("  乌克兰战区存在显著的空间自激发效应")
else:
    print("\n✗ 不拒绝 H₀")
    print("  无显著的空间自激发效应")

stability = alpha_opt / (1 - beta_opt) if beta_opt < 1 else np.inf
print(
    f"\n稳定性条件: α/(1-β) = {stability:.4f} {'< 1 ✓' if stability < 1 else '≥ 1 ✗'}")

# ========== 12. 计算精确的全局条件强度并可视化 ==========
print("\n正在根据最优参数计算精确的每日条件强度 λ_t...")

# 1. 重新构建最优的空间权重矩阵 W
lam0_opt = np.exp(mu0_opt)
W_opt = np.exp(-dist_matrix / (2 * sigma_opt**2))
sum_W_opt = W_opt.sum(axis=1, keepdims=True)
W_opt = np.where(sum_W_opt > 0, W_opt / sum_W_opt, 0)

# 2. 重新计算精确的时空自激发传播量
S_total_opt = Y_raion @ W_opt.T
S_shifted_opt = np.zeros_like(S_total_opt)
S_shifted_opt[1:, :] = S_total_opt[:-1, :]

trigger_opt = np.zeros_like(Y_raion)
for t in range(1, T):
    trigger_opt[t, :] = beta_opt * trigger_opt[t-1, :] + S_shifted_opt[t, :]

# 3. 计算每个区在每天的条件强度 (T x N_raions)
lam_matrix_opt = lam0_opt + alpha_opt * trigger_opt

# 4. 在空间上求和，得到每日全局的总条件强度 λ_t
lam_fitted = lam_matrix_opt.sum(axis=1)

# 开始画图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

Y_total = Y_raion.sum(axis=1)

# 左图：总轰炸次数时间序列
ax1 = axes[0]
ax1.plot(Y_total, alpha=0.7, color='steelblue', linewidth=0.8, label='每日实际轰炸次数')
ax1.set_xlabel('天数')
ax1.set_ylabel('每日总轰炸次数')
ax1.set_title(f'乌克兰每日总轰炸次数\n(总计 {Y_total.sum():.0f} 次)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 右图：精确拟合 vs 实际对比
ax2 = axes[1]
ax2.plot(Y_total, alpha=0.4, color='gray', label='实际值', linewidth=0.8)
ax2.plot(lam_fitted, 'r-', label='拟合全局强度 $\lambda_t$', linewidth=1.2)
ax2.set_xlabel('天数')
ax2.set_ylabel('轰炸次数 / 条件强度')
ax2.set_title(
    f'全局 Hawkes 精确拟合对比\n(α={alpha_opt:.3f}, β={beta_opt:.3f}, σ={sigma_opt:.1f} km)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('S-T_output/Ukraine_Global_Hawkes_Test.png', dpi=150)
plt.show()

print("图片已成功更新并保存: S-T_output/Ukraine_Global_Hawkes_Test.png")

# ========== 13. 最终结论 ==========
print("\n" + "="*70)
print("最终结论")
print("="*70)
print(f"全局自激发系数 α = {alpha_opt:.4f}")
print(f"似然比 LR = {LR:.4f}")
print(f"p-value = {p_value:.6e}")
if p_value < 0.05:
    print("\n结论：乌克兰战区的袭击事件具有显著的时空传染/自激特征。")
    print("       一次袭击会显著提高周围区域在未来短期内的袭击概率。")
else:
    print("\n结论：乌克兰战区的袭击事件未检测到显著的时空传染效应。")
