from matplotlib.offsetbox import AnchoredText
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.special import gammaln
import shapefile
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("乌克兰二级行政区 - 普通泊松时空 Hawkes 分析")
print("时间核: β^(t-s-1) (无限记忆递推)")
print("空间核: exp(-d/(2σ²)) (指数核，哈弗辛距离 km)")
print("优化参数: μ₀, α, β, σ")
print("="*70)

output_dir = './S-T_output'
os.makedirs(output_dir, exist_ok=True)

# ========== 参数设置 ==========
ALPHA_MAX = 0.8
SIGMA_MIN = 10.0    # σ 下限 (km)
SIGMA_MAX = 1000.0  # σ 上限 (km)
EARTH_RADIUS = 6371  # 地球半径 (km)

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
Y_raion = np.zeros((T, n_raions), dtype=np.float32)
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

# ========== 8. Hawkes 拟合函数（普通泊松，无零膨胀） ==========


def neg_log_lik_poisson(params, Y_target, Y_raion, dist_row):
    """
    负对数似然（普通泊松 Hawkes，无零膨胀）
    """
    mu0, alpha, beta, sigma = params

    if mu0 > 10:
        return 1e10
    if alpha < 0 or alpha > ALPHA_MAX:
        return 1e10
    if beta < 0 or beta >= 1:
        return 1e10
    if sigma < SIGMA_MIN or sigma > SIGMA_MAX:
        return 1e10

    lam0 = np.exp(mu0)

    # 动态计算空间权重 h(d) = exp(-d / (2*sigma^2))
    spatial_row = np.exp(-dist_row / (2 * sigma**2))
    sum_w = spatial_row.sum()
    if sum_w > 0:
        spatial_row = spatial_row / sum_w
    else:
        spatial_row = np.zeros_like(spatial_row)

    # 动态计算空间影响序列 S_shifted
    S_total = Y_raion @ spatial_row
    S_shifted = np.zeros_like(S_total)
    S_shifted[1:] = S_total[:-1]

    # 递推计算 trigger（无限记忆几何衰减）
    T_len = len(Y_target)
    trigger = np.zeros(T_len)
    for t in range(1, T_len):
        trigger[t] = beta * trigger[t-1] + S_shifted[t]

    lam = lam0 + alpha * trigger
    lam = np.clip(lam, 1e-8, None)

    # 普通泊松对数似然
    log_lik = np.sum(Y_target * np.log(lam) - lam - gammaln(Y_target + 1))
    return -log_lik


def fit_region(rid, Y_raion, dist_matrix):
    """拟合单个行政区（普通泊松 Hawkes）并计算相关性"""
    target_series = Y_raion[:, rid]
    if target_series.sum() == 0:
        return None

    init_sigma = 150.0  # 初始带宽 150 km
    dist_row = dist_matrix[rid]

    try:
        init_params = [
            np.log(max(np.mean(target_series), 0.01)), 0.05, 0.7, init_sigma]
        bounds = [(-10, 10), (0, ALPHA_MAX),
                  (0.01, 0.99), (SIGMA_MIN, SIGMA_MAX)]

        result = minimize(neg_log_lik_poisson, x0=init_params,
                          args=(target_series, Y_raion, dist_row),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 150, 'ftol': 1e-6})

        if not result.success:
            return None
        mu0, alpha, beta, sigma_opt = result.x
        ll_full = -result.fun

        # --- 计算 Lambda 与 Y 的相关系数 ---
        lam0 = np.exp(mu0)
        spatial_row = np.exp(-dist_row / (2 * sigma_opt**2))
        sum_w = spatial_row.sum()
        if sum_w > 0:
            spatial_row = spatial_row / sum_w
        else:
            spatial_row = np.zeros_like(spatial_row)
        S_total = Y_raion @ spatial_row
        S_shifted = np.zeros_like(S_total)
        S_shifted[1:] = S_total[:-1]

        T_len = len(target_series)
        trigger = np.zeros(T_len)
        for t in range(1, T_len):
            trigger[t] = beta * trigger[t-1] + S_shifted[t]
        lam = lam0 + alpha * trigger

        # 皮尔逊相关系数
        if np.std(lam) > 1e-8 and np.std(target_series) > 1e-8:
            corr = np.corrcoef(target_series, lam)[0, 1]
        else:
            corr = 0.0

    except Exception as e:
        return None

    # 零模型（α=0，普通泊松）
    def neg_log_lik_null(params):
        mu0 = params[0]
        if mu0 > 10:
            return 1e10
        lam0 = np.exp(mu0)
        log_lik = np.sum(target_series * np.log(lam0) -
                         lam0 - gammaln(target_series + 1))
        return -log_lik

    try:
        init_null = [np.log(max(np.mean(target_series), 0.01))]
        result_null = minimize(neg_log_lik_null, x0=init_null,
                               method='L-BFGS-B', bounds=[(-10, 10)])
        if not result_null.success:
            return None
        ll_null = -result_null.fun
    except Exception as e:
        return None

    LR = max(2 * (ll_full - ll_null), 0)
    p_value = 0.5 * (1 - chi2.cdf(LR, df=1))

    return {
        'rid': rid,
        'alpha': alpha,
        'beta': beta,
        'sigma': sigma_opt,
        'corr': corr,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


# ========== 9. 执行拟合 ==========
print(f"\n开始拟合 {len(active_raions)} 个区...")
results = []
for i, rid in enumerate(active_raions):
    res = fit_region(rid, Y_raion, dist_matrix)
    if res:
        results.append(res)
    if (i+1) % 10 == 0:
        print(f"  已完成 {i+1}/{len(active_raions)}，成功 {len(results)} 个")

print(f"成功拟合: {len(results)} 个区")

# 构建结果数组
alpha_by_raion = np.full(n_raions, np.nan)
beta_by_raion = np.full(n_raions, np.nan)
sigma_by_raion = np.full(n_raions, np.nan)
corr_by_raion = np.full(n_raions, np.nan)
p_by_raion = np.full(n_raions, 1.0)

for res in results:
    alpha_by_raion[res['rid']] = res['alpha']
    beta_by_raion[res['rid']] = res['beta']
    sigma_by_raion[res['rid']] = res['sigma']
    corr_by_raion[res['rid']] = res['corr']
    p_by_raion[res['rid']] = res['p_value']

# ========== 10. 加载边界 ==========


def load_polygons(shp_path):
    polygons = []
    try:
        sf = shapefile.Reader(shp_path)
        for shape in sf.shapes():
            pts = shape.points
            parts = list(shape.parts) + [len(pts)]
            for i in range(len(parts)-1):
                lons = [pts[j][0] for j in range(parts[i], parts[i+1])]
                lats = [pts[j][1] for j in range(parts[i], parts[i+1])]
                polygons.append((lons, lats))
    except Exception as e:
        print(f"  加载 {shp_path} 失败: {e}")
    return polygons


print("\n加载边界...")
ukraine_polygons = load_polygons(shp_path_adm0)
oblast_polygons = load_polygons(shp_path_adm1)

# ========== 11. 封装可视化绘图函数 ==========


def plot_heatmap(data_array, title, color_list, vmin, vmax, cbar_label, filename):
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), facecolor='white')
    ax.set_facecolor('#def1fe')

    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_cmap', color_list, N=256)
    norm = Normalize(vmin=vmin, vmax=vmax)

    lon_min, lon_max = df['longitude'].min() - 0.5, df['longitude'].max() + 0.5
    lat_min, lat_max = df['latitude'].min() - 0.5, df['latitude'].max() + 0.5

    ax.add_patch(Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                           facecolor='#fbfafb', edgecolor='none', zorder=0))

    for plon, plat in ukraine_polygons:
        ax.fill(plon, plat, color='#fef7ec',
                edgecolor='none', alpha=1.0, zorder=1)

    for rid, raion in enumerate(raions):
        val = data_array[rid]
        if np.isnan(val):
            facecolor = '#fef7ec'
        else:
            facecolor = custom_cmap(norm(np.clip(val, vmin, vmax)))

        for plon, plat in raion['parts']:
            if len(plon) > 2:
                ax.fill(plon, plat, color=facecolor,
                        edgecolor='none', alpha=0.9, zorder=2)
                ax.plot(plon, plat, color='#dfb492',
                        linewidth=0.6, alpha=0.7, zorder=4)

    for plon, plat in oblast_polygons:
        ax.plot(plon, plat, color='black', linewidth=1.2,
                linestyle='-', alpha=0.8, zorder=5)

    for plon, plat in ukraine_polygons:
        ax.plot(plon, plat, color='black', linewidth=2.5,
                linestyle='-', alpha=1.0, zorder=6)

    # 画出显著性边界
    for rid in active_raions:
        if p_by_raion[rid] < 0.05 and not np.isnan(alpha_by_raion[rid]):
            for plon, plat in raions[rid]['parts']:
                if len(plon) > 2:
                    ax.plot(plon, plat, color='#78aaee', linewidth=1.5,
                            linestyle='dashdot', alpha=0.9, zorder=7)
                    break

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('经度', fontsize=12)
    ax.set_ylabel('纬度', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.1, linestyle=':')

    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label(cbar_label, fontsize=12)

    legend_elements = [
        Patch(facecolor='#fef7ec', edgecolor='#dfb492', label='未被轰炸区域'),
        Patch(facecolor='#def1fe', edgecolor='none', label='海域'),
        Patch(facecolor='#fbfafb', edgecolor='none', label='非本国区域'),
        Line2D([0], [0], color='black', linewidth=2.5, label='国家边界'),
        Line2D([0], [0], color='black', linewidth=1.2, label='州边界'),
        Line2D([0], [0], color='#78aaee', linewidth=1.5,
               linestyle='dashdot', label='显著区域 (p<0.05)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()


# ========== 12. 调用绘图并显示四张图 ==========
# 1. Alpha 图 (红色系)
colors_red = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272',
              '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
plot_heatmap(alpha_by_raion,
             f'自激发系数 α 分布 (普通泊松 Hawkes)\n(红色越深 → 受周围影响越大)\n时间核: β^(t-s-1) | 空间核: exp(-d/(2σ²))',
             colors_red, 0, ALPHA_MAX, 'α', 'S-T_output/Ukraine_Hawkes_Alpha.png')

# 2. Beta 图 (蓝色系)
colors_blue = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1',
               '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
plot_heatmap(beta_by_raion,
             f'时间记忆衰减系数 β 分布 (普通泊松 Hawkes)\n(蓝色越深 → 记忆保留越久，衰减越慢)',
             colors_blue, 0, 1.0, 'β', 'S-T_output/Ukraine_Hawkes_Beta.png')

# 3. Sigma 图 (绿色系)
colors_green = ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b',
                '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
# 截断最高值以便更好的显示颜色对比度
sigma_max_plot = min(np.nanmax(sigma_by_raion), 300) if not np.isnan(
    np.nanmax(sigma_by_raion)) else SIGMA_MAX
plot_heatmap(sigma_by_raion,
             f'空间带宽系数 σ 分布 (普通泊松 Hawkes)\n(绿色越深 → 空间影响波及的范围越远, 单位 km)',
             colors_green, SIGMA_MIN, sigma_max_plot, 'σ (km)', 'S-T_output/Ukraine_Hawkes_Sigma.png')

# 4. Correlation 图 (紫色系)
colors_purple = ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc',
                 '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d']
plot_heatmap(corr_by_raion,
             f'模型拟合强度 λ 与实际发生数 Y 随时间皮尔逊相关系数\n(紫色越深 → 模型拟合与真实序列越吻合)',
             colors_purple, 0, 1.0, 'Pearson r', 'S-T_output/Ukraine_Hawkes_Correlation.png')

# ========== 13. 输出结果 ==========
print("\n" + "="*70)
print("统计结果")
print("="*70)
print(f"成功拟合: {len(results)} 个区")
sig_count = sum(1 for res in results if res['significant'])
print(
    f"显著区域数量 (p<0.05): {sig_count}/{len(results)} ({sig_count/len(results)*100:.1f}%)")
print(f"平均 Alpha: {np.nanmean(alpha_by_raion):.4f}")
print(f"平均 Beta: {np.nanmean(beta_by_raion):.4f}")
print(f"平均 Sigma: {np.nanmean(sigma_by_raion):.1f} km")
print(f"平均 皮尔逊相关性: {np.nanmean(corr_by_raion):.4f}")

output_results = []
for res in results:
    output_results.append({
        '区名称': raions[res['rid']]['name_2'],
        '州名称': raions[res['rid']]['name_1'],
        'Alpha': res['alpha'],
        'Beta': res['beta'],
        'Sigma_km': res['sigma'],
        'Corr(Y, λ)': res['corr'],
        'p_value': res['p_value'],
        '显著': '✓' if res['significant'] else ''
    })
df_results = pd.DataFrame(output_results).sort_values('Alpha', ascending=False)
print("\nAlpha 最高的前10个区:")
print(df_results.head(10).to_string(index=False))

df_results.to_csv('S-T_output/Ukraine_Hawkes_Spatio-temporal_Results.csv',
                  index=False, encoding='utf-8-sig')
print("\n结果已保存: S-T_output/Ukraine_Hawkes_Spatio-temporal_Results.csv")
