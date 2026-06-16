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
print("巴勒斯坦-以色列冲突 - 时空 Hawkes 自激发系数热力图")
print("时间核: β^(t-s-1) (无限记忆递推)")
print("空间核: exp(-d/(2σ²)) (指数核，哈弗辛距离 km)")
print("="*70)

output_dir = './S-T_output'
os.makedirs(output_dir, exist_ok=True)

# ========== 参数设置 ==========
ALPHA_MAX = 0.8
SIGMA_MIN = 10.0
SIGMA_MAX = 1000.0
EARTH_RADIUS = 6371

# ========== 辅助函数 ==========
def point_in_polygon(lon, lat, poly_lons, poly_lats):
    """射线法判断点是否在多边形内"""
    if lon < min(poly_lons) or lon > max(poly_lons) or lat < min(poly_lats) or lat > max(poly_lats):
        return False
    inside = False
    n = len(poly_lons)
    for i in range(n):
        x1, y1 = poly_lons[i], poly_lats[i]
        x2, y2 = poly_lons[(i+1) % n], poly_lats[(i+1) % n]
        if ((y1 > lat) != (y2 > lat)) and (lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside

def load_shapefile_with_flexible_fields(shp_path, level_name, level=2):
    """灵活加载shapefile，自动识别字段名"""
    try:
        if not os.path.exists(shp_path):
            return [], None

        sf = shapefile.Reader(shp_path)
        field_names = [field[0] for field in sf.fields[1:]]

        name_field = None
        for possible_name in ['NAME_2', 'NAME_1', 'NL_NAME_2', 'VARNAME_2', 'NAME_2_', 'NAME_1_', 'NAME_0']:
            if possible_name in field_names:
                name_field = possible_name
                break

        polygons = []
        for idx, shape in enumerate(sf.shapes()):
            parts_coords = []
            parts_idx = list(shape.parts) + [len(shape.points)]
            for i in range(len(parts_idx)-1):
                part_lons = [p[0] for p in shape.points[parts_idx[i]:parts_idx[i+1]]]
                part_lats = [p[1] for p in shape.points[parts_idx[i]:parts_idx[i+1]]]
                parts_coords.append((part_lons, part_lats))

            rec = sf.record(idx)
            if name_field:
                name = rec[field_names.index(name_field)]
            else:
                name = f"{level_name}_{idx}"

            all_lons = []
            all_lats = []
            for plon, plat in parts_coords:
                all_lons.extend(plon)
                all_lats.extend(plat)

            if all_lons and all_lats:
                center_lon = (min(all_lons) + max(all_lons)) / 2
                center_lat = (min(all_lats) + max(all_lats)) / 2
                bbox = (min(all_lons), max(all_lons), min(all_lats), max(all_lats))
            else:
                center_lon, center_lat = 0.0, 0.0
                bbox = (0.0, 0.0, 0.0, 0.0)

            polygons.append({
                'id': idx,
                'name': name,
                'parts': parts_coords,
                'country': level_name,
                'center': (center_lon, center_lat),
                'bbox': bbox
            })

        return polygons, sf
    except Exception as e:
        print(f"警告：无法加载 {shp_path}: {e}")
        return [], None

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

# ========== 1. 加载冲突数据 ==========
df = pd.read_csv('../data/Palestine_Israel.csv')
print(f"数据集列名: {df.columns.tolist()}")

# 适配列名
if 'event_date' not in df.columns:
    for col in ['date', 'event_date', 'date_time', 'Date']:
        if col in df.columns:
            df.rename(columns={col: 'event_date'}, inplace=True)
            break

if 'longitude' not in df.columns:
    for col in ['longitude', 'lon', 'Longitude', 'Lon', 'X']:
        if col in df.columns:
            df.rename(columns={col: 'longitude'}, inplace=True)
            break

if 'latitude' not in df.columns:
    for col in ['latitude', 'lat', 'Latitude', 'Lat', 'Y']:
        if col in df.columns:
            df.rename(columns={col: 'latitude'}, inplace=True)
            break

print(f"使用列: 日期={df['event_date'].name}, 经度={df['longitude'].name}, 纬度={df['latitude'].name}")

# 转换日期
df['event_date'] = pd.to_datetime(df['event_date'])
df = df.sort_values('event_date').reset_index(drop=True)
start_date = df['event_date'].min()
df['t'] = (df['event_date'] - start_date).dt.days
T = df['t'].max() + 1

# ✅ 修复：保存原始长度
original_len = len(df)

print(f"时间跨度: {T} 天, 事件总数: {original_len}")

# 设置经纬度范围
lon_min, lon_max = 30, 65  # 经度范围
lat_min, lat_max = 10, 40.9  # 纬度范围

print(f"经纬度范围: 经度 [{lon_min:.2f}, {lon_max:.2f}], 纬度 [{lat_min:.2f}, {lat_max:.2f}]")

# ========== 2. 加载行政区划 ==========
countries_config = [
    ('gadm41_IRN_shp', 'gadm41_IRN_shp/gadm41_IRN_2.shp', 'Palestine'),
    ('gadm41_ISR_shp', 'gadm41_ISR_shp/gadm41_ISR_1.shp', 'Israel'),  # ✅ 改为 _1.shp
    ('gadm41_LBN_shp', 'gadm41_LBN_shp/gadm41_LBN_2.shp', 'Lebanon'),
    ('gadm41_PSE_shp', 'gadm41_PSE_shp/gadm41_PSE_2.shp', 'Palestine'),
    ('gadm41_SYR_shp', 'gadm41_SYR_shp/gadm41_SYR_2.shp', 'Syria'),
    ('gadm41_YEM_shp', 'gadm41_YEM_shp/gadm41_YEM_2.shp', 'Yemen')
]

all_districts = []
for country_dir, shp_path, country_name in countries_config:
    print(f"尝试加载 {country_name} 行政区划...")
    if os.path.exists(shp_path):
        districts, _ = load_shapefile_with_flexible_fields(shp_path, country_name, level=2)
        if districts:
            print(f"✓ 成功加载 {country_name}: {len(districts)} 个区级单元")
            all_districts.extend(districts)
        else:
            print(f"✗ {country_name} 加载失败（无数据）")
    else:
        print(f"✗ 文件不存在: {shp_path}")

print(f"\n总共加载了 {len(all_districts)} 个区级单元")
n_districts = len(all_districts)

if n_districts == 0:
    raise ValueError("没有加载到任何行政区划数据！")

# ========== 3. 匹配事件到行政区 ==========
print("\n正在将事件匹配到行政区...")
df['district_id'] = -1

district_bbox = []
for d in all_districts:
    district_bbox.append(d['bbox'])

matched_count = 0
for i, row in df.iterrows():
    lon, lat = row['longitude'], row['latitude']
    for did, d in enumerate(all_districts):
        bminx, bmaxx, bminy, bmaxy = district_bbox[did]
        if not (bminx <= lon <= bmaxx and bminy <= lat <= bmaxy):
            continue
        matched = False
        for plon, plat in d['parts']:
            if point_in_polygon(lon, lat, plon, plat):
                df.at[i, 'district_id'] = did
                matched = True
                matched_count += 1
                break
        if matched:
            break

# ✅ 修复：这里 original_len 已定义
df = df[df['district_id'] >= 0].copy()
print(f"成功匹配: {len(df)}/{original_len} ({len(df)/original_len*100:.1f}%)")

if len(df) == 0:
    raise ValueError("没有事件匹配到任何行政区！")

# ========== 4. 构建时间序列 ==========
Y_district = np.zeros((T, n_districts), dtype=np.float64)
for _, row in df.iterrows():
    Y_district[int(row['t']), int(row['district_id'])] += 1

active_districts = np.where(Y_district.sum(axis=0) > 0)[0]
print(f"有事件的区: {len(active_districts)}/{n_districts}")

# ========== 5. 预计算距离矩阵 (哈弗辛距离) ==========
centers = np.array([d['center'] for d in all_districts])
dist_matrix = haversine_distance_matrix(centers)
print(f"哈弗辛距离矩阵计算完成，范围: [{dist_matrix.min():.1f}, {dist_matrix.max():.1f}] km")

# ========== 6. 时空 Hawkes 拟合函数 ==========
def neg_log_lik_spatiotemporal(params, Y_target, Y_district, dist_row):
    mu0, alpha, beta, sigma = params

    if mu0 > 10: return 1e10
    if alpha < 0 or alpha > ALPHA_MAX: return 1e10
    if beta < 0 or beta >= 1: return 1e10
    if sigma < SIGMA_MIN or sigma > SIGMA_MAX: return 1e10

    lam0 = np.exp(mu0)

    spatial_row = np.exp(-dist_row / (2 * sigma**2))
    sum_w = spatial_row.sum()
    if sum_w > 0:
        spatial_row = spatial_row / sum_w
    else:
        spatial_row = np.zeros_like(spatial_row)

    S_total = Y_district @ spatial_row
    S_shifted = np.zeros_like(S_total)
    S_shifted[1:] = S_total[:-1]

    T_len = len(Y_target)
    trigger = np.zeros(T_len)
    for t in range(1, T_len):
        trigger[t] = beta * trigger[t-1] + S_shifted[t]

    lam = lam0 + alpha * trigger
    lam = np.clip(lam, 1e-8, None)

    log_lik = np.sum(Y_target * np.log(lam) - lam - gammaln(Y_target + 1))
    return -log_lik

def fit_region_spatiotemporal(rid, Y_district, dist_matrix):
    """拟合单个行政区（时空 Hawkes）并计算相关性"""
    target_series = Y_district[:, rid]
    if target_series.sum() == 0:
        return None

    init_sigma = 150.0
    dist_row = dist_matrix[rid]

    try:
        init_params = [np.log(max(np.mean(target_series), 0.01)), 0.05, 0.7, init_sigma]
        bounds = [(-10, 10), (0, ALPHA_MAX), (0.01, 0.99), (SIGMA_MIN, SIGMA_MAX)]

        result = minimize(neg_log_lik_spatiotemporal, x0=init_params,
                          args=(target_series, Y_district, dist_row),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 150, 'ftol': 1e-6})

        if not result.success:
            return None
        mu0, alpha, beta, sigma_opt = result.x
        ll_full = -result.fun
        
        # --- 计算 Lambda 与 Y 的皮尔逊相关系数 ---
        lam0 = np.exp(mu0)
        spatial_row = np.exp(-dist_row / (2 * sigma_opt**2))
        sum_w = spatial_row.sum()
        if sum_w > 0: spatial_row = spatial_row / sum_w
        else: spatial_row = np.zeros_like(spatial_row)
        
        S_total = Y_district @ spatial_row
        S_shifted = np.zeros_like(S_total)
        S_shifted[1:] = S_total[:-1]
        
        T_len = len(target_series)
        trigger = np.zeros(T_len)
        for t in range(1, T_len):
            trigger[t] = beta * trigger[t-1] + S_shifted[t]
        lam = lam0 + alpha * trigger
        
        if np.std(lam) > 1e-8 and np.std(target_series) > 1e-8:
            corr = np.corrcoef(target_series, lam)[0, 1]
        else:
            corr = 0.0
            
    except Exception as e:
        return None

    def neg_log_lik_null(params):
        mu0 = params[0]
        if mu0 > 10: return 1e10
        lam0 = np.exp(mu0)
        log_lik = np.sum(target_series * np.log(lam0) - lam0 - gammaln(target_series + 1))
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

# ========== 7. 执行拟合 ==========
print(f"\n开始拟合 {len(active_districts)} 个有事件的区（时空 Hawkes）...")
print("时间核: β^(t-s-1) | 空间核: exp(-d/(2σ²)) | 距离: 哈弗辛距离")

results = []
for i, rid in enumerate(active_districts):
    res = fit_region_spatiotemporal(rid, Y_district, dist_matrix)
    if res:
        results.append(res)
    if (i+1) % 10 == 0:
        print(f"  已完成 {i+1}/{len(active_districts)}，成功 {len(results)} 个")

print(f"成功拟合: {len(results)} 个区")

# 构建结果数组
alpha_by_district = np.full(n_districts, np.nan)
beta_by_district = np.full(n_districts, np.nan)
sigma_by_district = np.full(n_districts, np.nan)
corr_by_district = np.full(n_districts, np.nan)
p_by_district = np.full(n_districts, 1.0)

for res in results:
    alpha_by_district[res['rid']] = res['alpha']
    beta_by_district[res['rid']] = res['beta']
    sigma_by_district[res['rid']] = res['sigma']
    corr_by_district[res['rid']] = res['corr']
    p_by_district[res['rid']] = res['p_value']

# ========== 8. 预加载国家边界（避免画图时重复加载4次） ==========
country_shp_paths = [
    ('gadm41_IRN_shp/gadm41_IRN_0.shp', 'Palestine'),
    ('gadm41_ISR_shp/gadm41_ISR_0.shp', 'Israel'),
    ('gadm41_LBN_shp/gadm41_LBN_0.shp', 'Lebanon'),
    ('gadm41_PSE_shp/gadm41_PSE_0.shp', 'Palestine'),
    ('gadm41_SYR_shp/gadm41_SYR_0.shp', 'Syria'),
    ('gadm41_YEM_shp/gadm41_YEM_0.shp', 'Yemen')
]
country_borders = []
print("\n预加载国家边界用于绘图...")
for shp_path, country_name in country_shp_paths:
    if os.path.exists(shp_path):
        try:
            sf0 = shapefile.Reader(shp_path)
            for shape in sf0.shapes():
                parts_idx = list(shape.parts) + [len(shape.points)]
                for i in range(len(parts_idx)-1):
                    lons = [p[0] for p in shape.points[parts_idx[i]:parts_idx[i+1]]]
                    lats = [p[1] for p in shape.points[parts_idx[i]:parts_idx[i+1]]]
                    country_borders.append((lons, lats))
            print(f"  ✓ {country_name} 边界已加载")
        except Exception as e:
            print(f"  ✗ {country_name} 边界加载失败: {e}")

# ========== 9. 封装可视化绘图函数 ==========
def plot_heatmap(data_array, title, color_list, vmin, vmax, cbar_label, filename):
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), facecolor='white')
    ax.set_facecolor('#def1fe')

    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', color_list, N=256)
    norm = Normalize(vmin=vmin, vmax=vmax)

    ax.add_patch(Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                           facecolor='#fbfafb', edgecolor='none', zorder=0))

    # 绘制区级热力填充
    for did, district in enumerate(all_districts):
        val = data_array[did] if did < len(data_array) else np.nan
        if np.isnan(val):
            facecolor = '#fef7ec'
        else:
            facecolor = custom_cmap(norm(np.clip(val, vmin, vmax)))

        for plon, plat in district['parts']:
            if len(plon) > 2:
                ax.fill(plon, plat, color=facecolor, edgecolor='none', alpha=0.9, zorder=2)
                ax.plot(plon, plat, color='#dfb492', linewidth=0.6, alpha=0.7, zorder=4)

    # 绘制预加载的国家边界
    for lons, lats in country_borders:
        ax.plot(lons, lats, color='black', linewidth=2.0, linestyle='-', alpha=0.9, zorder=5)

    # 绘制显著性边界
    sig_count = 0
    for rid in active_districts:
        if p_by_district[rid] < 0.05 and not np.isnan(alpha_by_district[rid]):
            for plon, plat in all_districts[rid]['parts']:
                if len(plon) > 2:
                    ax.plot(plon, plat, color='#78aaee', linewidth=1.5,
                            linestyle='dashdot', alpha=0.8, zorder=6)
                    sig_count += 1
                    break

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('经度', fontsize=12)
    ax.set_ylabel('纬度', fontsize=12)

    for lat in range(int(np.floor(lat_min)), int(np.ceil(lat_max)) + 1, 5):
        ax.axhline(y=lat, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    for lon in range(int(np.floor(lon_min)), int(np.ceil(lon_max)) + 1, 5):
        ax.axvline(x=lon, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.1, linestyle=':', zorder=0)

    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=30)
    cbar.set_label(cbar_label, fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    legend_elements = [
        Patch(facecolor='#fef7ec', edgecolor='#dfb492', label='无事件区域'),
        Patch(facecolor='#def1fe', edgecolor='none', label='海域'),
        Patch(facecolor='#fbfafb', edgecolor='none', label='非研究区域'),
        Line2D([0], [0], color='black', linewidth=2.0, label='国家边界'),
        Line2D([0], [0], color='#dfb492', linewidth=0.6, label='区界线'),
        Line2D([0], [0], color='#78aaee', linewidth=1.5,
               linestyle='dashdot', label='显著区域 (p < 0.05)')
    ]

    ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
              framealpha=0.9, edgecolor='black', fancybox=True)

    at = AnchoredText(f"时间跨度: {T} 天\n有事件区: {len(active_districts)}\n显著区: {sig_count}",
                      prop=dict(size=9), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.3")
    ax.add_artist(at)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

# ========== 10. 调用绘图并显示四张图 ==========
print("\n正在生成热力图...")

# 1. Alpha 图 (红色系)
colors_red = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
plot_heatmap(alpha_by_district, 
             '巴勒斯坦-以色列冲突：时空自激发系数 (Alpha) 空间分布\n(红色越深 → 受周围影响越强 | 空间核: exp(-d/(2σ²)))', 
             colors_red, 0, ALPHA_MAX, '自激发系数 α', 'S-T_output/Palestine_Israel_Hawkes_Alpha.png')

# 2. Beta 图 (蓝色系)
colors_blue = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
plot_heatmap(beta_by_district, 
             '巴勒斯坦-以色列冲突：时间记忆衰减系数 (Beta) 空间分布\n(蓝色越深 → 记忆保留越久，衰减越慢)', 
             colors_blue, 0, 1.0, '衰减系数 β', 'S-T_output/Palestine_Israel_Hawkes_Beta.png')

# 3. Sigma 图 (绿色系)
colors_green = ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
sigma_max_plot = min(np.nanmax(sigma_by_district), 300) if not np.isnan(np.nanmax(sigma_by_district)) else SIGMA_MAX
plot_heatmap(sigma_by_district, 
             '巴勒斯坦-以色列冲突：空间带宽系数 (Sigma) 空间分布\n(绿色越深 → 空间波及范围越远, 单位: km)', 
             colors_green, SIGMA_MIN, sigma_max_plot, '带宽 σ (km)', 'S-T_output/Palestine_Israel_Hawkes_Sigma.png')

# 4. Correlation 图 (紫色系)
colors_purple = ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d']
plot_heatmap(corr_by_district, 
             '巴勒斯坦-以色列冲突：模型拟合强度 λ 与实际发生数 Y 皮尔逊相关系数\n(紫色越深 → 模型拟合与真实序列越吻合)', 
             colors_purple, 0, 1.0, 'Pearson r', 'S-T_output/Palestine_Israel_Hawkes_Correlation.png')

# ========== 11. 输出统计信息 ==========
print("\n" + "="*70)
print("统计结果")
print("="*70)
print(f"成功拟合: {len(results)} 个区")
sig_count = sum(1 for res in results if res['significant'])
print(f"显著区域数量 (p<0.05): {sig_count}/{len(results)} ({sig_count/len(results)*100:.1f}%)")
print(f"平均 Alpha: {np.nanmean(alpha_by_district):.4f}")
print(f"平均 Beta: {np.nanmean(beta_by_district):.4f}")
print(f"平均 Sigma: {np.nanmean(sigma_by_district):.1f} km")
print(f"平均 皮尔逊相关性: {np.nanmean(corr_by_district):.4f}")

result_data = []
for res in results:
    did = res['rid']
    result_data.append({
        '国家': all_districts[did]['country'],
        '行政区名称': all_districts[did]['name'],
        'Alpha': res['alpha'],
        'Beta': res['beta'],
        'Sigma_km': res['sigma'],
        'Corr(Y, λ)': res['corr'],
        'p_value': res['p_value'],
        '显著': '✓' if res['significant'] else ''
    })

if result_data:
    result_df = pd.DataFrame(result_data).sort_values('Alpha', ascending=False)
    print("\nAlpha 最高的前10个行政区:")
    print(result_df.head(10).to_string(index=False))

    country_stats = result_df.groupby('国家').agg({
        'Alpha': ['count', 'mean', 'max'],
        'Corr(Y, λ)': ['mean'],
        '显著': lambda x: (x == '✓').sum()
    }).round(4)
    print("\n=== 按国家统计 ===")
    print(country_stats)

    result_df.to_csv('S-T_output/Palestine_Israel_Spatio-temporal_Results.csv', index=False, encoding='utf-8-sig')
    print("\n详细结果已保存为: S-T_output/Palestine_Israel_Spatio-temporal_Results.csv")