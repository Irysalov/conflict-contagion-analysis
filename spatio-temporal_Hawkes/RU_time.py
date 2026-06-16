from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.special import gammaln
import shapefile
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("乌克兰 Hawkes 自激发系数热力图 (按区级行政区划)")
print("="*70)

output_dir = './S-T_output'
os.makedirs(output_dir, exist_ok=True)

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


# ========== 1. 加载轰炸数据 ==========
df = pd.read_csv('../data/Russia_Ukraine.csv')
selected_types = ['Shelling/artillery/missile attack', 'Air/drone strike']
df = df[df['sub_event_type'].isin(selected_types)].copy()

df['event_date'] = pd.to_datetime(df['event_date'])
df = df.sort_values('event_date').reset_index(drop=True)
start_date = df['event_date'].min()
df['t'] = (df['event_date'] - start_date).dt.days
T = df['t'].max() + 1

print(f"时间跨度: {T} 天, 事件总数: {len(df)}")

# ========== 2. 加载乌克兰区级行政区划 (ADM2) ==========
shp_path_adm2 = r'gadm_UKR\gadm41_UKR_2.shp'

try:
    sf_adm2 = shapefile.Reader(shp_path_adm2)
    print(f"成功加载区级边界文件，共 {len(sf_adm2.shapes())} 个区")

    # 打印字段信息以便调试
    print("\n字段列表:")
    for i, field in enumerate(sf_adm2.fields):
        print(f"  {i}: {field}")

except Exception as e:
    raise FileNotFoundError(
        f"未找到区级边界文件 {shp_path_adm2}，请先下载 GADM level-2 数据。错误: {e}")

# 获取字段名列表
field_names = [field[0] for field in sf_adm2.fields[1:]]  # 跳过第一个删除标记字段
print(f"\n字段名: {field_names}")

# 智能查找区名称和州名称字段


def find_field_name(field_names, possible_names):
    """在字段名列表中查找可能的字段名"""
    for name in possible_names:
        if name in field_names:
            return name
    return None


# 查找区名称字段（可能的名称）
raion_name_fields = ['NAME_2', 'NL_NAME_2', 'VARNAME_2', 'NAME_2_']
raion_field = find_field_name(field_names, raion_name_fields)

# 查找州名称字段
oblast_name_fields = ['NAME_1', 'NL_NAME_1', 'VARNAME_1', 'NAME_1_']
oblast_field = find_field_name(field_names, oblast_name_fields)

print(f"使用区名称字段: {raion_field}")
print(f"使用州名称字段: {oblast_field}")

# 为每个区分配唯一 ID，并提取多边形
raions = []
for idx, shape in enumerate(sf_adm2.shapes()):
    # 合并所有 parts 的坐标
    parts_coords = []
    parts_idx = list(shape.parts) + [len(shape.points)]
    for i in range(len(parts_idx)-1):
        part_lons = [p[0] for p in shape.points[parts_idx[i]:parts_idx[i+1]]]
        part_lats = [p[1] for p in shape.points[parts_idx[i]:parts_idx[i+1]]]
        parts_coords.append((part_lons, part_lats))

    # 读取区名称和州名称
    rec = sf_adm2.record(idx)

    if raion_field:
        name_2 = rec[field_names.index(raion_field)]
    else:
        name_2 = f"Raion_{idx}"

    if oblast_field:
        name_1 = rec[field_names.index(oblast_field)]
    else:
        name_1 = ""

    raions.append({
        'id': idx,
        'name_2': name_2,
        'name_1': name_1,
        'parts': parts_coords
    })

n_raions = len(raions)
print(f"\n共 {n_raions} 个区级单元")
print(f"示例区: {raions[0]['name_2']} (州: {raions[0]['name_1']})")

# ========== 3. 将每个轰炸事件分配到所属的区 ==========
print("\n正在将事件匹配到行政区 (可能需要几十秒)...")
df['raion_id'] = -1

# 预先计算每个区的 bbox
raion_bbox = []
for r in raions:
    all_lons = []
    all_lats = []
    for (plon, plat) in r['parts']:
        all_lons.extend(plon)
        all_lats.extend(plat)
    if all_lons and all_lats:  # 确保有数据
        minx, maxx = min(all_lons), max(all_lons)
        miny, maxy = min(all_lats), max(all_lats)
        raion_bbox.append((minx, maxx, miny, maxy))
    else:
        raion_bbox.append((None, None, None, None))

matched_count = 0
for i, row in df.iterrows():
    lon, lat = row['longitude'], row['latitude']
    matched = False
    for rid, r in enumerate(raions):
        bminx, bmaxx, bminy, bmaxy = raion_bbox[rid]
        if bminx is None:  # 跳过无效多边形
            continue
        # 先用 bbox 快速排除
        if not (bminx <= lon <= bmaxx and bminy <= lat <= bmaxy):
            continue
        # 精确匹配
        for (plon, plat) in r['parts']:
            if point_in_polygon(lon, lat, plon, plat):
                df.at[i, 'raion_id'] = rid
                matched = True
                matched_count += 1
                break
        if matched:
            break

    # 进度提示
    if (i + 1) % 5000 == 0:
        print(f"  已处理 {i+1}/{len(df)} 个事件，匹配成功 {matched_count} 个")

# 保留有匹配到区的记录
original_len = len(df)
df = df[df['raion_id'] >= 0].copy()
print(
    f"\n成功匹配到区的记录数: {len(df)} / {original_len} ({len(df)/original_len*100:.1f}%)")

if len(df) == 0:
    raise ValueError("没有事件匹配到任何区！请检查坐标系是否匹配或shapefile是否包含克里米亚等地区")

# ========== 4. 构建时间序列 Y ==========
Y_raion = np.zeros((T, n_raions), dtype=np.float32)
for _, row in df.iterrows():
    t = int(row['t'])
    rid = int(row['raion_id'])
    Y_raion[t, rid] += 1

# 只保留至少发生过一次事件的区
active_raions = np.where(Y_raion.sum(axis=0) > 0)[0]
print(f"发生过事件的区数量: {len(active_raions)} / {n_raions}")

# ========== 5. Hawkes 1D 拟合函数 ==========
def neg_log_lik_1d(params, series):
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta < 0 or beta >= 1:
        return 1e10
    lam = np.zeros_like(series, dtype=np.float64)
    lam[0] = mu
    for t in range(1, len(series)):
        lam[t] = mu + alpha * series[t-1] + beta * (lam[t-1] - mu)
    lam = np.clip(lam, 1e-8, None)
    log_lik = np.sum(series * np.log(lam) - lam - gammaln(series + 1))
    return -log_lik if np.isfinite(log_lik) else 1e10

def fit_hawkes_1d(series):
    # 如果序列全为0，返回nan
    if np.sum(series) == 0:
        return None, None, None, -np.inf

    init_params = [max(np.mean(series)*0.5, 0.01), 0.1, 0.5]
    bounds = [(1e-6, None), (1e-6, None), (1e-6, 0.999)]
    result = minimize(neg_log_lik_1d, x0=init_params, args=(series,),
                      method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-6})
    if result.success or result.status == 1:
        return result.x[0], result.x[1], result.x[2], -result.fun
    return None, None, None, -np.inf

# ========== 6. 对每个有事件的区拟合 Alpha, Beta 及计算相关性 ==========
alpha_by_raion = np.full(n_raions, np.nan)
beta_by_raion = np.full(n_raions, np.nan)
corr_by_raion = np.full(n_raions, np.nan)
p_by_raion = np.full(n_raions, 1.0)
total_events_by_raion = Y_raion.sum(axis=0)

print(f"\n开始拟合 {len(active_raions)} 个有事件的区...")
successful_fits = 0

for j, rid in enumerate(active_raions):
    series = Y_raion[:, rid]
    mu, alpha, beta, ll_full = fit_hawkes_1d(series)

    if mu is not None and alpha is not None:
        # 计算 Lambda 序列以获取相关性
        lam = np.zeros_like(series, dtype=np.float64)
        lam[0] = mu
        for t in range(1, len(series)):
            lam[t] = mu + alpha * series[t-1] + beta * (lam[t-1] - mu)
        
        # 计算皮尔逊相关系数
        if np.std(lam) > 1e-8 and np.std(series) > 1e-8:
            corr = np.corrcoef(series, lam)[0, 1]
        else:
            corr = 0.0

        # 计算零模型以便得到 p-value
        mu0 = np.mean(series) + 1e-8
        ll_null = np.sum(series * np.log(mu0) - mu0 - gammaln(series + 1))
        LR = max(2 * (ll_full - ll_null), 0)
        p_val = 0.5 * (1 - chi2.cdf(LR, df=1))
        
        alpha_by_raion[rid] = alpha
        beta_by_raion[rid] = beta
        corr_by_raion[rid] = corr
        p_by_raion[rid] = max(p_val, 1e-10)
        successful_fits += 1

    if (j+1) % 20 == 0:
        print(f"  已完成 {j+1} / {len(active_raions)} 个区...")

print(f"\n成功拟合的区数量: {successful_fits} / {len(active_raions)}")

# ========== 7. 预加载乌克兰全国与州边界（用于可视化）==========
ukraine_polygons = []
try:
    sf_adm0 = shapefile.Reader(r'gadm_UKR\gadm41_UKR_0.shp')
    for shape in sf_adm0.shapes():
        parts_idx = list(shape.parts) + [len(shape.points)]
        for i in range(len(parts_idx)-1):
            lons = [p[0] for p in shape.points[parts_idx[i]:parts_idx[i+1]]]
            lats = [p[1] for p in shape.points[parts_idx[i]:parts_idx[i+1]]]
            ukraine_polygons.append((lons, lats))
    print("已加载乌克兰国家边界 (ADM0)")
except Exception as e:
    print(f"警告：未找到国家边界文件 {e}")

oblast_polygons = []
try:
    sf_adm1 = shapefile.Reader(r'gadm_UKR\gadm41_UKR_1.shp')
    for shape in sf_adm1.shapes():
        parts_idx = list(shape.parts) + [len(shape.points)]
        for i in range(len(parts_idx)-1):
            lons = [p[0] for p in shape.points[parts_idx[i]:parts_idx[i+1]]]
            lats = [p[1] for p in shape.points[parts_idx[i]:parts_idx[i+1]]]
            oblast_polygons.append((lons, lats))
    print("已加载乌克兰州边界 (ADM1)")
except Exception as e:
    print(f"警告：未叠加州边界: {e}")

# ========== 8. 封装可视化绘图函数 ==========
def plot_heatmap(data_array, title, color_list, vmin, vmax, cbar_label, filename):
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), facecolor='white')
    ax.set_facecolor('#def1fe')

    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', color_list, N=256)
    norm = Normalize(vmin=vmin, vmax=vmax)

    lon_min, lon_max = df['longitude'].min() - 0.5, df['longitude'].max() + 0.5
    lat_min, lat_max = df['latitude'].min() - 0.5, df['latitude'].max() + 0.5

    ax.add_patch(Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                           facecolor='#fbfafb', edgecolor='none', zorder=0))

    for (plon, plat) in ukraine_polygons:
        ax.fill(plon, plat, color='#fef7ec', edgecolor='none', alpha=1.0, zorder=1)

    for rid, raion in enumerate(raions):
        val = data_array[rid]
        if np.isnan(val):
            facecolor = '#fef7ec'
        else:
            facecolor = custom_cmap(norm(np.clip(val, vmin, vmax)))

        for (plon, plat) in raion['parts']:
            if len(plon) > 2:
                ax.fill(plon, plat, color=facecolor, edgecolor='none', alpha=0.9, zorder=2)
                ax.plot(plon, plat, color='#dfb492', linewidth=0.8, alpha=0.8, zorder=4)

    for (plon, plat) in oblast_polygons:
        ax.plot(plon, plat, color='black', linewidth=1.5, linestyle='-', alpha=0.9, zorder=5)

    for (plon, plat) in ukraine_polygons:
        ax.plot(plon, plat, color='black', linewidth=2.5, linestyle='-', alpha=1.0, zorder=6)

    significant_count = 0
    for rid in active_raions:
        if p_by_raion[rid] < 0.05 and not np.isnan(alpha_by_raion[rid]):
            for (plon, plat) in raions[rid]['parts']:
                if len(plon) > 2:
                    ax.plot(plon, plat, color='#78aaee', linewidth=1.8,
                            linestyle='dashdot', alpha=0.9, zorder=7)
                    significant_count += 1
                    break

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('经度', fontsize=12)
    ax.set_ylabel('纬度', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.1, linestyle=':', zorder=0)

    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=30)
    cbar.set_label(cbar_label, fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    legend_elements = [
        Patch(facecolor='#fef7ec', edgecolor='#dfb492', label='未被轰炸区域'),
        Patch(facecolor='#def1fe', edgecolor='none', label='海域'),
        Patch(facecolor='#fbfafb', edgecolor='none', label='非本国区域'),
        Line2D([0], [0], color='black', linewidth=2.5, label='国家边界'),
        Line2D([0], [0], color='black', linewidth=1.5, label='州边界'),
        Line2D([0], [0], color='#dfb492', linewidth=0.8, label='区界线'),
        Line2D([0], [0], color='#78aaee', linewidth=1.8, linestyle='dashdot', label='显著区域 (p < 0.05)')
    ]

    ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
              framealpha=0.9, edgecolor='black', fancybox=True)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

# ========== 9. 生成三张热力图 ==========
print("\n正在生成热力图...")

# 1. Alpha (自激发系数，红色系)
colors_red = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
plot_heatmap(alpha_by_raion, 
             '乌克兰区级自激发系数 (Alpha) 分布\n(红色越深 → 该区轰炸后再次轰炸概率越高)', 
             colors_red, 0, 0.6, '自激发系数 α (Alpha)', 'S-T_output/Ukraine_Hawkes1D_Alpha.png')

# 2. Beta (时间衰减系数，蓝色系)
colors_blue = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
plot_heatmap(beta_by_raion, 
             '乌克兰区级时间衰减系数 (Beta) 分布\n(蓝色越深 → 记忆保留越久，衰减越慢)', 
             colors_blue, 0, 1.0, '衰减系数 β (Beta)', 'S-T_output/Ukraine_Hawkes1D_Beta.png')

# 3. Correlation (相关性，紫色系)
colors_purple = ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d']
plot_heatmap(corr_by_raion, 
             '乌克兰区级模型拟合强度 λ 与实际发生数 Y 皮尔逊相关系数\n(紫色越深 → 模型拟合与真实序列越吻合)', 
             colors_purple, 0, 1.0, 'Pearson r', 'S-T_output/Ukraine_Hawkes1D_Correlation.png')

# ========== 10. 输出统计信息 ==========
print("\n" + "="*70)
print("图片已保存为:")
print("  - S-T_output/Ukraine_Hawkes1D_Alpha.png")
print("  - S-T_output/Ukraine_Hawkes1D_Beta.png")
print("  - S-T_output/Ukraine_Hawkes1D_Correlation.png")

print("\n=== 统计结果 ===")
significant_raions = np.sum((p_by_raion < 0.05) & (~np.isnan(alpha_by_raion)))
print(f"显著区数量 (p<0.05): {significant_raions} / {len(active_raions)}")
print(f"平均 Alpha (所有区): {np.nanmean(alpha_by_raion):.4f}")
print(f"平均 Beta (所有区): {np.nanmean(beta_by_raion):.4f}")
print(f"平均 皮尔逊相关性: {np.nanmean(corr_by_raion):.4f}")

# 输出各区的详细结果
result_df = pd.DataFrame({
    '区ID': range(n_raions),
    '区名称': [r['name_2'] for r in raions],
    '州名称': [r['name_1'] for r in raions],
    'Alpha': alpha_by_raion,
    'Beta': beta_by_raion,
    'Corr(Y, λ)': corr_by_raion,
    'P值': p_by_raion,
    '显著': p_by_raion < 0.05,
    '事件总数': [int(Y_raion[:, i].sum()) for i in range(n_raions)]
})
result_df = result_df.dropna(subset=['Alpha']).sort_values('Alpha', ascending=False)

print("\nAlpha 值最高的前10个区:")
print(result_df.head(10).to_string(index=False))

# 保存详细结果到 CSV
result_df.to_csv('S-T_output/Ukraine_Hawkes1D_Results.csv', index=False, encoding='utf-8-sig')
print("\n详细结果已保存为: S-T_output/Ukraine_Hawkes1D_Results.csv")