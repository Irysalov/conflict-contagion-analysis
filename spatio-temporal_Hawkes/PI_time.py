from matplotlib.offsetbox import AnchoredText
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
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
print("巴勒斯坦-以色列冲突 Hawkes 自激发系数热力图 (按区级行政区划)")
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


def load_shapefile_with_center(shp_path, country_name):
    """
    加载 shapefile 并提取中心点坐标
    返回: (districts列表, centers数组)
    """
    if not os.path.exists(shp_path):
        print(f"  文件不存在: {shp_path}")
        return [], None

    try:
        sf = shapefile.Reader(shp_path)
        field_names = [field[0] for field in sf.fields[1:]]

        # 查找名称字段
        name_field = None
        for possible_name in ['NAME_2', 'NAME_1', 'NL_NAME_2', 'VARNAME_2', 'NAME_2_', 'NAME_1_', 'NAME_0']:
            if possible_name in field_names:
                name_field = possible_name
                break

        districts = []
        centers = []

        for idx, shape in enumerate(sf.shapes()):
            # 提取多边形的所有 parts
            parts_coords = []
            parts_idx = list(shape.parts) + [len(shape.points)]
            for i in range(len(parts_idx)-1):
                part_lons = [p[0]
                             for p in shape.points[parts_idx[i]:parts_idx[i+1]]]
                part_lats = [p[1]
                             for p in shape.points[parts_idx[i]:parts_idx[i+1]]]
                parts_coords.append((part_lons, part_lats))

            # 读取名称
            rec = sf.record(idx)
            if name_field:
                name = rec[field_names.index(name_field)]
            else:
                name = f"{country_name}_{idx}"

            # 计算中心点和边界框
            all_lons = []
            all_lats = []
            for plon, plat in parts_coords:
                all_lons.extend(plon)
                all_lats.extend(plat)

            if all_lons and all_lats:
                center_lon = (min(all_lons) + max(all_lons)) / 2
                center_lat = (min(all_lats) + max(all_lats)) / 2
                bbox = (min(all_lons), max(all_lons),
                        min(all_lats), max(all_lats))
            else:
                center_lon, center_lat = 0.0, 0.0
                bbox = (0.0, 0.0, 0.0, 0.0)

            districts.append({
                'id': idx,
                'name': name,
                'parts': parts_coords,
                'country': country_name,
                'center': (center_lon, center_lat),
                'bbox': bbox
            })
            centers.append([center_lon, center_lat])

        return districts, np.array(centers)

    except Exception as e:
        print(f"  加载失败: {e}")
        return [], None


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

print(
    f"使用列: 日期={df['event_date'].name}, 经度={df['longitude'].name}, 纬度={df['latitude'].name}")

# 转换日期
df['event_date'] = pd.to_datetime(df['event_date'])
df = df.sort_values('event_date').reset_index(drop=True)
start_date = df['event_date'].min()
df['t'] = (df['event_date'] - start_date).dt.days
T = df['t'].max() + 1
original_len = len(df)

print(f"时间跨度: {T} 天, 事件总数: {original_len}")
print(f"数据范围: 经度 [{df['longitude'].min():.2f}, {df['longitude'].max():.2f}], "
      f"纬度 [{df['latitude'].min():.2f}, {df['latitude'].max():.2f}]")

# ========== 设置经纬度范围 ==========
# 使用数据自动扩展
lon_range = df['longitude'].max() - df['longitude'].min()
lat_range = df['latitude'].max() - df['latitude'].min()

lon_min = df['longitude'].min() - lon_range * 0.15
lon_max = df['longitude'].max() + lon_range * 0.15
lat_min = df['latitude'].min() - lat_range * 0.15
lat_max = df['latitude'].max() + lat_range * 0.15

# 限制在中东合理范围
lon_min = max(lon_min, 30)
lon_max = min(lon_max, 65)
lat_min = max(lat_min, 10)
lat_max = min(lat_max, 45)

print(
    f"地图范围: 经度 [{lon_min:.2f}°E - {lon_max:.2f}°E], 纬度 [{lat_min:.2f}°N - {lat_max:.2f}°N]")

# ========== 2. 加载所有相关国家的行政区划 ==========
countries_config = [
    ('gadm41_IRN_shp', 'gadm41_IRN_shp/gadm41_IRN_2.shp', 'Palestine'),
    ('gadm41_ISR_shp', 'gadm41_ISR_shp/gadm41_ISR_1.shp', 'Israel'),  # 用 _1.shp
    ('gadm41_LBN_shp', 'gadm41_LBN_shp/gadm41_LBN_2.shp', 'Lebanon'),
    ('gadm41_PSE_shp', 'gadm41_PSE_shp/gadm41_PSE_2.shp', 'Palestine'),
    ('gadm41_SYR_shp', 'gadm41_SYR_shp/gadm41_SYR_2.shp', 'Syria'),
    ('gadm41_YEM_shp', 'gadm41_YEM_shp/gadm41_YEM_2.shp', 'Yemen')
]

all_districts = []
all_centers = []

print("\n加载行政区划...")
for country_dir, shp_path, country_name in countries_config:
    print(f"  加载 {country_name}...")

    # 尝试不同的路径变体
    possible_paths = [
        shp_path,
        f"{country_dir}/gadm41_{country_name[:3].upper()}_2.shp",
        f"{country_dir}/{country_name}_adm2.shp",
        f"{country_dir}/{country_name}_2.shp"
    ]

    loaded = False
    for path in possible_paths:
        if os.path.exists(path):
            districts, centers = load_shapefile_with_center(path, country_name)
            if districts:
                print(f"    ✓ 成功加载 {country_name}: {len(districts)} 个区级单元")
                all_districts.extend(districts)
                if centers is not None:
                    all_centers.extend(centers)
                loaded = True
                break

    if not loaded:
        print(f"    ✗ {country_name} 加载失败，跳过")

print(f"\n总共加载了 {len(all_districts)} 个区级单元")
n_districts = len(all_districts)

if n_districts == 0:
    raise ValueError("没有加载到任何行政区划数据！")

# ========== 3. 匹配事件到行政区 ==========
print("\n正在将事件匹配到行政区...")
df['district_id'] = -1

district_bbox = [d['bbox'] for d in all_districts]

matched_count = 0
for i, row in df.iterrows():
    lon, lat = row['longitude'], row['latitude']

    # 检查是否在绘图范围内
    if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
        continue

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

    if (i+1) % 5000 == 0:
        print(f"  已处理 {i+1}/{original_len} 个事件，匹配成功 {matched_count} 个")

df = df[df['district_id'] >= 0].copy()
print(f"成功匹配: {len(df)}/{original_len} ({len(df)/original_len*100:.1f}%)")

if len(df) == 0:
    raise ValueError("没有事件匹配到任何行政区！")

# ========== 4. 构建时间序列 ==========
Y_district = np.zeros((T, n_districts), dtype=np.float32)
for _, row in df.iterrows():
    Y_district[int(row['t']), int(row['district_id'])] += 1

active_districts = np.where(Y_district.sum(axis=0) > 0)[0]
print(f"有事件的区: {len(active_districts)}/{n_districts}")

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
alpha_by_district = np.full(n_districts, np.nan)
beta_by_district = np.full(n_districts, np.nan)
corr_by_district = np.full(n_districts, np.nan)
p_by_district = np.full(n_districts, 1.0)

print(f"\n开始拟合 {len(active_districts)} 个有事件的区...")
successful_fits = 0

for j, did in enumerate(active_districts):
    series = Y_district[:, did]
    mu, alpha, beta, ll_full = fit_hawkes_1d(series)

    if mu is not None and alpha is not None:
        # 重建 Lambda 序列以获取相关性
        lam = np.zeros_like(series, dtype=np.float64)
        lam[0] = mu
        for t in range(1, len(series)):
            lam[t] = mu + alpha * series[t-1] + beta * (lam[t-1] - mu)
        
        # 计算皮尔逊相关系数
        if np.std(lam) > 1e-8 and np.std(series) > 1e-8:
            corr = np.corrcoef(series, lam)[0, 1]
        else:
            corr = 0.0

        # 零模型对数似然 (用于算 p-value)
        mu0 = np.mean(series) + 1e-8
        ll_null = np.sum(series * np.log(mu0) - mu0 - gammaln(series + 1))
        LR = max(2 * (ll_full - ll_null), 0)
        p_val = 0.5 * (1 - chi2.cdf(LR, df=1))
        
        alpha_by_district[did] = alpha
        beta_by_district[did] = beta
        corr_by_district[did] = corr
        p_by_district[did] = max(p_val, 1e-10)
        successful_fits += 1

    if (j+1) % 50 == 0:
        print(f"  已完成 {j+1} / {len(active_districts)} 个区...")

print(f"\n成功拟合的区数量: {successful_fits} / {len(active_districts)}")

# ========== 7. 预加载国家边界（避免绘图时重复加载3次） ==========
country_shp_paths = [
    ('gadm41_IRN_shp/gadm41_IRN_0.shp', 'Palestine'),
    ('gadm41_ISR_shp/gadm41_ISR_0.shp', 'Israel'),
    ('gadm41_LBN_shp/gadm41_LBN_0.shp', 'Lebanon'),
    ('gadm41_PSE_shp/gadm41_PSE_0.shp', 'Palestine'),
    ('gadm41_SYR_shp/gadm41_SYR_0.shp', 'Syria'),
    ('gadm41_YEM_shp/gadm41_YEM_0.shp', 'Yemen')
]
country_borders = []

print("\n正在预加载国家边界...")
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

# ========== 8. 封装可视化绘图函数 ==========
def plot_heatmap(data_array, title, color_list, vmin, vmax, cbar_label, filename):
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), facecolor='white')
    ax.set_facecolor('#def1fe')

    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', color_list, N=256)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # 背景（非研究区域）
    ax.add_patch(Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                           facecolor='#fbfafb', edgecolor='none', zorder=0))

    # 绘制所有行政区填充
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

    # 绘制显著性轮廓（p < 0.05）
    significant_count = 0
    for did in active_districts:
        if p_by_district[did] < 0.05 and not np.isnan(alpha_by_district[did]):
            for plon, plat in all_districts[did]['parts']:
                if len(plon) > 2:
                    ax.plot(plon, plat, color='#78aaee', linewidth=1.5,
                            linestyle='dashdot', alpha=0.8, zorder=6)
                    significant_count += 1
                    break

    # 设置图形属性
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('经度', fontsize=12)
    ax.set_ylabel('纬度', fontsize=12)

    for lat in [30, 35, 40]:
        ax.axhline(y=lat, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    for lon in [40, 50, 60]:
        ax.axvline(x=lon, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.1, linestyle=':', zorder=0)

    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=30)
    cbar.set_label(cbar_label, fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # 图例
    legend_elements = [
        Patch(facecolor='#fef7ec', edgecolor='#dfb492', label='无事件区域'),
        Patch(facecolor='#def1fe', edgecolor='none', label='海域'),
        Patch(facecolor='#fbfafb', edgecolor='none', label='非研究区域'),
        Line2D([0], [0], color='black', linewidth=2.0, label='国家边界'),
        Line2D([0], [0], color='#dfb492', linewidth=0.6, label='区界线'),
        Line2D([0], [0], color='#78aaee', linewidth=1.5, linestyle='dashdot', label='显著区域 (p < 0.05)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
              framealpha=0.9, edgecolor='black', fancybox=True)

    # 添加信息框
    at = AnchoredText(f"时间跨度: {T} 天\n有事件区: {len(active_districts)}\n显著区: {significant_count}",
                      prop=dict(size=9), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.3")
    ax.add_artist(at)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

# ========== 9. 生成三张热力图 ==========
print("\n正在生成热力图...")

# 1. Alpha 图 (红色系)
colors_red = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
plot_heatmap(alpha_by_district, 
             '巴勒斯坦-以色列冲突：自激发系数 (Alpha) 空间分布\n(红色越深 → 轰炸后再次轰炸概率越高)', 
             colors_red, 0, 0.6, '自激发系数 α (Alpha)', 'S-T_output/Palestine_Israel_Hawkes1D_Alpha.png')

# 2. Beta 图 (蓝色系)
colors_blue = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
plot_heatmap(beta_by_district, 
             '巴勒斯坦-以色列冲突：时间记忆衰减系数 (Beta) 空间分布\n(蓝色越深 → 记忆保留越久，衰减越慢)', 
             colors_blue, 0, 1.0, '衰减系数 β (Beta)', 'S-T_output/Palestine_Israel_Hawkes1D_Beta.png')

# 3. Correlation 图 (紫色系)
colors_purple = ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d']
plot_heatmap(corr_by_district, 
             '巴勒斯坦-以色列冲突：模型拟合强度 λ 与实际发生数 Y 皮尔逊相关系数\n(紫色越深 → 模型拟合与真实序列越吻合)', 
             colors_purple, 0, 1.0, 'Pearson r', 'S-T_output/Palestine_Israel_Hawkes1D_Correlation.png')

# ========== 10. 输出统计信息 ==========
print("\n" + "="*70)
print("统计结果")
print("="*70)
significant_count = sum(1 for did in active_districts if p_by_district[did] < 0.05 and not np.isnan(alpha_by_district[did]))
print(f"总行政区数量: {n_districts}")
print(f"有事件行政区: {len(active_districts)}")
print(f"显著区数量 (p<0.05): {significant_count}")
if len(active_districts) > 0:
    print(f"平均 Alpha: {np.nanmean(alpha_by_district):.4f}")
    print(f"平均 Beta: {np.nanmean(beta_by_district):.4f}")
    print(f"平均 皮尔逊相关性: {np.nanmean(corr_by_district):.4f}")

# 输出详细结果
result_data = []
for did in active_districts:
    if did < len(all_districts):
        result_data.append({
            '国家': all_districts[did]['country'],
            '行政区名称': all_districts[did]['name'],
            'Alpha': alpha_by_district[did],
            'Beta': beta_by_district[did],
            'Corr(Y, λ)': corr_by_district[did],
            'P值': p_by_district[did],
            '显著': '✓' if p_by_district[did] < 0.05 else '',
            '事件总数': int(Y_district[:, did].sum())
        })

if result_data:
    result_df = pd.DataFrame(result_data).sort_values('Alpha', ascending=False)
    print("\nAlpha 值最高的前10个行政区:")
    print(result_df.head(10).to_string(index=False))

    print("\n=== 按国家统计 ===")
    country_stats = result_df.groupby('国家').agg({
        'Alpha': ['count', 'mean', 'max'],
        'Beta': ['mean'],
        'Corr(Y, λ)': ['mean'],
        '显著': lambda x: (x == '✓').sum()
    }).round(4)
    print(country_stats)

    result_df.to_csv('S-T_output/Palestine_Israel_Hawkes1D_Results.csv', index=False, encoding='utf-8-sig')
    print("\n详细结果已保存为: S-T_output/Palestine_Israel_Hawkes1D_Results.csv")