"""
visualize.py - 冲突事件分析结果可视化
功能：
1. 分支比对比柱状图
2. 参数估计对比图
3. 模型拟合效果图（实际 vs 预测）
4. 强度序列图
5. 综合结果展示
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# 导入已有模块
from read import ConflictDataLoader
from hawkes import DiscreteHawkesMLE, DiscreteHawkesAnalysis


class ConflictVisualizer:
    """
    冲突分析结果可视化类
    复用 DiscreteHawkesAnalysis 的结果
    """

    def __init__(self, hawkes_analyzer):
        """
        初始化可视化器

        Parameters
        ----------
        hawkes_analyzer : DiscreteHawkesAnalysis
            已完成分析的离散Hawkes分析器
        """
        self.analyzer = hawkes_analyzer
        self.results = hawkes_analyzer.results
        self.output_dir = Path('./output/visualization')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_branching_ratio_comparison(self):
        """
        绘制分支比对比柱状图
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        regions = []
        ratios = []
        colors = []
        ci_lower = []
        ci_upper = []

        for region, res in self.results.items():
            regions.append(region.replace('_', '-').title())
            ratios.append(res['params']['branching_ratio'])
            colors.append('#E74C3C' if 'palestine' in region else '#3498DB')

        bars = ax.bar(regions, ratios, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

        # 添加数值标签
        for bar, ratio in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{ratio:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # 添加参考线
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Critical threshold (ρ=1)')
        ax.axhline(y=0.9, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='High contagion (ρ=0.9)')

        ax.set_xlabel('Conflict Region', fontsize=12)
        ax.set_ylabel('Branching Ratio (ρ)', fontsize=12)
        ax.set_title('Branching Ratio Comparison: palestine-Israel vs Russia-Ukraine', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        # 添加解释文本
        ax.text(0.02, 0.02, 'ρ = α/(1-β) where α is contagion, β is persistence\nρ > 0.9 indicates near-critical state',
                transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        fig.savefig(self.output_dir / 'branching_ratio_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved: {self.output_dir / 'branching_ratio_comparison.png'}")

    def plot_parameter_comparison(self):
        """
        绘制参数估计对比图
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        params_config = [
            {'name': 'mu', 'title': 'μ (background intensity)', 'color': '#2ECC71'},
            {'name': 'alpha', 'title': 'α (contagion)', 'color': '#E74C3C'},
            {'name': 'beta', 'title': 'β (persistence)', 'color': '#3498DB'}
        ]

        regions = list(self.results.keys())
        region_labels = [r.replace('_', '-').title() for r in regions]

        for idx, config in enumerate(params_config):
            ax = axes[idx]
            param_name = config['name']
            values = [self.results[r]['params'][param_name] for r in regions]

            bars = ax.bar(region_labels, values, color=config['color'], edgecolor='black', alpha=0.7)

            # 添加数值标签
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)

            ax.set_xlabel('Conflict Region', fontsize=11)
            ax.set_ylabel(config['title'], fontsize=11)
            ax.set_title(config['title'], fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Parameter Estimates Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(self.output_dir / 'parameter_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved: {self.output_dir / 'parameter_comparison.png'}")

    def plot_fitted_vs_actual(self, region='palestine_israel'):
        """
        绘制单个地区的拟合值 vs 实际值对比图

        Parameters
        ----------
        region : str
            'palestine_israel' 或 'russia_ukraine'
        """
        if region not in self.results:
            print(f"  Region {region} not found in results")
            return

        # 重新计算lambda序列
        loader = ConflictDataLoader(jitter_days=0.0001)

        if region == 'palestine_israel':
            df = loader.load_palestine_israel()
        else:
            df = loader.load_russia_ukraine()

        # 按天聚合
        daily_counts = df.groupby(df['event_date'].dt.date).size()
        date_range = pd.date_range(df['event_date'].min(), df['event_date'].max(), freq='D')
        daily_counts = daily_counts.reindex(date_range.date, fill_value=0)
        counts = daily_counts.values

        # 使用估计的参数计算lambda
        params = self.results[region]['params']
        lambda_t = np.zeros(len(counts))
        lambda_t[0] = max(counts[0], 1e-6)

        for t in range(1, len(counts)):
            lambda_t[t] = params['mu'] + params['alpha'] * counts[t - 1] + params['beta'] * lambda_t[t - 1]
            lambda_t[t] = max(lambda_t[t], 1e-6)

        # 创建图表（全貌 + 局部放大）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        time = range(1, len(counts) + 1)

        # 上子图：全貌
        ax1.scatter(time, counts, s=2, alpha=0.3, color='gray', label='Actual')
        ax1.plot(time, lambda_t, 'r-', linewidth=1.5, alpha=0.8, label='Fitted λ_t')

        window = 7
        if len(counts) >= window:
            ma = np.convolve(counts, np.ones(window) / window, mode='same')
            ax1.plot(time, ma, 'b-', linewidth=2, alpha=0.6, label=f'{window}-day Moving Average')

        ax1.set_xlabel('Day', fontsize=12)
        ax1.set_ylabel('Event Count / Intensity', fontsize=12)
        ax1.set_title(f'Fitted vs Actual - {region.replace("_", "-").title()} (Full)',
                      fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        corr = np.corrcoef(counts, lambda_t)[0, 1]
        mae = np.mean(np.abs(counts - lambda_t))
        ax1.text(0.02, 0.98, f'Correlation: {corr:.4f}\nMAE: {mae:.2f}',
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 下子图：局部放大（前100天）
        n_days = min(100, len(counts))
        time_sub = range(1, n_days + 1)
        counts_sub = counts[:n_days]
        lambda_sub = lambda_t[:n_days]

        ax2.scatter(time_sub, counts_sub, s=20, alpha=0.7, color='gray', label='Actual')
        ax2.plot(time_sub, lambda_sub, 'r-', linewidth=2, alpha=0.8, label='Fitted λ_t')

        if len(counts_sub) >= window:
            ma_sub = np.convolve(counts_sub, np.ones(window) / window, mode='same')
            ax2.plot(time_sub, ma_sub, 'b-', linewidth=2, alpha=0.6, label=f'{window}-day MA')

        ax2.set_xlabel('Day', fontsize=12)
        ax2.set_ylabel('Event Count / Intensity', fontsize=12)
        ax2.set_title(f'Fitted vs Actual (First {n_days} Days)', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.output_dir / f'{region}_fitted_vs_actual.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved: {self.output_dir / f'{region}_fitted_vs_actual.png'}")

    def plot_intensity_sequence(self, region='palestine_israel', n_days=100):
        """
        绘制强度序列图（局部放大）

        Parameters
        ----------
        region : str
            'palestine_israel' 或 'russia_ukraine'
        n_days : int
            显示的天数（从开始算起）
        """
        if region not in self.results:
            print(f"  Region {region} not found in results")
            return

        # 重新计算lambda序列
        loader = ConflictDataLoader(jitter_days=0.0001)

        if region == 'palestine_israel':
            df = loader.load_palestine_israel()
        else:
            df = loader.load_russia_ukraine()

        # 按天聚合
        daily_counts = df.groupby(df['event_date'].dt.date).size()
        date_range = pd.date_range(df['event_date'].min(), df['event_date'].max(), freq='D')
        daily_counts = daily_counts.reindex(date_range.date, fill_value=0)
        counts = daily_counts.values[:n_days]

        # 使用估计的参数计算lambda
        params = self.results[region]['params']
        lambda_t = np.zeros(len(counts))
        lambda_t[0] = max(counts[0], 1e-6)

        for t in range(1, len(counts)):
            lambda_t[t] = params['mu'] + params['alpha'] * counts[t - 1] + params['beta'] * lambda_t[t - 1]
            lambda_t[t] = max(lambda_t[t], 1e-6)

        # 绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        time = range(1, len(counts) + 1)

        # 上子图：事件计数
        ax1.bar(time, counts, width=1, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Event Count', fontsize=12)
        ax1.set_title(f'Daily Event Counts - {region.replace("_", "-").title()} (First {n_days} days)',
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # 添加统计信息
        ax1.text(0.95, 0.95, f'Max: {max(counts)}\nMean: {np.mean(counts):.1f}',
                 transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 下子图：条件强度
        ax2.plot(time, lambda_t, 'r-', linewidth=2)
        ax2.fill_between(time, 0, lambda_t, alpha=0.3, color='red')
        ax2.set_xlabel('Day', fontsize=12)
        ax2.set_ylabel('Conditional Intensity λ_t', fontsize=12)
        ax2.set_title('Conditional Intensity Sequence', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.output_dir / f'{region}_intensity_sequence.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved: {self.output_dir / f'{region}_intensity_sequence.png'}")

    def create_results_table(self):
        """
        创建结果汇总表（CSV和图片格式）
        """
        # 构建数据框
        rows = []
        for region, res in self.results.items():
            rows.append({
                'Region': region.replace('_', '-').title(),
                'Days': res['n_days'],
                'Total Events': f"{res['total_events']:,}",
                'Mean Daily': f"{res['mean_daily']:.2f}",
                'mu': f"{res['params']['mu']:.4f}",
                'alpha': f"{res['params']['alpha']:.4f}",
                'beta': f"{res['params']['beta']:.4f}",
                'Branching Ratio': f"{res['params']['branching_ratio']:.4f}",
                'p-value': f"{res['p_value']:.6f}"
            })

        df = pd.DataFrame(rows)

        # 保存CSV
        df.to_csv(self.output_dir / 'results_summary.csv', index=False)
        print(f"  Saved: {self.output_dir / 'results_summary.csv'}")

        # 创建表格图片
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.axis('off')

        table = ax.table(cellText=df.values, colLabels=df.columns,
                         cellLoc='center', loc='center',
                         colColours=['#4472C4'] * len(df.columns))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # 设置表头颜色
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4472C4')

        ax.set_title('Hawkes Model Estimation Results', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        fig.savefig(self.output_dir / 'results_table.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved: {self.output_dir / 'results_table.png'}")

        return df

    def run_all_visualizations(self):
        """
        运行所有可视化
        """
        print("\n" + "=" * 60)
        print("Generating Visualizations")
        print("=" * 60)

        # 1. 分支比对比
        print("\n[1] Branching ratio comparison...")
        self.plot_branching_ratio_comparison()

        # 2. 参数对比
        print("\n[2] Parameter comparison...")
        self.plot_parameter_comparison()

        # 3. 拟合效果图（两个地区）
        print("\n[3] Fitted vs actual plots...")
        for region in self.results.keys():
            self.plot_fitted_vs_actual(region)

        # 4. 强度序列图（两个地区）
        print("\n[4] Intensity sequence plots...")
        for region in self.results.keys():
            self.plot_intensity_sequence(region, n_days=100)

        # 5. 结果汇总表
        print("\n[5] Results summary table...")
        self.create_results_table()

        print("\n" + "=" * 60)
        print(f"All visualizations saved to: {self.output_dir}")
        print("=" * 60)


def main():
    """
    主函数：运行可视化
    """
    print("\n" + "=" * 70)
    print("CONFLICT ANALYSIS VISUALIZATION")
    print("=" * 70)

    # 运行离散Hawkes分析
    print("\nRunning Discrete Hawkes Analysis...")
    analyzer = DiscreteHawkesAnalysis(jitter_days=0.0001)

    for region in ['palestine_israel', 'russia_ukraine']:
        try:
            analyzer.run_analysis(region=region)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        except Exception as e:
            print(f"Error analyzing {region}: {e}")
            return

    # 生成可视化
    visualizer = ConflictVisualizer(analyzer)
    visualizer.run_all_visualizations()

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()