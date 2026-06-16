"""
eda.py - 冲突事件数据探索性分析
复用 read.py 中的 ConflictDataLoader 类
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# 导入 read.py 中的数据加载类
from read import ConflictDataLoader

# 设置图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class ConflictEDA:
    """冲突数据探索性分析类（复用 ConflictDataLoader）"""

    def __init__(self, jitter_days=0.01):
        self.loader = ConflictDataLoader(jitter_days=jitter_days)

    def plot_cumulative_events(self, df, title, ax=None):
        """绘制累积事件曲线"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        df_sorted = df.sort_values('event_timestamp')
        cumulative = np.arange(1, len(df_sorted) + 1)
        time_days = (df_sorted['event_timestamp'] - df_sorted['event_timestamp'].min()).dt.total_seconds() / (24 * 3600)

        ax.plot(time_days, cumulative, 'b-', linewidth=2)
        ax.fill_between(time_days, 0, cumulative, alpha=0.2, color='blue')
        ax.set_xlabel('Time (days since first event)')
        ax.set_ylabel('Cumulative Events')
        ax.set_title(f'Cumulative Events - {title}')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_interarrival_hist(self, df, title, ax=None):
        """绘制事件间隔时间直方图"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        df_sorted = df.sort_values('event_timestamp')
        interarrival = df_sorted['event_timestamp'].diff().dt.total_seconds() / (24 * 3600)
        interarrival = interarrival.dropna()

        # 去除异常大的间隔
        q99 = interarrival.quantile(0.99)
        interarrival_trimmed = interarrival[interarrival <= q99]

        ax.hist(interarrival_trimmed, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Interarrival Time (days)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Interarrival Time Distribution - {title}')
        ax.grid(True, alpha=0.3, axis='y')

        # 统计信息
        stats_text = (
            f'Mean: {interarrival.mean():.3f} days\n'
            f'Median: {interarrival.median():.3f} days\n'
            f'CV: {interarrival.std()/interarrival.mean():.3f}'
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        return ax

    def plot_daily_counts(self, df, title, ax=None):
        """绘制每日事件数柱状图"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))

        daily_counts = df.groupby(df['event_date'].dt.date).size()

        ax.bar(daily_counts.index, daily_counts.values, width=0.8, color='coral', alpha=0.7)
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Events per Day')
        ax.set_title(f'Daily Event Counts - {title}')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        return ax

    def generate_summary_table(self, df, name):
        """生成数据汇总表"""
        df_sorted = df.sort_values('event_timestamp')
        interarrival = df_sorted['event_timestamp'].diff().dt.total_seconds() / (24 * 3600)
        interarrival = interarrival.dropna()
        daily_counts = df.groupby(df['event_date'].dt.date).size()

        summary = pd.DataFrame([{
            'Region': name,
            'Total Events': len(df),
            'Start Date': df['event_date'].min().strftime('%Y-%m-%d'),
            'End Date': df['event_date'].max().strftime('%Y-%m-%d'),
            'Time Span (days)': (df['event_date'].max() - df['event_date'].min()).days,
            'Mean Interarrival (days)': f'{interarrival.mean():.3f}',
            'Median Interarrival (days)': f'{interarrival.median():.3f}',
            'CV of Interarrival': f'{interarrival.std()/interarrival.mean():.3f}',
            'Max Daily Events': daily_counts.max(),
            'Days with Events': len(daily_counts),
        }])

        return summary


def main():
    """主函数"""
    eda = ConflictEDA(jitter_days=0.01)
    output_dir = Path('./output/eda')
    output_dir.mkdir(exist_ok=True)

    regions = [
        ('Palestine_Israel', eda.loader.load_palestine_israel),
        ('Russia_Ukraine', eda.loader.load_russia_ukraine)
    ]

    all_summaries = []

    for name, load_func in regions:
        print(f"\n{'='*60}\n正在分析: {name}\n{'='*60}")

        try:
            df = load_func()

            if len(df) == 0:
                print(f"警告: {name} 没有找到符合条件的事件")
                continue

            # 汇总表
            summary = eda.generate_summary_table(df, name)
            all_summaries.append(summary)
            print("\n数据汇总:")
            print(summary.to_string(index=False))

            # 绘图
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Exploratory Data Analysis - {name}', fontsize=14)

            eda.plot_cumulative_events(df, name, ax=axes[0, 0])
            eda.plot_interarrival_hist(df, name, ax=axes[0, 1])

            fig.delaxes(axes[1, 1])
            eda.plot_daily_counts(df, name, ax=axes[1, 0])
            axes[1, 0].set_position([0.1, 0.1, 0.8, 0.35])

            plt.tight_layout()
            plt.savefig(output_dir / f'{name}_eda.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n图表已保存: {output_dir}/{name}_eda.png")

        except FileNotFoundError as e:
            print(f"错误: {e}")
        except Exception as e:
            print(f"处理 {name} 时出错: {e}")

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(output_dir / 'data_summary.csv', index=False)

    print("\nEDA 分析完成")


if __name__ == "__main__":
    main()