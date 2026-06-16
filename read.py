import pandas as pd
import numpy as np
from pathlib import Path


class ConflictDataLoader:
    """冲突数据加载器，筛选导弹/空袭事件"""

    TARGET_EVENT_TYPES = ['Explosions/Remote violence']
    TARGET_SUB_TYPES = ['Shelling/artillery/missile attack', 'Air/drone strike']

    def __init__(self, jitter_days=0.1):
        self.russia_ukraine_file = './data/Russia_Ukraine.csv'
        self.palestine_israel_file = './data/Palestine_Israel.csv'
        self.jitter_days = jitter_days  # 时间抖动参数

    @staticmethod
    def _filter_events(df):
        """筛选导弹/空袭事件"""
        mask = (
            df['event_type'].isin(ConflictDataLoader.TARGET_EVENT_TYPES) &
            df['sub_event_type'].isin(ConflictDataLoader.TARGET_SUB_TYPES)
        )
        return df[mask].copy()

    def _add_time_jitter(self, df):
        """为同一天的事件添加微小时间抖动，避免重复时间戳"""
        df = df.copy()
        df['event_timestamp'] = df['event_date'].copy()

        for date in df['event_date'].unique():
            mask = df['event_date'] == date
            n_events = mask.sum()
            if n_events > 1:
                # 添加渐进偏移，保证时间顺序与原数据一致
                # 使用 (i / n_events) * jitter_days 确保同一天内事件按原顺序排列
                indices = np.arange(n_events)
                jitter = (indices / max(1, n_events - 1)) * self.jitter_days if n_events > 1 else np.zeros(n_events)
                df.loc[mask, 'event_timestamp'] = df.loc[mask, 'event_date'] + pd.to_timedelta(jitter, unit='D')

        return df

    def load_russia_ukraine(self):
        """加载俄乌冲突数据"""
        if not Path(self.russia_ukraine_file).exists():
            raise FileNotFoundError(f"文件不存在: {self.russia_ukraine_file}")

        df = pd.read_csv(self.russia_ukraine_file)
        df_filtered = self._filter_events(df)
        df_filtered['event_date'] = pd.to_datetime(df_filtered['event_date'])
        df_filtered = df_filtered.sort_values('event_date').reset_index(drop=True)

        # 添加时间抖动
        df_filtered = self._add_time_jitter(df_filtered)

        result = df_filtered[['event_date', 'event_timestamp', 'event_type',
                              'sub_event_type', 'country', 'location', 'fatalities']].copy()

        return result

    def load_palestine_israel(self):
        """加载巴以冲突数据"""
        if not Path(self.palestine_israel_file).exists():
            raise FileNotFoundError(f"文件不存在: {self.palestine_israel_file}")

        df = pd.read_csv(self.palestine_israel_file)
        df_filtered = self._filter_events(df)
        df_filtered['event_date'] = pd.to_datetime(df_filtered['event_date'])
        df_filtered = df_filtered.sort_values('event_date').reset_index(drop=True)

        # 添加时间抖动
        df_filtered = self._add_time_jitter(df_filtered)

        result = df_filtered[['event_date', 'event_timestamp', 'event_type',
                              'sub_event_type', 'country', 'location', 'fatalities']].copy()

        return result

    def get_time_series(self, region='palestine_israel'):
        """获取用于 Hawkes 模型的时间序列（秒为单位）"""
        if region == 'russia_ukraine':
            df = self.load_russia_ukraine()
        else:
            df = self.load_palestine_israel()

        # 转换为数值时间（秒，从第一个事件开始）
        start_time = df['event_timestamp'].min()
        time_seconds = (df['event_timestamp'] - start_time).dt.total_seconds()

        return time_seconds.values, df


def demo():
    """演示：加载数据并输出前10条sample"""
    loader = ConflictDataLoader(jitter_days=0.01)

    print("=" * 70)
    print("巴以冲突 - 导弹/空袭事件 (前10条，已处理时间抖动)")
    print("=" * 70)
    try:
        ir = loader.load_palestine_israel()
        print(f"总事件数: {len(ir)}")
        print("\n事件统计（按天）:")
        print(ir.groupby(ir['event_date'].dt.date).size())
        print("\n前10条数据:")
        print(ir.head(10).to_string())

        # 获取用于Hawkes模型的时间序列
        times, df = loader.get_time_series('palestine_israel')
        print(f"\n时间序列（秒，相对时间）: {times[:10]}")

    except FileNotFoundError as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    demo()