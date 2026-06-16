"""
diagnostics.py - 离散时间Hawkes模型诊断
适用于INGARCH模型，复用已有模块
功能：
1. 残差分析（Pearson残差）
2. QQ图（检验残差是否服从正态分布）
3. 残差自相关检验
4. 预测 vs 实际对比图
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import poisson

# 导入已有模块
from read import ConflictDataLoader
from hawkes import DiscreteHawkesMLE


class DiscreteHawkesDiagnostics:
    """
    离散时间Hawkes模型诊断类
    用于检验INGARCH模型拟合优度
    """

    def __init__(self, counts, mu, alpha, beta):
        """
        初始化诊断器

        Parameters
        ----------
        counts : np.ndarray
            每日事件计数序列
        mu, alpha, beta : float
            模型参数
        """
        self.counts = np.asarray(counts).flatten()
        self.T = len(self.counts)
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

        # 计算条件强度和残差
        self.lambda_t, self.residuals = self._compute_lambda_and_residuals()

    def _compute_lambda_and_residuals(self):
        """
        计算条件强度 λ_t 和 Pearson 残差

        Pearson残差公式：
            r_t = (Y_t - λ_t) / sqrt(λ_t)

        如果模型正确，残差应近似服从 N(0, 1)

        Returns
        -------
        lambda_t : np.ndarray
            条件强度序列
        residuals : np.ndarray
            Pearson残差序列
        """
        lambda_t = np.zeros(self.T)

        # 初始化：使用第一个观测值作为初始强度
        lambda_t[0] = max(self.counts[0], 1e-6)

        # 递归计算 λ_t
        for t in range(1, self.T):
            lambda_t[t] = self.mu + self.alpha * self.counts[t-1] + self.beta * lambda_t[t-1]
            lambda_t[t] = max(lambda_t[t], 1e-6)  # 避免零值

        # 计算 Pearson 残差
        # 注意：当 λ_t 很小时，残差可能不稳定，但这种情况在冲突数据中少见
        residuals = (self.counts - lambda_t) / np.sqrt(lambda_t)

        # 替换可能出现的 NaN 或 Inf
        residuals = np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)

        return lambda_t, residuals

    def residual_qq_plot(self, ax=None, title=None):
        """
        绘制残差的QQ图（与标准正态分布对比）

        Parameters
        ----------
        ax : matplotlib.axes, optional
            绘图轴
        title : str, optional
            图表标题

        Returns
        -------
        ax : matplotlib.axes
            绘图轴
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        residuals_sorted = np.sort(self.residuals)
        n = len(residuals_sorted)

        # 理论正态分位数
        p = np.arange(1, n + 1) / (n + 1)
        theoretical_quantiles = stats.norm.ppf(p)

        # 绘制QQ图
        ax.scatter(theoretical_quantiles, residuals_sorted, alpha=0.5, s=10, color='steelblue')

        # 参考线 y = x
        max_val = max(abs(theoretical_quantiles).max(), abs(residuals_sorted).max())
        ax.plot([-max_val, max_val], [-max_val, max_val], 'r--', linewidth=2, label='y = x')

        ax.set_xlabel('Theoretical Normal Quantiles', fontsize=12)
        ax.set_ylabel('Sample Quantiles (Pearson Residuals)', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('QQ Plot: Residuals vs Standard Normal', fontsize=14)

        ax.legend()
        ax.grid(True, alpha=0.3)

        # 正态性检验
        if n >= 3 and n <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(self.residuals)
            ax.text(0.05, 0.95, f'Shapiro-Wilk p-value: {shapiro_p:.6f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            # 大样本用 Kolmogorov-Smirnov
            ks_stat, ks_p = stats.kstest(self.residuals, 'norm')
            ax.text(0.05, 0.95, f'KS test p-value: {ks_p:.6f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        return ax

    def residual_histogram(self, ax=None, title=None):
        """
        绘制残差直方图（与标准正态密度对比）

        Parameters
        ----------
        ax : matplotlib.axes, optional
            绘图轴
        title : str, optional
            图表标题

        Returns
        -------
        ax : matplotlib.axes
            绘图轴
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制直方图
        ax.hist(self.residuals, bins=50, density=True, alpha=0.7,
                color='steelblue', edgecolor='black', label='Pearson Residuals')

        # 绘制理论正态分布密度
        x = np.linspace(-4, 4, 100)
        theoretical_pdf = stats.norm.pdf(x, 0, 1)
        ax.plot(x, theoretical_pdf, 'r-', linewidth=2, label='N(0,1) PDF')

        ax.set_xlabel('Pearson Residual', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Residual Distribution vs Standard Normal', fontsize=14)

        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加统计量
        mean_res = np.mean(self.residuals)
        std_res = np.std(self.residuals)
        skew_res = stats.skew(self.residuals)
        kurt_res = stats.kurtosis(self.residuals)

        stats_text = (f'Mean: {mean_res:.4f} (expected 0)\n'
                      f'Std: {std_res:.4f} (expected 1)\n'
                      f'Skewness: {skew_res:.4f}\n'
                      f'Kurtosis: {kurt_res:.4f}')
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        return ax

    def residual_autocorrelation(self, max_lag=20, ax=None, title=None):
        """
        绘制残差自相关函数

        Parameters
        ----------
        max_lag : int
            最大滞后阶数
        ax : matplotlib.axes, optional
            绘图轴
        title : str, optional
            图表标题

        Returns
        -------
        ax : matplotlib.axes
            绘图轴
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        # 计算自相关
        n = len(self.residuals)
        lags = range(1, min(max_lag, n) + 1)
        autocorr = []
        ci_values = []

        residuals_centered = self.residuals - np.mean(self.residuals)

        for lag in lags:
            if lag >= n:
                autocorr.append(0)
                ci_values.append(0)
            else:
                corr = np.corrcoef(residuals_centered[:-lag], residuals_centered[lag:])[0, 1]
                autocorr.append(corr if not np.isnan(corr) else 0)

        # 绘制条形图
        ax.bar(lags, autocorr, width=0.8, color='steelblue', alpha=0.7, edgecolor='black')

        # 添加95%置信区间
        ci = 1.96 / np.sqrt(n)
        ax.axhline(ci, color='r', linestyle='--', linewidth=1, label=f'95% CI (±{ci:.3f})')
        ax.axhline(-ci, color='r', linestyle='--', linewidth=1)
        ax.axhline(0, color='black', linewidth=0.5)

        ax.set_xlabel('Lag', fontsize=12)
        ax.set_ylabel('Autocorrelation', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Residual Autocorrelation Function', fontsize=14)

        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 计算超出置信区间的比例
        outside_ci = np.sum(np.abs(autocorr) > ci) / len(autocorr) if len(autocorr) > 0 else 0
        ax.text(0.95, 0.05, f'% outside CI: {outside_ci*100:.1f}%',
                transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        return ax

    def residual_sequence_plot(self, ax=None, title=None):
        """
        绘制残差序列图（用于检验独立性）

        Parameters
        ----------
        ax : matplotlib.axes, optional
            绘图轴
        title : str, optional
            图表标题

        Returns
        -------
        ax : matplotlib.axes
            绘图轴
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        time = range(1, self.T + 1)

        # 绘制残差序列
        ax.plot(time, self.residuals, 'o', markersize=2, alpha=0.5, color='steelblue')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Zero line')
        ax.axhline(y=2, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='±2 SD')
        ax.axhline(y=-2, color='orange', linestyle=':', linewidth=1, alpha=0.7)

        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Pearson Residual', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Residual Sequence (Independence Check)', fontsize=14)

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 添加统计信息
        mean_res = np.mean(self.residuals)
        std_res = np.std(self.residuals)
        ax.text(0.02, 0.98, f'Mean: {mean_res:.4f}\nStd: {std_res:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        return ax

    def run_all_diagnostics(self, region_name, n_samples_preview=100):
        """运行所有诊断并生成综合图表"""
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f'Model Diagnostics - {region_name}', fontsize=14, fontweight='bold')

        # 2x2 布局，只保留诊断图
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 子图1：QQ图
        ax1 = fig.add_subplot(gs[0, 0])
        self.residual_qq_plot(ax=ax1)
        ax1.set_xlim([-20, 20])  # 统一设置范围
        ax1.set_ylim([-20, 20])

        # 子图2：残差直方图
        ax2 = fig.add_subplot(gs[0, 1])
        self.residual_histogram(ax=ax2)
        ax2.set_xlim([-20, 20])
        ax2.set_ylim([0.00, 0.50])

        # 子图3：自相关函数
        ax3 = fig.add_subplot(gs[1, 0])
        self.residual_autocorrelation(ax=ax3)
        ax3.set_xlim([0.0, 20.0])
        ax3.set_ylim([-0.20, 0.40])

        # 子图4：残差序列图（替代拟合图）
        ax4 = fig.add_subplot(gs[1, 1])
        self.residual_sequence_plot(ax=ax4)  # 新增残差序列图
        ax4.set_ylim([-15, 20])

        plt.tight_layout()

        # 汇总诊断统计量
        diagnostics_summary = {
            'n_days': self.T,
            'mean_residual': np.mean(self.residuals),
            'std_residual': np.std(self.residuals),
            'residual_skewness': stats.skew(self.residuals),
            'residual_kurtosis': stats.kurtosis(self.residuals),
            'corr_actual_fitted': np.corrcoef(self.counts, self.lambda_t)[0, 1],
            'mae': np.mean(np.abs(self.counts - self.lambda_t)),
            'rmse': np.sqrt(np.mean((self.counts - self.lambda_t) ** 2))
        }

        # 正态性检验
        if self.T >= 3 and self.T <= 5000:
            _, shapiro_p = stats.shapiro(self.residuals)
            diagnostics_summary['normality_p_value'] = shapiro_p
        else:
            _, ks_p = stats.kstest(self.residuals, 'norm')
            diagnostics_summary['normality_p_value'] = ks_p

        return fig, diagnostics_summary


def run_diagnostics_for_region(region='palestine_israel', n_samples_preview=100):
    """
    对指定地区运行完整的模型诊断

    Parameters
    ----------
    region : str
        'palestine_israel' 或 'russia_ukraine'
    n_samples_preview : int
        局部预览图显示的天数

    Returns
    -------
    diagnostics : DiscreteHawkesDiagnostics
        诊断器对象
    summary : dict
        诊断汇总
    """
    print(f"\n{'='*60}")
    print(f"Discrete Hawkes Diagnostics - {region.upper()}")
    print(f"{'='*60}")

    # 加载数据（不需要时间抖动）
    loader = ConflictDataLoader(jitter_days=0.0001)

    if region == 'palestine_israel':
        df = loader.load_palestine_israel()
    else:
        df = loader.load_russia_ukraine()

    # 按天聚合
    daily_counts = df.groupby(df['event_date'].dt.date).size()

    # 填充缺失日期
    date_range = pd.date_range(df['event_date'].min(), df['event_date'].max(), freq='D')
    daily_counts = daily_counts.reindex(date_range.date, fill_value=0)

    counts = daily_counts.values
    T = len(counts)
    total_events = counts.sum()

    print(f"  Days: {T}")
    print(f"  Total events: {total_events:,}")
    print(f"  Mean daily: {counts.mean():.2f}")
    print(f"  Zero days: {(counts == 0).sum()} ({100*(counts==0).mean():.1f}%)")
    print(f"  Max daily: {counts.max()}")

    # 从之前保存的结果加载参数
    results_path = Path('./output/hawkes/hawkes_results_discrete.json')
    if results_path.exists():
        import json
        with open(results_path, 'r') as f:
            saved_results = json.load(f)
        params = saved_results.get(region, {})
        mu = params.get('mu', None)
        alpha = params.get('alpha', None)
        beta = params.get('beta', None)

        if mu is not None:
            print(f"\n  Loading parameters from saved results:")
            print(f"    mu = {mu:.6f}")
            print(f"    alpha = {alpha:.6f}")
            print(f"    beta = {beta:.6f}")
        else:
            # 重新拟合
            print(f"\n  Re-fitting model (parameters not found)...")
            model = DiscreteHawkesMLE(counts)
            result = model.fit()
            mu = result['mu']
            alpha = result['alpha']
            beta = result['beta']
            print(f"    mu = {mu:.6f}")
            print(f"    alpha = {alpha:.6f}")
            print(f"    beta = {beta:.6f}")
    else:
        # 重新拟合
        print(f"\n  Fitting model (saved results not found)...")
        model = DiscreteHawkesMLE(counts)
        result = model.fit()
        mu = result['mu']
        alpha = result['alpha']
        beta = result['beta']
        print(f"    mu = {mu:.6f}")
        print(f"    alpha = {alpha:.6f}")
        print(f"    beta = {beta:.6f}")

    # 创建诊断器
    diagnostics = DiscreteHawkesDiagnostics(counts, mu, alpha, beta)

    # 运行诊断
    fig, summary = diagnostics.run_all_diagnostics(region.upper(), n_samples_preview=n_samples_preview)

    # 保存图表
    output_dir = Path('./output/diagnostics')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f'{region}_diagnostics_discrete.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  Diagnostics summary:")
    print(f"    Mean residual: {summary['mean_residual']:.4f} (expected 0)")
    print(f"    Std residual: {summary['std_residual']:.4f} (expected 1)")
    print(f"    Skewness: {summary['residual_skewness']:.4f} (expected 0)")
    print(f"    Kurtosis: {summary['residual_kurtosis']:.4f} (expected 0)")
    print(f"    Normality p-value: {summary['normality_p_value']:.6f}")
    print(f"    Correlation actual/fitted: {summary['corr_actual_fitted']:.4f}")
    print(f"    MAE: {summary['mae']:.2f}")
    print(f"    RMSE: {summary['rmse']:.2f}")

    if summary['normality_p_value'] > 0.05:
        print(f"    Conclusion: Residuals follow N(0,1) (p > 0.05)")
    else:
        print(f"    Conclusion: Residuals deviate from N(0,1) (p <= 0.05)")

    print(f"\n  Chart saved: {output_dir}/{region}_diagnostics_discrete.png")

    return diagnostics, summary


def main():
    """
    主函数：运行两个地区的模型诊断
    """
    print("\n" + "="*70)
    print("DISCRETE HAWKES MODEL DIAGNOSTICS")
    print("="*70)

    all_summaries = []

    # 诊断两个地区
    for region in ['palestine_israel', 'russia_ukraine']:
        try:
            diag, summary = run_diagnostics_for_region(region=region, n_samples_preview=100)
            summary['region'] = region
            all_summaries.append(summary)
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error diagnosing {region}: {e}")
            import traceback
            traceback.print_exc()

    # 汇总诊断结果
    if all_summaries:
        print("\n" + "="*70)
        print("DIAGNOSTICS SUMMARY")
        print("="*70)

        summary_df = pd.DataFrame(all_summaries)
        # 选择要显示的列
        display_cols = ['region', 'n_days', 'mean_residual', 'std_residual',
                        'residual_skewness', 'residual_kurtosis', 'normality_p_value',
                        'corr_actual_fitted', 'mae', 'rmse']
        print(summary_df[display_cols].to_string(index=False))

    print("\n" + "="*70)
    print("DIAGNOSTICS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()