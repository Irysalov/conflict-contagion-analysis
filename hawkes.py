"""
hawkes.py - 离散时间Hawkes模型（INGARCH）
适用于日度计数数据，无需时间抖动
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize

from read import ConflictDataLoader


class DiscreteHawkesMLE:
    """
    离散时间Hawkes模型 (INGARCH)

    模型形式：
        Y_t ~ Poisson(λ_t)
        λ_t = μ + α * Y_{t-1} + β * λ_{t-1}

    其中 Y_t 是第 t 天的事件数
    """

    def __init__(self, daily_counts):
        """
        初始化

        Parameters
        ----------
        daily_counts : np.ndarray
            每日事件计数序列
        """
        self.counts = np.asarray(daily_counts).flatten()
        self.T = len(self.counts)

    def compute_lambda(self, mu, alpha, beta, initial_lambda=None):
        """
        递归计算条件强度 λ_t

        Parameters
        ----------
        mu, alpha, beta : float
            模型参数
        initial_lambda : float, optional
            初始强度，默认为第一个观测值

        Returns
        -------
        lambda_t : np.ndarray
            每日条件强度
        """
        lambda_t = np.zeros(self.T)

        # 初始化
        if initial_lambda is None:
            lambda_t[0] = max(self.counts[0], 1e-6)
        else:
            lambda_t[0] = initial_lambda

        # 递归计算
        for t in range(1, self.T):
            lambda_t[t] = mu + alpha * self.counts[t-1] + beta * lambda_t[t-1]
            lambda_t[t] = max(lambda_t[t], 1e-6)  # 避免零值

        return lambda_t

    def negative_log_likelihood(self, params):
        """
        负对数似然函数

        Parameters
        ----------
        params : array-like
            [mu, alpha, beta]

        Notes
        -----
        Poisson 对数似然：log L = Σ [Y_t * log(λ_t) - λ_t - log(Y_t!)]
        其中 -log(Y_t!) 是常数项，与参数无关，在优化中可省略。
        因此实际使用：log L = Σ [Y_t * log(λ_t) - λ_t]
        """
        mu, alpha, beta = params

        # 参数约束
        if mu <= 1e-8 or alpha < 0 or beta < 0:
            return 1e10
        if alpha + beta >= 0.999:  # 平稳性条件：α + β < 1
            return 1e10 + (alpha + beta - 0.999) * 1e6

        try:
            lambda_t = self.compute_lambda(mu, alpha, beta)

            # Poisson 对数似然（省略常数项 -log(Y_t!)，因为与参数无关）
            log_lik = np.sum(self.counts * np.log(lambda_t) - lambda_t)

            # 检查数值稳定性
            if np.isnan(log_lik) or np.isinf(log_lik):
                return 1e10

            return -log_lik
        except:
            return 1e10

    def fit(self, method='L-BFGS-B'):
        """
        最大似然估计

        Returns
        -------
        params : dict
            估计结果
        """
        # 初始化
        mean_count = np.mean(self.counts)
        mu_init = mean_count * 0.3
        alpha_init = 0.3
        beta_init = 0.3

        # 参数边界（放宽上界以允许更大的参数值）
        bounds = [
            (1e-8, mean_count * 2),  # mu: 不超过平均率的2倍
            (1e-8, 0.99),            # alpha: 放宽到0.99
            (1e-8, 0.99)             # beta: 放宽到0.99
        ]

        # 多起点优化
        best_result = None
        best_nll = np.inf

        for scale in [0.5, 1.0, 1.5]:
            params_init = [mu_init * scale, alpha_init, beta_init]

            try:
                result = minimize(
                    self.negative_log_likelihood,
                    params_init,
                    method=method,
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False}
                )

                if result.success and result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
            except:
                continue

        # 如果所有优化都失败，返回启发式参数
        if best_result is None:
            return {
                'mu': mean_count,
                'alpha': 0.0,
                'beta': 0.0,
                'branching_ratio': 0.0,
                'log_likelihood': -np.inf,
                'success': False,
                'n_iterations': 0
            }

        mu, alpha, beta = best_result.x
        log_lik = -best_result.fun

        # 计算分支比
        # INGARCH 模型的分支比：ρ = α / (1 - β)
        if beta < 0.999:
            branching_ratio = alpha / (1 - beta)
        else:
            branching_ratio = alpha / 0.001  # 防止除零

        return {
            'mu': mu,
            'alpha': alpha,
            'beta': beta,
            'branching_ratio': branching_ratio,
            'log_likelihood': log_lik,
            'success': best_result.success,
            'n_iterations': best_result.nit if hasattr(best_result, 'nit') else 0
        }

    @staticmethod
    def fit_null(counts):
        """
        拟合零模型（独立同分布 Poisson）

        零假设 H0: α = 0, β = 0
        此时模型退化为 Y_t ~ Poisson(μ)，其中 μ 是常数

        Parameters
        ----------
        counts : np.ndarray
            每日事件计数序列

        Returns
        -------
        result : dict
            包含 mu 和 log_likelihood
        """
        counts = np.asarray(counts).flatten()
        mu = np.mean(counts)

        # 与全模型保持一致：省略常数项 -log(Y_t!)
        # 因为 LRT 是差值，常数项抵消，不影响检验结果
        log_lik = np.sum(counts * np.log(mu) - mu)

        return {'mu': mu, 'log_likelihood': log_lik}

    @staticmethod
    def likelihood_ratio_test(log_lik_full, log_lik_null):
        """
        似然比检验（边界问题修正）

        零假设 H0: alpha = 0, beta = 0
        备择假设 H1: alpha > 0 或 beta > 0

        由于零假设位于参数空间边界，标准卡方分布不适用。
        根据 Chernoff 定理，渐近分布为混合分布：

            LRT → 0.25 × χ²(0) + 0.5 × χ²(1) + 0.25 × χ²(2)

        其中 χ²(0) 是退化分布（恒为 0）

        Parameters
        ----------
        log_lik_full : float
            全模型的对数似然
        log_lik_null : float
            零模型的对数似然

        Returns
        -------
        lrt : float
            似然比统计量
        p_value : float
            边界校正后的 p 值
        """
        lrt = -2 * (log_lik_null - log_lik_full)

        if lrt <= 0:
            p_value = 1.0
        else:
            # 混合分布：0.5 * χ²(1) + 0.25 * χ²(2)
            # χ²(0) 贡献为 0，忽略
            p1 = 0.5 * (1 - stats.chi2.cdf(lrt, df=1))
            p2 = 0.25 * (1 - stats.chi2.cdf(lrt, df=2))
            p_value = p1 + p2

        return lrt, p_value


class DiscreteHawkesAnalysis:
    """
    离散时间Hawkes分析类
    负责数据加载、模型拟合、结果汇总
    """

    def __init__(self, jitter_days=0.0001):
        self.loader = ConflictDataLoader(jitter_days=jitter_days)
        self.results = {}

    def get_daily_counts(self, region='palestine_israel'):
        """
        获取每日事件计数（不需要时间抖动）

        Parameters
        ----------
        region : str
            'palestine_israel' 或 'russia_ukraine'

        Returns
        -------
        daily_counts : pd.Series
            每日事件数（按日期索引）
        df_raw : pd.DataFrame
            原始数据
        """
        if region == 'palestine_israel':
            df = self.loader.load_palestine_israel()
        else:
            df = self.loader.load_russia_ukraine()

        if len(df) == 0:
            raise ValueError(f"No events found for region: {region}")

        # 按天聚合，不使用时间抖动
        daily_counts = df.groupby(df['event_date'].dt.date).size()

        # 填充缺失日期
        date_range = pd.date_range(df['event_date'].min(), df['event_date'].max(), freq='D')
        daily_counts = daily_counts.reindex(date_range.date, fill_value=0)

        return daily_counts, df

    def run_analysis(self, region='palestine_israel'):
        """
        运行完整的离散时间Hawkes分析

        Parameters
        ----------
        region : str
            'palestine_israel' 或 'russia_ukraine'

        Returns
        -------
        results : dict
            分析结果
        """
        print(f"\n{'='*70}")
        print(f"Discrete Hawkes Analysis - {region.upper()}")
        print(f"{'='*70}")

        # Step 1: 加载每日计数
        print("\n[Step 1] Loading daily counts...")
        daily_counts, df = self.get_daily_counts(region)

        print(f"  Total days: {len(daily_counts)}")
        print(f"  Total events: {daily_counts.sum():,}")
        print(f"  Mean daily events: {daily_counts.mean():.2f}")
        print(f"  Zero days: {(daily_counts == 0).sum()} ({100*(daily_counts==0).mean():.1f}%)")

        # Step 2: 拟合零模型
        print("\n[Step 2] Fitting null model (i.i.d. Poisson)...")
        null_result = DiscreteHawkesMLE.fit_null(daily_counts.values)
        print(f"  mu: {null_result['mu']:.4f}")
        print(f"  Log-likelihood: {null_result['log_likelihood']:.2f}")

        # Step 3: 拟合离散Hawkes模型
        print("\n[Step 3] Fitting discrete Hawkes model (INGARCH)...")
        model = DiscreteHawkesMLE(daily_counts.values)
        full_result = model.fit()

        print(f"\n  Results:")
        print(f"    mu (background): {full_result['mu']:.6f}")
        print(f"    alpha (contagion from previous day): {full_result['alpha']:.6f}")
        print(f"    beta (autoregressive): {full_result['beta']:.6f}")
        print(f"    Branching ratio (alpha/(1-beta)): {full_result['branching_ratio']:.6f}")
        print(f"    Log-likelihood: {full_result['log_likelihood']:.2f}")

        if not full_result['success']:
            print(f"    Warning: Optimization did not fully converge")

        # Step 4: 似然比检验
        print("\n[Step 4] Likelihood Ratio Test...")
        lrt, p_value = DiscreteHawkesMLE.likelihood_ratio_test(
            full_result['log_likelihood'],
            null_result['log_likelihood']
        )

        print(f"  LRT statistic: {lrt:.4f}")
        print(f"  p-value: {p_value:.6f}")

        if p_value < 0.05:
            print(f"  Conclusion: REJECT H0 - Significant contagion effect detected")
        else:
            print(f"  Conclusion: FAIL TO REJECT H0 - No significant contagion effect")

        # 存储结果
        self.results[region] = {
            'region': region,
            'n_days': len(daily_counts),
            'total_events': int(daily_counts.sum()),
            'mean_daily': daily_counts.mean(),
            'params': full_result,
            'null': null_result,
            'lrt': lrt,
            'p_value': p_value
        }

        return self.results[region]

    def compare_regions(self):
        """
        对比两个地区的分析结果
        """
        if len(self.results) < 2:
            print("Please run analysis for both regions first")
            return

        print(f"\n{'='*70}")
        print("COMPARISON: Palestine-Israel vs Russia-Ukraine")
        print(f"{'='*70}")

        comp_data = []
        for region, res in self.results.items():
            comp_data.append({
                'Region': region.upper(),
                'Days': res['n_days'],
                'Total Events': f"{res['total_events']:,}",
                'Mean Daily': f"{res['mean_daily']:.2f}",
                'mu': f"{res['params']['mu']:.4f}",
                'alpha': f"{res['params']['alpha']:.4f}",
                'beta': f"{res['params']['beta']:.4f}",
                'Branching Ratio': f"{res['params']['branching_ratio']:.4f}",
                'p-value': f"{res['p_value']:.6f}"
            })

        comp_df = pd.DataFrame(comp_data)
        print(comp_df.to_string(index=False))

        print("\nInterpretation of Branching Ratio:")
        for region, res in self.results.items():
            br = res['params']['branching_ratio']
            if br > 0.95:
                interp = "Near-critical/Supercritical: Strong contagion chain, potential for escalation"
            elif br > 0.9:
                interp = "Near-critical: Strong contagion chain"
            elif br > 0.7:
                interp = "Strong contagion: Clear clustering"
            elif br > 0.5:
                interp = "Moderate contagion"
            else:
                interp = "Weak contagion"
            print(f"  {region.upper()}: {br:.4f} -> {interp}")

        return comp_df


def main():
    """主函数"""
    output_dir = Path('./output/hawkes')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("DISCRETE TIME HAWKES ANALYSIS (INGARCH)")
    print("="*70)

    analyzer = DiscreteHawkesAnalysis(jitter_days=0.0001)

    for region in ['palestine_israel', 'russia_ukraine']:
        try:
            analyzer.run_analysis(region=region)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        except Exception as e:
            print(f"Error analyzing {region}: {e}")
            import traceback
            traceback.print_exc()
            return

    analyzer.compare_regions()

    # 保存结果
    results_summary = {
        region: {
            'branching_ratio': res['params']['branching_ratio'],
            'alpha': res['params']['alpha'],
            'beta': res['params']['beta'],
            'mu': res['params']['mu'],
            'p_value': res['p_value'],
            'total_events': res['total_events'],
            'n_days': res['n_days']
        }
        for region, res in analyzer.results.items()
    }

    import json
    with open(output_dir / 'hawkes_results_discrete.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/hawkes_results_discrete.json")


if __name__ == "__main__":
    main()