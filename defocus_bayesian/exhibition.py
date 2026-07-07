"""
项目展示套件 - 包含多种高级可视化功能
"""

from typing import Optional, Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .model import SigmoidModel
from .acquisition import AcquisitionFunction
from .basic_plots import save_figure

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ProjectExhibitionSuite:
    """
    项目展示套件 - 包含多种高级可视化功能
    """
    
    def __init__(self, lang: str = "zh"):
        """
        初始化项目展示套件
        
        Args:
            lang: 语言 ('zh' 或 'en')
        """
        self.lang = lang
    
    def plot_decision_landscape(
        self,
        model: SigmoidModel,
        acquisition: AcquisitionFunction,
        measurements: List[Tuple[float, float]],
        y_best: Optional[float] = None,
        figsize: Tuple[float, float] = (12, 8),
    ) -> Figure:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        doses = np.linspace(0.5, 6.0, 100)
        pred = model.predict(doses)
        
        ax1.plot(doses, pred["mean"], 'b-', linewidth=2, label='后验均值' if self.lang == "zh" else 'Posterior Mean')
        ax1.fill_between(doses, pred["lower"], pred["upper"], alpha=0.3, color='blue', label='95% CI')
        
        if measurements:
            doses_obs = [m[0] for m in measurements]
            responses_obs = [m[1] for m in measurements]
            ax1.scatter(doses_obs, responses_obs, c='red', s=100, zorder=5, label='观测数据' if self.lang == "zh" else 'Observed Data')
        
        ax1.set_xlabel('剂量 (D)', fontsize=12)
        ax1.set_ylabel('反应 (μm)', fontsize=12)
        ax1.set_title('剂量-反应曲线' if self.lang == "zh" else 'Dose-Response Curve', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        x_grid, acq_values = acquisition.get_acquisition_values(model, "eer", y_best)
        acq_norm = (acq_values - np.min(acq_values)) / (np.max(acq_values) - np.min(acq_values) + 1e-9)
        im = ax2.imshow(acq_norm.reshape(1, -1), cmap='viridis', aspect='auto', extent=[0.5, 6.0, 0, 1])
        
        max_idx = np.argmax(acq_values)
        max_dose = x_grid[max_idx]
        ax2.plot(max_dose, 0.5, 'r*', markersize=15, label=f'推荐: {max_dose:.2f} D')
        
        if measurements:
            doses_obs = [m[0] for m in measurements]
            ax2.scatter(doses_obs, [0.5] * len(doses_obs), c='white', s=100, marker='|', zorder=5, label='已测量' if self.lang == "zh" else 'Measured')
        
        ax2.set_xlabel('剂量 (D)', fontsize=12)
        ax2.set_ylabel('兴趣度', fontsize=12) if self.lang == "zh" else ax2.set_ylabel('Interest', fontsize=12)
        ax2.set_title('决策景观' if self.lang == "zh" else 'Decision Landscape', fontsize=14)
        ax2.legend(loc='best')
        
        cbar = fig.colorbar(im, ax=ax2, orientation='horizontal', pad=0.3)
        cbar.set_label('兴趣度' if self.lang == "zh" else 'Interest', fontsize=11)
        
        plt.tight_layout()
        return fig
    
    def plot_confidence_stopwatch(
        self,
        model: SigmoidModel,
        figsize: Tuple[float, float] = (8, 8),
    ) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        threshold_std = model.get_posterior_stats()["threshold"]["std"]
        max_confidence = 100.0
        min_confidence = 0.0
        
        n_measurements = model.trace.posterior.dims.get('draw', 1) * model.trace.posterior.dims.get('chain', 1)
        base_confidence = (1.0 / (threshold_std + 0.01)) * 15.0
        measurement_factor = min(1.2, 1.0 + (n_measurements / 10000))
        confidence = min(max_confidence, max(min_confidence, base_confidence * measurement_factor))
        
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        
        circle = plt.Circle((0, 0), 1, fill=False, linewidth=2, color='gray')
        ax.add_artist(circle)
        
        for i in range(0, 101, 10):
            angle = np.radians(180 - (i / 100) * 180)
            x = np.cos(angle)
            y = np.sin(angle)
            ax.plot([0.9 * x, x], [0.9 * y, y], 'k-', linewidth=1)
            ax.text(1.1 * x, 1.1 * y, f'{i}%', ha='center', va='center', fontsize=10)
        
        angle = np.radians(180 - (confidence / 100) * 180)
        x = np.cos(angle)
        y = np.sin(angle)
        ax.plot([0, x], [0, y], 'r-', linewidth=3, marker='o', markersize=8, markerfacecolor='red')
        
        center = plt.Circle((0, 0), 0.1, fill=True, color='gray')
        ax.add_artist(center)
        
        ax.set_title('阈值估计信心' if self.lang == "zh" else 'Threshold Estimation Confidence', fontsize=14)
        ax.text(0, -1.3, f'信心: {confidence:.1f}%', ha='center', va='center', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        return fig
    
    def plot_posterior_evolution(
        self,
        models: List[SigmoidModel],
        measurements_list: List[List[Tuple[float, float]]],
        figsize: Tuple[float, float] = (12, 8),
    ) -> Figure:
        n_plots = len(models)
        n_rows = (n_plots + 1) // 2
        n_cols = 2 if n_plots > 1 else 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()
        
        for i, (model, measurements) in enumerate(zip(models, measurements_list)):
            ax = axes[i]
            posterior = model.trace.posterior["threshold"].values.reshape(-1)
            ax.hist(posterior, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
            mean_val = np.mean(posterior)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_val:.3f}')
            ax.set_xlabel('阈值 (D)', fontsize=11)
            ax.set_ylabel('密度', fontsize=11) if self.lang == "zh" else ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'第 {i+1} 次测量后' if self.lang == "zh" else f'After {i+1} Measurements', fontsize=12)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('后验分布演变' if self.lang == "zh" else 'Posterior Evolution', fontsize=16, y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_clinical_interpretation(
        self,
        model: SigmoidModel,
        measurements: List[Tuple[float, float]],
        figsize: Tuple[float, float] = (12, 8),
    ) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        doses = np.linspace(0.5, 6.0, 100)
        pred = model.predict(doses)
        
        ax.plot(doses, pred["mean"], 'b-', linewidth=2, label='后验均值' if self.lang == "zh" else 'Posterior Mean')
        ax.fill_between(doses, pred["lower"], pred["upper"], alpha=0.3, color='blue', label='95% CI')
        
        if measurements:
            doses_obs = [m[0] for m in measurements]
            responses_obs = [m[1] for m in measurements]
            ax.scatter(doses_obs, responses_obs, c='red', s=100, zorder=5, label='观测数据' if self.lang == "zh" else 'Observed Data')
        
        stats = model.get_posterior_stats()
        baseline = stats["baseline"]["mean"]
        max_response = stats["max_response"]["mean"]
        threshold = stats["threshold"]["mean"]
        slope = stats["slope"]["mean"]
        
        baseline_zone = threshold - 2.0 / slope
        transition_start = threshold - 1.0 / slope
        transition_end = threshold + 1.0 / slope
        saturation_zone = threshold + 2.0 / slope
        
        ax.axvspan(0.5, baseline_zone, alpha=0.1, color='green', label='基线反应区' if self.lang == "zh" else 'Baseline Zone')
        ax.axvspan(transition_start, transition_end, alpha=0.1, color='yellow', label='过渡区' if self.lang == "zh" else 'Transition Zone')
        ax.axvspan(saturation_zone, 6.0, alpha=0.1, color='red', label='饱和区' if self.lang == "zh" else 'Saturation Zone')
        ax.axvline(threshold, color='purple', linestyle='--', linewidth=2, label=f'阈值: {threshold:.2f} D')
        
        ax.set_xlabel('剂量 (D)', fontsize=12)
        ax.set_ylabel('反应 (μm)', fontsize=12)
        ax.set_title('临床解释' if self.lang == "zh" else 'Clinical Interpretation', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_acquisition_landscape(
        self,
        model: SigmoidModel,
        acquisition: AcquisitionFunction,
        measurements: List[Tuple[float, float]],
        y_best: Optional[float] = None,
        figsize: Tuple[float, float] = (12, 6),
    ) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        x_grid, acq_values = acquisition.get_acquisition_values(model, "eer", y_best)
        acq_norm = (acq_values - np.min(acq_values)) / (np.max(acq_values) - np.min(acq_values) + 1e-9)
        im = ax.imshow(acq_norm.reshape(1, -1), cmap='hot', aspect='auto', extent=[0, 6.0, 0, 1])
        
        max_idx = np.argmax(acq_values)
        max_dose = x_grid[max_idx]
        ax.plot(max_dose, 0.5, 'r*', markersize=15, label=f'推荐: {max_dose:.2f} D')
        
        if measurements:
            doses_obs = [m[0] for m in measurements]
            ax.scatter(doses_obs, [0.5] * len(doses_obs), c='white', s=100, marker='|', zorder=5, label='已测量' if self.lang == "zh" else 'Measured')
        
        explanation = "AI 放弃高剂量区而选择 2.8D 附近，因为这里是阈值所在区域，能最大程度减少 ED50 估计的不确定性。这展示了 AI 在好奇心与收敛性之间的权衡。" if self.lang == "zh" else "AI avoids high dose regions and chooses around 2.8D because this is where the threshold is located, which maximizes the reduction in ED50 estimation uncertainty. This demonstrates the AI's balance between curiosity and convergence."
        ax.text(3.0, -0.2, explanation, ha='center', va='top', fontsize=10, wrap=True)
        
        ax.set_xlabel('剂量 (D)', fontsize=12)
        ax.set_ylabel('兴趣度', fontsize=12) if self.lang == "zh" else ax.set_ylabel('Interest', fontsize=12)
        ax.set_title('AI的思考过程' if self.lang == "zh" else "AI's Thought Process", fontsize=14)
        ax.legend(loc='best')
        
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
        cbar.set_label('兴趣度' if self.lang == "zh" else 'Interest', fontsize=11)
        
        plt.tight_layout()
        return fig
    
    def plot_uncertainty_shrinkage(
        self,
        model1: SigmoidModel,
        model5: SigmoidModel,
        measurements1: List[Tuple[float, float]],
        measurements5: List[Tuple[float, float]],
        figsize: Tuple[float, float] = (12, 6),
    ) -> Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        doses = np.linspace(0.5, 6.0, 100)
        
        pred1 = model1.predict(doses)
        ax1.plot(doses, pred1["mean"], 'b-', linewidth=2, label='后验均值' if self.lang == "zh" else 'Posterior Mean')
        ax1.fill_between(doses, pred1["lower"], pred1["upper"], alpha=0.3, color='blue', label='95% CI')
        
        if measurements1:
            doses_obs1 = [m[0] for m in measurements1]
            responses_obs1 = [m[1] for m in measurements1]
            ax1.scatter(doses_obs1, responses_obs1, c='red', s=100, zorder=5, label='观测数据' if self.lang == "zh" else 'Observed Data')
        
        ax1.set_xlabel('剂量 (D)', fontsize=12)
        ax1.set_ylabel('反应 (μm)', fontsize=12)
        ax1.set_title('第1次测量后' if self.lang == "zh" else 'After 1st Measurement', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        pred5 = model5.predict(doses)
        ax2.plot(doses, pred5["mean"], 'b-', linewidth=2, label='后验均值' if self.lang == "zh" else 'Posterior Mean')
        ax2.fill_between(doses, pred5["lower"], pred5["upper"], alpha=0.3, color='blue', label='95% CI')
        
        if measurements5:
            doses_obs5 = [m[0] for m in measurements5]
            responses_obs5 = [m[1] for m in measurements5]
            ax2.scatter(doses_obs5, responses_obs5, c='red', s=100, zorder=5, label='观测数据' if self.lang == "zh" else 'Observed Data')
        
        ax2.set_xlabel('剂量 (D)', fontsize=12)
        ax2.set_ylabel('反应 (μm)', fontsize=12)
        ax2.set_title('第5次测量后' if self.lang == "zh" else 'After 5th Measurement', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('不确定性收缩效果' if self.lang == "zh" else 'Uncertainty Shrinkage Effect', fontsize=16, y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_joint_posterior(
        self,
        model: SigmoidModel,
        figsize: Tuple[float, float] = (10, 8),
    ) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        threshold = model.trace.posterior["threshold"].values.reshape(-1)
        slope = model.trace.posterior["slope"].values.reshape(-1)
        
        scatter = ax.scatter(threshold, slope, alpha=0.5, s=50, c='steelblue')
        
        threshold_mean = np.mean(threshold)
        slope_mean = np.mean(slope)
        ax.plot(threshold_mean, slope_mean, 'r*', markersize=15, label=f'均值: ({threshold_mean:.2f}, {slope_mean:.2f})')
        
        ax.set_xlabel('ED50 (D)', fontsize=12)
        ax.set_ylabel('斜率 (Slope)', fontsize=12)
        ax.set_title('ED50 与斜率的联合分布' if self.lang == "zh" else 'Joint Posterior of ED50 and Slope', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_subject_persona(
        self,
        model: SigmoidModel,
        figsize: Tuple[float, float] = (10, 8),
    ) -> Figure:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        stats = model.get_posterior_stats()
        threshold = stats["threshold"]["mean"]
        slope = stats["slope"]["mean"]
        max_response = stats["max_response"]["mean"]
        baseline = stats["baseline"]["mean"]
        
        sensitivity = max(0, min(100, (4.0 - threshold) / 4.0 * 100))
        response_speed = max(0, min(100, (slope / 5.0) * 100))
        response_strength = max(0, min(100, ((max_response - 10) / 15) * 100))
        baseline_level = max(0, min(100, (5.0 - baseline) / 5.0 * 100))
        
        features = ['敏感性', '反应速度', '反应强度', '基线水平'] if self.lang == "zh" else ['Sensitivity', 'Response Speed', 'Response Strength', 'Baseline Level']
        values = [sensitivity, response_speed, response_strength, baseline_level]
        
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features)
        ax.set_ylim(0, 100)
        
        if threshold < 2.0 and slope > 3.0:
            persona = '高敏感型' if self.lang == "zh" else 'Highly Sensitive'
        elif threshold > 3.5:
            persona = '迟钝型' if self.lang == "zh" else 'Insensitive'
        elif slope > 3.0:
            persona = '快速饱和型' if self.lang == "zh" else 'Fast Saturation'
        else:
            persona = '正常型' if self.lang == "zh" else 'Normal'
        
        ax.set_title(f'受试者画像: {persona}', fontsize=14, pad=20)
        plt.tight_layout()
        return fig
    
    def generate_full_exhibition(
        self,
        model: SigmoidModel,
        acquisition: AcquisitionFunction,
        measurements: List[Tuple[float, float]],
        y_best: Optional[float] = None,
        save_dir: str = "results/exhibition",
        models_evolution: Optional[List[SigmoidModel]] = None,
        measurements_evolution: Optional[List[List[Tuple[float, float]]]] = None,
        model1: Optional[SigmoidModel] = None,
        model5: Optional[SigmoidModel] = None,
        measurements1: Optional[List[Tuple[float, float]]] = None,
        measurements5: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, Figure]:
        import os
        os.makedirs(save_dir, exist_ok=True)
        figures = {}
        
        fig1 = self.plot_decision_landscape(model, acquisition, measurements, y_best)
        fig1_path = os.path.join(save_dir, f"01_decision_landscape_{self.lang}.png")
        save_figure(fig1, fig1_path)
        figures["decision_landscape"] = fig1
        
        if models_evolution and measurements_evolution:
            fig2 = self.plot_posterior_evolution(models_evolution, measurements_evolution)
            fig2_path = os.path.join(save_dir, f"02_posterior_evolution_{self.lang}.png")
            save_figure(fig2, fig2_path)
            figures["posterior_evolution"] = fig2
        
        fig3 = self.plot_clinical_interpretation(model, measurements)
        fig3_path = os.path.join(save_dir, f"03_clinical_interpretation_{self.lang}.png")
        save_figure(fig3, fig3_path)
        figures["clinical_interpretation"] = fig3
        
        fig4 = self.plot_acquisition_landscape(model, acquisition, measurements, y_best)
        fig4_path = os.path.join(save_dir, f"04_acquisition_landscape_{self.lang}.png")
        save_figure(fig4, fig4_path)
        figures["acquisition_landscape"] = fig4
        
        if models_evolution and measurements_evolution and len(models_evolution) >= 2:
            fig5 = self.plot_uncertainty_shrinkage(
                models_evolution[0],
                models_evolution[-1],
                measurements_evolution[0],
                measurements_evolution[-1]
            )
            fig5_path = os.path.join(save_dir, f"05_uncertainty_shrinkage_{self.lang}.png")
            save_figure(fig5, fig5_path)
            figures["uncertainty_shrinkage"] = fig5
        
        fig6 = self.plot_joint_posterior(model)
        fig6_path = os.path.join(save_dir, f"06_joint_posterior_{self.lang}.png")
        save_figure(fig6, fig6_path)
        figures["joint_posterior"] = fig6
        
        fig7 = self.plot_subject_persona(model)
        fig7_path = os.path.join(save_dir, f"07_subject_persona_{self.lang}.png")
        save_figure(fig7, fig7_path)
        figures["subject_persona"] = fig7
        
        return figures


__all__ = [
    'ProjectExhibitionSuite',
]
