"""
可视化模块（向后兼容版）
新代码请使用以下模块：
- from .basic_plots import *
- from .learning_plots import *
- from .evolution_plots import *
- from .exhibition import *
"""

from .basic_plots import (
    plot_dose_response_curve,
    plot_posterior_distribution,
    plot_all_posteriors,
    plot_simulation_results,
    save_figure,
)
from .learning_plots import (
    plot_acquisition_function,
    plot_learning_process,
    plot_learning_trajectory,
    plot_error_distribution,
    plot_measurement_count,
    plot_true_vs_estimated,
    plot_acquisition_curve,
    plot_summary_report,
    plot_all_figures,
)
from .evolution_plots import (
    plot_posterior_evolution_standalone,
    plot_uncertainty_shrinkage_standalone,
)
from .exhibition import ProjectExhibitionSuite

__all__ = [
    'plot_dose_response_curve',
    'plot_acquisition_function',
    'plot_learning_process',
    'plot_posterior_distribution',
    'plot_all_posteriors',
    'plot_simulation_results',
    'save_figure',
    'plot_learning_trajectory',
    'plot_error_distribution',
    'plot_measurement_count',
    'plot_true_vs_estimated',
    'plot_acquisition_curve',
    'plot_summary_report',
    'plot_all_figures',
    'plot_posterior_evolution_standalone',
    'plot_uncertainty_shrinkage_standalone',
    'ProjectExhibitionSuite',
]
