"""
个性化离焦剂量探索系统（贝叶斯 Sigmoid + 主动学习）

基于贝叶斯 Sigmoid（四参数逻辑斯蒂）模型和主动学习策略，
为每个受试者动态推荐下一个离焦测量剂量。
"""

from .model import SigmoidModel
from .acquisition import AcquisitionFunction
from .learner import SigmoidActiveLearner, LearningResult
from .simulate import SubjectSimulator, SubjectParameters
from .plot import ProjectExhibitionSuite, plot_all_figures
from .config import config, Config

__version__ = "1.0.0"
__all__ = [
    "SigmoidModel",
    "AcquisitionFunction",
    "SigmoidActiveLearner",
    "LearningResult",
    "SubjectSimulator",
    "SubjectParameters",
    "ProjectExhibitionSuite",
    "config",
    "Config",
    "plot_all_figures",
]
