# robots/models/__init__.py
"""
Robot model implementations.
Each robot should be defined in its own module and inherit from MobileRobot.
"""
from .Barakuda import Barakuda, BarakudaConfig
from .Husky import Husky, HuskyConfig
from .Aru import Aru, AruConfig

__all__ = ["Barakuda", "BarakudaConfig", "Husky", "HuskyConfig", "Aru", "AruConfig"]
