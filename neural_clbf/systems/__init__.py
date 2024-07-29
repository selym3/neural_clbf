from warnings import warn

from .control_affine_system import ControlAffineSystem
from .observable_system import ObservableSystem
from .planar_lidar_system import PlanarLidarSystem
from .quad2d import Quad2D
from .quad3d import Quad3D
from .neural_lander import NeuralLander
from .inverted_pendulum import InvertedPendulum
from .kinematic_single_track_car import KSCar
from .single_track_car import STCar
from .segway import Segway
from .turtlebot import TurtleBot
from .turtlebot_2d import TurtleBot2D
from .linear_satellite import LinearSatellite
from .single_integrator_2d import SingleIntegrator2D
from .autorally import AutoRally
from .point_system import Point
from .point_wind_system import PointInWind
from .point_linear import LinearWind
from .x_point_system import XPoint
from .point_circular_wind_system import PointInCirWind
from .xpoint_circular_system import XCirPoint
from .xpoint_simple_wind import XPointSim
from .simple_balloon import SimpleBalloon
from .xpoint_linear_wind import XLinPoint
from .xpoint_layer_wind import XLayPoint
from .xpoint_observed import XLayObsPoint

__all__ = [
    "ControlAffineSystem",
    "ObservableSystem",
    "PlanarLidarSystem",
    "InvertedPendulum",
    "Quad2D",
    "Quad3D",
    "NeuralLander",
    "KSCar",
    "STCar",
    "TurtleBot",
    "TurtleBot2D",
    "Segway",
    "LinearSatellite",
    "SingleIntegrator2D",
    "AutoRally",
    "Point",
    "PointInWind",
    "XPoint",
    "LinearWind",
    "PointInCirWind",
    "XCirPoint",
    "XPointSim",
    "SimpleBalloon",
    "XLinPoint",
    "XLayPoint",
    "XLayObsPoint",
]

try:
    from .f16 import F16  # noqa

    __all__.append("F16")
except ImportError:
    warn("Could not import F16 module; is AeroBench installed")
