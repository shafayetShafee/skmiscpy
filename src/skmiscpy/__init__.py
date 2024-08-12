from importlib.metadata import version

__version__ = version("skmiscpy")

from skmiscpy.utils import here
from skmiscpy.cbs import compute_smd
from skmiscpy.plotting import plot_smd
from skmiscpy.plotting import plot_mirror_histogram
