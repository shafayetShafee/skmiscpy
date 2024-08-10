from importlib.metadata import version

__version__ = version("skmiscpy")

from skmiscpy.here import here
from skmiscpy.smd import compute_smd
from skmiscpy.plot_smd import plot_smd
from skmiscpy.plot_mirror_histogram import plot_mirror_histogram
