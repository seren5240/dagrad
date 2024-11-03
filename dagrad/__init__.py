import importlib
import subprocess
import sys
import warnings

from .core import dagrad, notears, topo, dagma
from .utils.utils import (
    is_dag,
    threshold_till_dag,
    simulate_dag,
    simulate_linear_sem,
    simulate_nonlinear_sem,
    set_random_seed,
    simulate_parameter,
    count_accuracy,
    generate_linear_data,
    generate_nonlinear_data
)

# Define what should be accessible when the package is imported
__all__ = [
    'dagrad', 'topo', 'notears', 'dagma',
    'is_dag', 'threshold_till_dag', 'simulate_dag', 'simulate_linear_sem', 'simulate_nonlinear_sem',
    'set_random_seed', 'simulate_parameter', 'count_accuracy', 'generate_linear_data', 'generate_nonlinear_data'
]


def install_notears():
    repo_url = "https://github.com/xunzheng/notears.git"
    try:
        # Check if 'notears.linear' is installed
        importlib.import_module('notears.linear')
    except ModuleNotFoundError or ImportError:
        # If not installed, attempt to install it from GitHub
        print("Module 'notears' not found.")
        print("Installing 'notears' from GitHub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+" + repo_url])

        # Force the reload of the module after installation
        importlib.invalidate_caches()  # Clear import cache
        try:
            # Now reload the entire 'notears' module
            notears = importlib.reload(importlib.import_module('notears'))
            importlib.import_module('notears.linear')  # Import the submodule again
        except ModuleNotFoundError or ImportError:
            warnings.warn(f"Installation failed or 'notears' still not found. "
                          f"Please install manually using: pip install git+{repo_url}")
            raise

# Call this during your package initialization
install_notears()


