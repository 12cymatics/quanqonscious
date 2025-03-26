# QuanQonscious/updater.py

import subprocess
import sys

# List of core dependencies to ensure up-to-date
core_dependencies = ["numpy", "cirq", "mpi4py"]
optional_dependencies = ["cupy", "cuda-quantum-cu12", "numba", "psutil"]

def update_all():
    """
    Update all core and optional dependencies to their latest versions using pip.
    This should be run in a context where internet access is available.
    """
    packages = core_dependencies + optional_dependencies
    print("[Updater] Updating QuanQonscious dependencies to latest versions...")
    for pkg in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkg])
            print(f"[Updater] Successfully updated {pkg}.")
        except Exception as e:
            print(f"[Updater] Warning: Could not update {pkg} ({e}). Continuing...")
    print("[Updater] Update process completed.")

def update_package(package_name: str):
    """
    Update a single dependency by name.
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
        print(f"[Updater] {package_name} is now up-to-date.")
    except Exception as e:
        print(f"[Updater] Failed to update {package_name}: {e}")

# The dynamic updater could also fetch the latest version of QuanQonscious itself from PyPI if needed:
# def update_quanqonscious():
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "QuanQonscious"])
