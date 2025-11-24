from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import tomlkit

def get_version():
    try:
        # First, try to get the version from installed package metadata
        return version("qLDPCsim")
    except PackageNotFoundError:
        # Fallback: parse pyproject.toml directly for development
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "r") as f:
                pyproject = tomlkit.load(f)
                return pyproject["project"]["version"]
        return "0.0.0-dev"  # Default fallback

__version__ = get_version()   