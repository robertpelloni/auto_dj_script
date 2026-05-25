import os

def get_version():
    """Reads the version string from VERSION.md, searching in current and parent directories."""
    # Priority: Project Root (where VERSION.md usually is)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, ".."))
    version_file = os.path.join(root_dir, "VERSION.md")

    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            return f.read().strip()

    # Fallback to current dir VERSION.md
    local_version = os.path.join(current_dir, "VERSION.md")
    if os.path.exists(local_version):
        with open(local_version, "r") as f:
            return f.read().strip()

    return "Unknown"

__version__ = get_version()
