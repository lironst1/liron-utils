#!/usr/bin/env python3
"""
Setup script for git hooks.
Run this script from the parent directory to install or update the git hooks:
`python .git_hooks/setup_hooks.py`
"""

import shutil
import stat
from pathlib import Path

# Map of source files to destination hook names
hooks_to_install = {
    "pre-commit": "pre_commit.py"
}  # {hook_name: source_file}


def setup_git_hooks():
    """Copy git hooks from git_hooks/ to .git/hooks/ and make them executable."""

    # Define paths - working from parent directory
    project_root = Path(__file__).parent.parent  # parent directory
    src = project_root / ".git_hooks"
    dest = project_root / ".git" / "hooks"

    # Ensure .git/hooks directory exists
    dest.mkdir(parents=True, exist_ok=True)

    for hook_name, file_src in hooks_to_install.items():
        path_src = src / file_src
        path_dest = dest / hook_name

        if path_src.exists():
            # Copy the file
            shutil.copy2(path_src, path_dest)

            # Make it executable
            current_permissions = path_dest.stat().st_mode
            path_dest.chmod(current_permissions | stat.S_IEXEC)
            print(f"✅ Installed {hook_name} hook")
        else:
            print(f"❌ Source file not found: {path_src}")


if __name__ == "__main__":
    print("Setting up git hooks...")
    setup_git_hooks()
    print("Git hook setup complete.")
