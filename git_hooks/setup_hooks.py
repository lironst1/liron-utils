#!/usr/bin/env python3
"""
Setup script for git hooks that automatically update version and license.
Run this script from the liron-utils directory to install or update the git hooks.
"""

import shutil
import stat
from pathlib import Path


def setup_git_hooks():
    """Copy git hooks from git_hooks/ to .git/hooks/ and make them executable."""

    # Define paths - working from liron-utils directory
    project_root = Path(__file__).parent.parent  # liron-utils directory
    git_hooks_source = project_root / "git_hooks"
    git_hooks_dest = project_root / ".git" / "hooks"

    # Ensure .git/hooks directory exists
    git_hooks_dest.mkdir(parents=True, exist_ok=True)

    # Map of source files to destination hook names
    hooks_to_install = {
        "pre_commit_version_updater.py": "pre-commit"
    }

    installed_hooks = []

    for source_file, hook_name in hooks_to_install.items():
        source_path = git_hooks_source / source_file
        dest_path = git_hooks_dest / hook_name

        if source_path.exists():
            # Copy the file
            shutil.copy2(source_path, dest_path)

            # Make it executable
            current_permissions = dest_path.stat().st_mode
            dest_path.chmod(current_permissions | stat.S_IEXEC)

            installed_hooks.append(hook_name)
            print(f"✅ Installed {hook_name} hook")
        else:
            print(f"❌ Source file not found: {source_path}")

    if installed_hooks:
        print(f"\n🎉 Successfully installed {len(installed_hooks)} git hook(s):")
        for hook in installed_hooks:
            print(f"   - {hook}")

        print("\nThese hooks will now automatically:")
        print("   - Update version in pyproject.toml to current date (YYYY.MM.DD)")
        print("   - Update copyright year in LICENSE to current year")
        print("   - Stage the updated files for commit")

        print("\nTo test the hooks, make a commit in the liron-utils directory!")
        print("Example: git add . && git commit -m 'Test automatic version update'")
    else:
        print("❌ No hooks were installed")


if __name__ == "__main__":
    print("Setting up git hooks for liron-utils package...")
    setup_git_hooks()
