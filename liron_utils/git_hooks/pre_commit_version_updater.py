#!/usr/bin/env python3
"""
Git pre-commit hook that automatically updates:
1. Version in pyproject.toml to current date (YYYY.MM.DD)
2. Copyright year in LICENSE file to current year
"""

import re
import sys
from datetime import datetime
from pathlib import Path

def update_pyproject_version():
    """Update version in pyproject.toml to current date."""
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.exists():
        print(f"Warning: {pyproject_path} not found")
        return False

    # Get current date in YYYY.MM.DD format
    current_date = datetime.now().strftime("%Y.%m.%d")

    # Read the file
    content = pyproject_path.read_text()

    # Update version line
    updated_content = re.sub(
        r'^version\s*=\s*"[^"]*"',
        f'version = "{current_date}"',
        content,
        flags=re.MULTILINE
    )

    # Check if anything changed
    if content != updated_content:
        pyproject_path.write_text(updated_content)
        print(f"Updated version in pyproject.toml to {current_date}")
        return True

    return False

def update_license_year():
    """Update copyright year in LICENSE file."""
    license_path = Path("LICENSE")

    if not license_path.exists():
        print(f"Warning: {license_path} not found")
        return False

    # Get current year
    current_year = datetime.now().year

    # Read the file
    content = license_path.read_text()

    # Update copyright year
    updated_content = re.sub(
        r'Copyright \(c\) \d{4}',
        f'Copyright (c) {current_year}',
        content
    )

    # Check if anything changed
    if content != updated_content:
        license_path.write_text(updated_content)
        print(f"Updated copyright year in LICENSE to {current_year}")
        return True

    return False

def main():
    """Main function to run both updates."""
    print("Running pre-commit hook: updating version and license...")

    # Track if any files were modified
    files_modified = []

    # Update pyproject.toml version
    if update_pyproject_version():
        files_modified.append("pyproject.toml")

    # Update LICENSE year
    if update_license_year():
        files_modified.append("LICENSE")

    # If files were modified, stage them for the commit
    if files_modified:
        import subprocess
        for file_path in files_modified:
            try:
                subprocess.run(["git", "add", file_path], check=True)
                print(f"Staged {file_path} for commit")
            except subprocess.CalledProcessError as e:
                print(f"Error staging {file_path}: {e}")
                return 1

    print("Pre-commit hook completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
