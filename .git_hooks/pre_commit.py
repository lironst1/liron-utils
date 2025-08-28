#!/usr/bin/env python3
import re
import subprocess
from datetime import datetime
from pathlib import Path


def update_init_version(version: str, file: Path = Path("liron_utils/__init__.py")):
    content = file.read_text()
    new_content, n = re.subn(
            r'__version__\s*=\s*["\']\d{4}\.\d{1,2}\.\d{1,2}["\']',
            f'__version__ = "{version}"',
            content
    )
    if n:
        file.write_text(new_content)
        subprocess.run(['git', 'add', str(file)])
        print(f"Updated {file} version to {version}")
    else:
        print(f"No version string found or already up to date for {file}.")


def update_pyproject_version(version: str, file: Path = Path("pyproject.toml")):
    content = file.read_text()
    new_content, n = re.subn(
            r'version\s*=\s*["\']\d{4}\.\d{1,2}\.\d{1,2}["\']',
            f'version = "{version}"',
            content
    )
    if n:
        file.write_text(new_content)
        subprocess.run(['git', 'add', str(file)])
        print(f"Updated {file} version to {version}")
    else:
        print(f"No version string found or already up to date for {file}.")


def update_license_year(year: str, file: Path = Path("LICENSE")):
    content = file.read_text()
    new_content, n = re.subn(
            r'Copyright \(c\) \d{4}',
            f'Copyright (c) {year}',
            content
    )
    if n:
        file.write_text(new_content)
        subprocess.run(['git', 'add', str(file)])
        print(f"Updated {file} year to {year}")
    else:
        print(f"No year string found or already up to date for {file}.")


if __name__ == "__main__":
    now = datetime.now()
    update_init_version(version=now.strftime("%Y.%m.%d"))
    update_pyproject_version(version=now.strftime("%Y.%m.%d"))
    update_license_year(year=now.strftime("%Y"))
