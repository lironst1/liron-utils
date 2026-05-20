import pip


def install(package: str) -> None:
    """Install a Python package via ``pip.main(["install", package])``.

    Args:
        package: Package specifier (e.g. ``"numpy"`` or ``"numpy==1.26"``).
    """
    pip.main(["install", package])
