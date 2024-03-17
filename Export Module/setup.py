import setuptools

from liron_utils import __ver__, __details__

# export liron_utils as a standalone Python module
# setup file must be located at the same parent directory as liron_utils (not inside it)
setuptools.setup(
    include_package_data=True,
    name="liron_utils",
    version=__ver__,

    author=__details__["author"],
    author_email=__details__["email"],
    url=__details__["url"],

    description="Utilities module",

    packages=setuptools.find_packages(),
    py_modules=["liron_utils"],
    install_requires=__details__["requirements"]
)

# 'distclass', 'script_name', 'script_args', 'options',
# 'name', 'version', 'author', 'author_email',
# 'maintainer', 'maintainer_email', 'url', 'license',
# 'description', 'long_description', 'keywords',
# 'platforms', 'classifiers', 'download_url',
# 'requires', 'provides', 'obsoletes',