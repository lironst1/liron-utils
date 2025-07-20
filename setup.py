import setuptools

__ver__ = "1.0"
__details__ = {
	"author":       "Liron Stettiner",
	"email":        "lironst1@gmail.com",
	"url":          "https://github.com/lironst1/Home",
	"requirements": ["numpy", "scipy", "matplotlib", "natsort", "pandas", "sympy", "scikit-learn", "audioread",
		"soundfile", "manim", "uncertainties", "pytube", "PyPDF2"]
}

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
