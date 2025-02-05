from setuptools import find_packages, setup

setup(
    name="kinpfn",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "kinpfn=kinpfn.model:main",
        ],
    },
)
