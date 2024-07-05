import pkg_resources
from setuptools import setup
from pathlib import Path


setup(
    name = "musicnet",
    version = "0.0.1",
    author="Leszek Wiesner",
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            Path(__file__).with_name("requirements.txt").open()
        )
    ],
)