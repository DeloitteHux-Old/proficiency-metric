"""
An Information Theoretic Metric for Multi-Class Categorization
"""
import sys

from setuptools import find_packages, setup


install_requires = [
    "cffi>=1.6.0,<2.0"
]
if sys.version_info[:2] < (2, 7):
    install_requires.extend(["argparse", "ordereddict"])

setup(
    name="predeval",
    version="0.1.0",
    author="Magnetic Engineering",
    author_email="engineering@magnetic.com",
    description=__doc__.strip().splitlines()[0],
    long_description=__doc__,
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=install_requires,
)
