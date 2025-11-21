# setup.py
from setuptools import setup, find_packages

setup(
    name="concrete_crack_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.5.0",
        "albumentations>=1.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.64.0",
    ],
)