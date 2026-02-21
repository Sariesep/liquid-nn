from setuptools import setup, find_packages

setup(
    name="liquidnn",
    version="0.1.0",
    description="Liquid Neural Network Language Model with Hebbian Plasticity",
    author="Eray",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "tiktoken>=0.5",
        "pyyaml>=6.0",
    ],
    extras_require={
        "data": ["datasets>=2.14"],
        "dev": ["pytest>=7.0"],
    },
)
