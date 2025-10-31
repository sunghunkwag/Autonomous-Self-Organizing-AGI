"""
Setup script for ASAGI (Autonomous Self-Organizing AGI)
This file provides backward compatibility for older pip versions.
Modern installations should use pyproject.toml.
"""

from setuptools import setup, find_packages

# Read version from pyproject.toml
VERSION = "0.1.0"

# Read long description from README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="asagi",
    version=VERSION,
    description="Autonomous Self-Organizing AGI: reward-free, self-goal-setting architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ASAGI Dev",
    author_email="dev@example.com",
    url="https://github.com/sunghunkwag/Autonomous-Self-Organizing-AGI",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "einops>=0.6.0",
        "opt-einsum>=3.3.0",
        "networkx>=2.8.0",
        "cvxpy>=1.2.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.64.0",
        "Pillow>=9.0.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "full": [
            "mamba-ssm>=1.0.0",
            "torch-uncertainty>=0.2.0",
            "gymnasium>=0.27.0",
            "pybullet>=3.2.0",
            "plotly>=5.0.0",
            "wandb>=0.13.0",
            "tensorboard>=2.8.0",
            "hydra-core>=1.2.0",
            "omegaconf>=2.2.0",
            "memory-profiler>=0.60.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "artificial-intelligence",
        "agi",
        "autonomous-systems",
        "self-organizing",
        "reward-free",
        "intrinsic-motivation",
        "causal-reasoning",
        "meta-learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/sunghunkwag/Autonomous-Self-Organizing-AGI/issues",
        "Source": "https://github.com/sunghunkwag/Autonomous-Self-Organizing-AGI",
        "Changelog": "https://github.com/sunghunkwag/Autonomous-Self-Organizing-AGI/blob/main/CHANGELOG.md",
    },
)
