#!/usr/bin/env python3
"""
T-Detect Setup Script

Simple setup for T-Detect: Tail-Aware Statistical Normalization for Robust AI Text Detection
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="t-detect",
    version="1.0.0",
    author="T-Detect Authors",
    author_email="contact@t-detect.org",
    description="Tail-Aware Statistical Normalization for Robust AI Text Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/t-detect",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        "demo": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "jupyter>=1.0.0",
        ],
        "api": [
            "openai>=1.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "t-detect=scripts.demo:main",
            "t-detect-test=test_t_detect:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-org/t-detect/issues",
        "Source": "https://github.com/your-org/t-detect",
        "Documentation": "https://github.com/your-org/t-detect#readme",
    },
    keywords="AI detection, text classification, adversarial robustness, machine learning, NLP",
)