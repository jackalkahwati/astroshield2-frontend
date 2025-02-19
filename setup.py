from setuptools import setup, find_packages

setup(
    name="asttroshield",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.23.0",
        "certifi>=2024.2.0",
        "urllib3>=2.0.0"
    ],
    python_requires=">=3.8",
) 