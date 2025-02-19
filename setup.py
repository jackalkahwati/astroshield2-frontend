from setuptools import setup, find_packages

setup(
    name="asttroshield",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "typing-extensions>=4.7.1",
        "urllib3>=2.0.7",
        "numpy>=1.21.0",  # Required for indicator models
    ],
    extras_require={
        'test': [
            'pytest>=7.3.1',
            'pytest-cov>=4.1.0',  # For coverage reporting
            'pytest-html>=3.2.0',  # For HTML test reports
        ],
    },
    python_requires=">=3.7",
    test_suite='tests',
) 