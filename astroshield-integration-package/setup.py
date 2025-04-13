from setuptools import setup, find_packages

setup(
    name="asttroshield-udl-integration",
    version="0.1.0",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        "requests>=2.25.0",
        "pyyaml>=5.4.0",
        "confluent-kafka>=1.5.0",
        "python-dotenv>=0.15.0",
        "prometheus-client>=0.9.0",
    ],
    entry_points={
        'console_scripts': [
            'udl-integration=asttroshield.udl_integration.integration:main',
        ],
    },
) 