from setuptools import setup, find_packages

setup(
    name='tsdb',           # Name of the package
    version='0.1.0',                      # Version number
    description='Custom library for TowerScout use cases',  # A short description of the package
    author='AIX Development Team',        # AIX ML team
    author_email='ac84@cdc.gov',          # Contact email
    
    packages=find_packages(),             # Automatically find all packages in the directory
    install_requires=[
        'numpy>=1.21.0',                  # Specifies the minimum version of numpy required
        'pandas>=1.3.0',                  # Specifies the minimum version of pandas required
    ],
    python_requires='>=3.7',              # Specify the Python versions supported
    classifiers=[
        'Programming Language :: Python :: 3',  # Python version classifier
    
    ],
    
)
