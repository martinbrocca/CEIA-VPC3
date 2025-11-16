from setuptools import setup, find_packages

setup(
    name="ceia-vpc3",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pillow>=9.5.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "mlflow>=2.3.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)
