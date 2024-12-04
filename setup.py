from setuptools import setup, find_packages

setup(
    name="InSAR-RiskLSTM",
    version="0.1.0",
    description="A framework for railway deformation risk prediction using InSAR data, spatial attention, and LSTM networks.",
    author="Your Name",
    author_email="your_email@example.com",
    url="https://github.com/yourusername/InSAR-RiskLSTM",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.2",
        "pandas>=1.5.3",
        "scikit-learn>=1.2.2",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "h5py>=3.8.0",
        "opencv-python>=4.7.0",
    ],
    extras_require={
        "dev": ["pytest", "flake8", "black"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "train=src.experiments.train:main",
            "evaluate=src.experiments.evaluate:main",
        ]
    },
)
