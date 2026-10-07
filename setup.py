from setuptools import setup, find_packages

setup(
    name="w2sg_uncertainty",
    version="0.1.0",
    description="Weak-to-Strong Preference Optimization: WDPO & CWPO baselines",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "trl>=0.8.6",
        "datasets>=2.18.0",
        "accelerate>=0.27.0",
        "peft>=0.10.0",
        "wandb>=0.16.0",
        "omegaconf>=2.3.0",
        "numpy>=1.24.0",
        "openai>=1.0.0",
    ],
)
