from setuptools import setup, find_packages

setup(
    name="synthgen",
    version="0.1",
    description="A Python package for synthetic data generation",
    author="Robin Jacobs",
    author_email="robin@jacobs.swiss",
    packages=find_packages(),
    install_requires=["opencv-python", "scipy", "matplotlib", "numpy", "trimesh"],
    classifiers=[
        "Development Status :: 1 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
