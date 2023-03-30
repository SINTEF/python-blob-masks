from setuptools import find_packages, setup

setup(
    name="blob-masks",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.2",
        "opencv-python>=4.7.0.72",
        "perlin-noise>=1.12",
        "Pillow>=9.4.0",
    ],
    author="Antoine Pultier",
    author_email="antoine.pultier@sintef.no",
    description="Generate white potato shaped blobs on black backgrounds",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SINTEF/python-blob-masks",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: WTFPL",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
