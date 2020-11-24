from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="easyfd",
    python_requires=">=3.6.0",
    version="0.0.1.dev0",
    description="EasyFD: Easy Face Detection",
    author="enhuiz",
    author_email="niuzhe.nz@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
    ],
    url="https://github.com/enhuiz/easyfd",
    entry_points={
        "console_scripts": [
            "efd-vcrop=easyfd.bin.efd_vcrop:main",
        ],
    },
)
