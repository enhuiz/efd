from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="easyfd",
    python_requires=">=3.6.0",
    version="0.0.1",
    description="EasyFD: Easy Face Detection",
    author="enhuiz",
    author_email="niuzhe.nz@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["easyfd"],
    install_requires=[
        "torch",
        "numpy",
    ],
    url="https://github.com/enhuiz/easyfd",
)
