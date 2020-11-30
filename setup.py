import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepop",
    version="0.1dev",
    author="Kartik Sharma",
    author_email="kartik4949@gmail.com",
    description="A mini Tensor framework for tensor operations on GPU.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kartik4949/deepops",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=["pycuda"],
)
