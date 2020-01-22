import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lambdata-unburied", # Replace with your own username
    version="0.0.3",
    author="unburied",
    author_email="davi86m@gmail.com",
    description="Package to display model interpretation plots",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/unburied/lambdata",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)