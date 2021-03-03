import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rl-projects",
    version="0.0.1",
    author="Jakob Stigenberg",
    description="Some RL stuff",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jakkes/RL_Projects",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "torch==1.7.1"
    ]
)

### Publish
### python3 setup.py sdist bdist_wheel
### twine upload dist/*