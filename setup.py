import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ai",
    version="0.0.4",
    author="Jakob Stigenberg",
    description="Collection of AI algorithms and agents.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jakkes/AI_Projects",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "torch~=1.9.0",
        "scipy~=1.6.1",
        "tensorboard~=2.4.1",
        "gym~=0.18.0",
        "typed-argument-parser~=1.6.2",
        "python-linq==2.0.3"
    ],
    python_requires=">=3.7, <3.9",
)

# Publish
# python setup.py sdist bdist_wheel
# twine upload dist/*
