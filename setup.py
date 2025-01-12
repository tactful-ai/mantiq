from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="mantiq",
    version="0.1.0",
    author="Tactful AI team",
    author_email="ahmed.osama@tactful.ai",
    description="A package for Cache-Augmented Generation (CAG).",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,  # Use requirements from the file
    include_package_data=True,
)
