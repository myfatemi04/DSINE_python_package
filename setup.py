from setuptools import setup, find_packages

setup(
    name="dsine",
    version="0.1.0",
    description="A Python package for the DSINE paper.",
    author="Gwangae Bin",
    maintainer="Michael Yoo Fatemi",
    maintainer_email="myfatemi04@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "geffnet",
        "matplotlib",
        "tensorboard",
        "opencv-python",
        # Add required dependencies here, e.g.,
        # 'numpy>=1.18.5',
        # 'pandas>=1.0.5',
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="data science, machine learning, utilities",
    url="https://github.com/myfatemi04/DSINE_python_package",
    project_urls={
        "Bug Tracker": "https://github.com/myfatemi04/DSINE_python_package/issues",
        "Source Code": "https://github.com/myfatemi04/DSINE_python_package",
    },
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
