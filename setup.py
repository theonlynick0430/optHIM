from setuptools import setup, find_packages

setup(
    name="optim",
    packages=[
        package for package in find_packages() if package.startswith("optim")
    ],
)
