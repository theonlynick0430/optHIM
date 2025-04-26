from setuptools import setup, find_packages

setup(
    name="optHIM",
    packages=[
        package for package in find_packages() if package.startswith("optHIM")
    ],
)
