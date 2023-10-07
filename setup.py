from setuptools import find_packages,setup
from pathlib import Path
from typing import List

def get_requirements(path:Path)-> List[str]:
    """this function will return all requirements"""
    HYPHEN_E_DOT = "-e ."
    with open(path, "r") as file_obj:
        requirements = file_obj.readlines()
        requirements = [requirement.replace("\n","") for requirement in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements
        
    
setup(
    name="aiusingtensorflow",
    version="0.0.1",
    author="Debi Prasad Rath",
    author_email="debi.rath817@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)