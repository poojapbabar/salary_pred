from setuptools import find_packages,setup 
from typing import List


HYPHEN_E_DOT = "-e ."


def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements



setup(
    name = "PIPE_LINE_PROJECT",
    version="0.0.1",
    author = "pooja babar",
    description="This is a end to end ml pipeline project ",
    author_email= "babarpooja2002@gmail.com",
    install_requires = get_requirements("requirements.txt"),
    packages=find_packages()
)