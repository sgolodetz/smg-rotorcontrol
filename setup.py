from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="smg-rotorcontrol",
    version="0.0.1",
    author="Stuart Golodetz",
    author_email="stuart.golodetz@cs.ox.ac.uk",
    description="Drone flight controllers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sgolodetz/smg-rotorcontrol",
    packages=find_packages(include=["smg.rotorcontrol"]),
    include_package_data=True,
    install_requires=[
        "smg-joysticks",
        "smg-navigation",
        "smg-rotory"
    ],
    extras_require={
        "all": [
            "amazon-transcribe",
            "sounddevice"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
