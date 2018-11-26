from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="obj_track",
    version="0.0.1",
    author="Paola Soto",
    author_email="paola.soto-arenas@uantwerpen.be",
    description="A demo on object tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/paolasoto88/obj_track.git", #complete after done
    packages=find_packages(),
    install_requires=["numpy<=1.14.5,>=1.13.3",
                      "scipy", "imutils", "opencv-python", "Cython",
                      "contextlib2", "pillow", "lxml", "jupyter", "matplotlib",
                      "keras", "h5py", "wget"],
    extras_require={
        "cpu": ["tensorflow==1.10.0"],
        "gpu": ["tensorflow-gpu==1.10.0"]
    },
    scripts=["bin/obj_track.py",
             "bin/convert_yad2k.py"],
    classifiers=[
        "Programming Language :: Python :: 3.6.5",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning",
    ],
)