from setuptools import setup
import sys
version = "0.1.0"

with open("./README.md") as fd:
    long_description = fd.read()

install_requires = [
        "numpy",
        "h5py",
        "tifffile",
        "tqdm"
    ]
#
# Backport support for enum.Flags
#
if sys.version < "3.6":
    install_requires.insert(0, "aenum")

setup(
    name="tsv",
    version=version,
    description=
    "TeraStitcher Volume library",
    long_description=long_description,
    install_requires=install_requires,
    author="Kwanghun Chung Lab",
    packages=["tsv"],
    entry_points={ 'console_scripts': [
        "tsv-convert-2D-tif=tsv.convert:main",
        "tsv-simple-convert-2d-tif=tsv.simple:main",
        "tsv-diag-stack=tsv.convert:diag",
        "tsv-renumber=tsv.renumber:main"
    ]},
    url="https://github.com/chunglabmit/tsv",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Programming Language :: Python :: 3.5',
    ]
)