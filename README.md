# tsv

[![Travis CI Status](https://travis-ci.org/chunglabmit/tsv.svg?branch=master)](https://travis-ci.org/chunglabmit/tsv)


Library to render terastitched volumes

This is a pure Python library that uses the XML output of Terastitcher
to load stitched volumes. Example usage:

```python

import tsv

volume = tsv.volume.TSVVolume.load(xml_path)
#
# Read from x=0, y=0, z=0 to x=1024, y=1024, z=100
#
img = volume.imread(tsv.volume.VExtent(0, 1024, 0, 1024, 0, 100))

```
