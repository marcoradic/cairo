# Cairo - Context Aware Image Resizing :camera:

Python implementation of Context Aware Image Resizing (or Seam Carving) as described in a paper by [Avidan et al](http://graphics.cs.cmu.edu/courses/15-463/2007_fall/hw/proj2/imret.pdf)..

Instead of naive linear scaling, an energy map of the image is generated to determine the least informative pixel information that is then iteratively removed using dynamic programming until the desired dimensions are reached.
Currently, only shrinking of images is implemented (Scaling <= 1.0).
Efficient operations are achieved using numpy arrays along with numba JIT optimizations.

As an example, consider the following image, taken by Pietro De Grandi of the Pragser Wildsee in Italy:

![Pragser Wildsee](resources/pietro.JPEG)

After taking away 400 pixels of width using Cairo, the image looks as follows:

![Retargeted Wildsee](resources/out.JPEG)

Import details are preserved, because these regions have high energy/gradient values.

## Installation

Using Python 3

```
pip install -r requirements.txt
```

## Usage

```
python cairo.py [-v] [scaling factor/ desired dimenions] [input file] [output file]
```

Example:

```
python cairo.py -v 600 700 input.jpg output.jpg
```
