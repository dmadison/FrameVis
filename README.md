![FrameVis Banner](images/FrameVis_Banner.jpg)

FrameVis is a Python script for generating video frame visualizations, also known as "movie barcodes". These visualizations are composed of frames taken from a video file at a regular interval, resized, and then stacked together to show the compressed color palette of the video and how it changes over time.

For more information, see [the blog post on PartsNotIncluded.com](http://www.partsnotincluded.com/programming/framevis/).

## Basic Usage

```bash
python framevis.py source_video.mkv result.png -n 1600
```

To use the script, invoke it from the command line and pass positional arguments for the source and destination file paths (respectively). You will also need to provide either the number of frames to use (`-n`), or a capture interval in seconds (`-i`). The script will then process the video and save the result to the file specified.

## Installation

To use FrameVis, you will need a copy of [Python 3](https://www.python.org/downloads/) installed and added to your computer's path. You will also need a copy of the OpenCV library for Python 3, which can either be built from source ([Windows](https://docs.opencv.org/master/d5/de5/tutorial_py_setup_in_windows.html), [Ubuntu](https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html), [Fedora](https://docs.opencv.org/master/dd/dd5/tutorial_py_setup_in_fedora.html)) or installed using [the **unofficial** binaries](https://pypi.org/project/opencv-python/) available via pip:

```python
pip install opencv-python
```

Test that both OpenCV (`cv2`) and NumPy (`numpy`) successfully import into Python before trying the script for the first time. Note that this script was developed using Python 3.6.4 and OpenCV version 3.4.1. More recent versions may not work properly.

## Command Line Arguments

### source (positional)

The first positional argument is the file path for the video file to be visualized. Works with all OpenCV compatible video codecs and wrappers.

### destination (positional)

The second positional argument is the file path to save the final, visualized image. [Compatible file formats](https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#imread) include jpeg, png, bmp, and tiff. A proper file extension *must* be included in the path or saving will fail.

### (n)frames and (i)nterval

One of these two arguments is required to set the number of frames to use in the visualization. You can either set the number of frames directly with `--(n)frames`, or indirectly by setting a capture `--(i)nterval` in seconds. Captured frames with either method are spaced throughout the entire video.

## Optional Arguments

### (h)eight and (w)idth

The number of pixels to use for height and width *per frame*. If unset, these default to 1 px in the concatenated direction and the full size of the video in the other.

### (d)irection

The direction in which to concatenate the video frames, either "horizontal" or "vertical". Defaults to "horizontal".

### (t)rim

Setting this flag attempts to automatically remove hard matting present in the video file ([letterboxing / pillarboxing](https://en.wikipedia.org/wiki/Letterboxing_(filming))) before resizing. Off by default.

### (a)verage

Postprocess effect that averages all of the colors in each frame. Off by default, mutually exclusive with [(b)lur](#blur). It's recommended to enable trimming when using this option, otherwise colors will be excessively darkened.

### (b)lur

Postprocess effect that blurs each frame to smooth the final image. The value is the kernel size used for [convolution](https://en.wikipedia.org/wiki/Kernel_(image_processing)). If the flag is set and no value is passed, defaults to 100. Off by default, mutually exclusive with [(a)verage](#average). It's recommended to enable trimming when using this option, otherwise frame colors will bleed into any present matting.

## License

This script is licensed under the terms of the [MIT license](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for more information.