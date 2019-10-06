#
#  Project     FrameVis - Video Frame Visualizer Script
#  @author     David Madison
#  @link       github.com/dmadison/FrameVis
#  @license    MIT - Copyright (c) 2019 David Madison
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

import cv2
import numpy as np
import argparse
from enum import Enum, auto


class FrameVis:
	"""
	Reads a video file and outputs an image comprised of n resized frames, spread evenly throughout the file.
	"""

	class Direction(Enum):
		HORIZONTAL = auto()  # left to right
		VERTICAL = auto()  # top to bottom

	default_frame_height = None  # auto, or in pixels
	default_frame_width = None  # auto, or in pixels
	default_concat_size = 1  # size of concatenated frame if automatically calculated, in pixels
	default_direction = Direction.HORIZONTAL

	def visualize(self, source, destination, nframes, height=default_frame_height, width=default_frame_width, direction=default_direction, quiet=True):
		"""
		Reads a video file and outputs an image comprised of n resized frames, spread evenly throughout the file.

		Parameters:
			source (str): filepath to source video file
			destination (str): filepath for output image
			nframes (int): number of frames to process from the video
			height (int): height of each frame, in pixels
			width (int): width of each frame, in pixels
			direction (enum): direction to concatenate frames (horizontal or vertical)
			quiet (bool): suppress console messages
		"""

		video = cv2.VideoCapture(source)  # open video file
		if not video.isOpened():
			raise FileNotFoundError("Source Video Not Found")
		
		# calculate keyframe interval
		video_total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)  # retrieve total frame count from metadata
		if not isinstance(nframes, int) or nframes < 1:
			raise ValueError("Number of frames must be a positive integer")
		elif nframes > video_total_frames:
			raise ValueError("Requested frame count larger than total available ({})".format(video_total_frames))
		keyframe_interval = video_total_frames / nframes  # calculate number of frames between captures

		# grab frame for dimension calculations
		success,image = video.read()  # get first frame
		if not success:
			raise IOError("Cannot read from video file")

		# calculate height
		if height is None:  # auto-calculate
			if direction is FrameVis.Direction.HORIZONTAL:  # non-concat, use video size
				height = image.shape[0]  # save frame height
			else:  # concat, use default value
				height = FrameVis.default_concat_size
		elif not isinstance(height, int) or height < 1:
			raise ValueError("Frame height must be a positive integer")
		
		# calculate width
		if width is None:  # auto-calculate
			if direction is FrameVis.Direction.VERTICAL:  # non-concat, use video size
				width = image.shape[1]  # save frame width
			else:  # concat, use default value
				width = FrameVis.default_concat_size
		elif not isinstance(width, int) or width < 1:
			raise ValueError("Frame width must be a positive integer")

		# assign direction function and calculate output size
		if direction is FrameVis.Direction.HORIZONTAL:
			concatenate = cv2.hconcat
			output_width = width * nframes
			output_height = height
		elif direction is FrameVis.Direction.VERTICAL:
			concatenate = cv2.vconcat
			output_width = width
			output_height = height * nframes
		else:
			raise ValueError("Invalid direction specified")

		if not quiet:
			print("\nVisualizing \"{}\" - {} by {}, from {} frames".format(source, output_width, output_height, nframes))

		# set up for the frame processing loop
		next_keyframe = keyframe_interval / 2  # frame number for the next frame grab, starting evenly offset from start/end
		finished_frames = 0  # counter for number of processed frames
		output_image = None

		while True:
			if finished_frames == nframes:
				break  # done!

			video.set(cv2.CAP_PROP_POS_FRAMES, int(next_keyframe))  # move cursor to next sampled frame
			success,image = video.read()  # read the next frame

			if not success:
				raise IOError("Cannot read from video file (frame {} out of {})".format(int(next_keyframe), video_total_frames))

			if output_image is None:
				output_image = cv2.resize(image, (width, height))
			else:
				output_image = concatenate([output_image, cv2.resize(image, (width, height))])  # concatenate horizontally from left -> right

			finished_frames += 1
			next_keyframe += keyframe_interval  # set next frame capture time, maintaining floats

			if not quiet:
				FrameVis.progress_bar(finished_frames / nframes)  # print progress bar to the console

		cv2.imwrite(destination, output_image)  # save visualization to file
		if not quiet:
			print("Visualization saved to {}".format(destination))

	@staticmethod
	def progress_bar(percent):
		"""Prints a progress bar to the console based on the input percentage (float)."""
		term_char = '\r' if percent < 1.0 else '\n'  # rewrite the line unless finished
		bar_length = 25  # size of the progress bar, in characters
		filled_size = int(round(bar_length * percent))  # number of 'filled' characters in the bar
		progress_string = "#" * filled_size + " " * (bar_length - filled_size)  # assembled progress bar, as a string
		print("Processing:\t[{0}]\t{1:.2%}".format(progress_string, percent), end=term_char, flush=True)


def main():
	parser = argparse.ArgumentParser(add_help=False)  # removing help so I can use '-h' for height

	parser.add_argument("source", help="file path for the video file to be visualized", type=str)
	parser.add_argument("destination", help="file path output for the final image", type=str)
	parser.add_argument("-n", "--nframes", help="the number of frames in the visualization", type=int, required=True)
	parser.add_argument("-h", "--height", help="the height of each frame, in pixels", type=int, default=FrameVis.default_frame_height)
	parser.add_argument("-w", "--width", help="the output width of each frame, in pixels", type=int, default=FrameVis.default_frame_width)
	parser.add_argument("-d", "--direction", help="direction to concatenate frames, horizontal or vertical", type=str, choices=["horizontal", "vertical"])
	parser.add_argument("-q", "--quiet", help="mute console outputs", action='store_true', default=False)
	parser.add_argument("--help", action="help", help="show this help message and exit")

	args = parser.parse_args()

	direction = FrameVis.default_direction
	if args.direction is not None:
		if args.direction == "horizontal":
			direction = FrameVis.Direction.HORIZONTAL
		elif args.direction == "vertical":
			direction = FrameVis.Direction.VERTICAL

	fv = FrameVis()
	fv.visualize(args.source, args.destination, args.nframes, height=args.height, width=args.width, direction=direction, quiet=args.quiet)


if __name__ == "__main__":
	main()
