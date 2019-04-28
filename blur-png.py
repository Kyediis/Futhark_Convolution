#!/usr/bin/env python

import image_conv
import numpy as np
from scipy import misc
import argparse

def main(infile, kernel, outfile, iterations):
    b = image_conv.image_conv()
    img = misc.imread(infile, mode='RGB')
    file1 = open(kernel,"r+")  
    krn = np.asarray(file1.readline())
    #(height, width, channels) = img.shape
    blurred = b.main(iterations, img, krn)
    # The .get() is to retrieve a Numpy array from the PyOpenCL array
    # being returned.
    misc.imsave(outfile, blurred.get().astype(np.uint8))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple Gaussian blur of a PNG file.')
    parser.add_argument('--output-file', metavar='OUTFILE', type=str, required=True,
                        help='Where the result is written')
    parser.add_argument('--iterations', metavar='INT', type=int, default=1,
                        help='Number of iterations to run')
    parser.add_argument('--kernel', metavar='INFILE', type=str,
                        help='Kernel to convolve image with')
    parser.add_argument('filename', metavar='INFILE', type=str,
                        help='The PNG file to blur')
    args = parser.parse_args()
    main(args.filename, args.kernel, args.output_file, args.iterations)
