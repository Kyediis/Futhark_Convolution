#!/usr/bin/env python

import image_conv
import numpy as np
from scipy import misc
import argparse

def main(infile, kernel, outfile, iterations):
    c = image_conv.image_conv()
    img = misc.imread(infile, mode='RGB')
    file1 = open(kernel,"r+")  
    krn = eval(file1.readline())
    array = np.asarray(krn, dtype=np.float32)
    conv = c.main(iterations, img, array)
    misc.imsave(outfile, conv.get().astype(np.uint8))

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
