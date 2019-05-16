#!/usr/bin/env python

import magnitude
import numpy as np
from scipy import misc
import argparse

def main(infile1, infile2, outfile):
    m = magnitude.magnitude()
    img1 = misc.imread(infile1, mode='RGB')
    img2 = misc.imread(infile2, mode='RGB')
    mag = m.main(img1, img2)
    misc.imsave(outfile, mag.get().astype(np.uint8))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find magnitude of two pngs')
    parser.add_argument('--output-file', metavar='OUTFILE', type=str, required=True,
                        help='Where the result is written')
    parser.add_argument('filename1', metavar='INFILE', type=str,
                        help='First png')
    parser.add_argument('filename2', metavar='INFILE', type=str,
                        help='Second png')
    args = parser.parse_args()
    main(args.filename1, args.filename1, args.output_file)