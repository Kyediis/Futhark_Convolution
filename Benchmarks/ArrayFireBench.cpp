/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <stdio.h>
#include "pch.h"
#include <cstdlib>
#include <arrayfire.h>
using namespace af;

// image to convolve
static array img1, img2, img3;

// 5x5 derivative with separable kernels
static float h_dx[] = { 1.f / 12, -8.f / 12, 1.f / 12 }; // five point stencil
static float h_spread[] = { 1.f / 5, 1.f / 5, 1.f / 5 };
static array dx, spread, kernel; // device kernels

static array full_out, dsep_out, hsep_out; // save output for value checks
// wrapper functions for timeit() below
static void full1() { full_out = convolve2(img1, kernel); }
static void full2() { full_out = convolve2(img2, kernel); }
static void full3() { full_out = convolve2(img3, kernel); }

static bool fail(array &left, array &right)
{
	return (max<float>(abs(left - right)) > 1e-6);
}

int main(int argc, char **argv)
{
	int device = argc > 1 ? atoi(argv[1]) : 0;
	af::setDevice(device);
	af::info();
	// setup image and device copies of kernels
	img1 = randu(512, 512);
	img2 = randu(1024, 1024);
	img3 = randu(2048, 2048);
	dx = array(5, 1, h_dx); // 3x1 kernel
	spread = array(1, 5, h_spread); // 1x3 kernel
	kernel = matmul(dx, spread); // 3x3 kernel
	printf("512x512 convolution:         %.5f seconds\n", timeit(full1));
	printf("1024x1024 convolution:         %.5f seconds\n", timeit(full2));
	printf("2048x2048 convolution:         %.5f seconds\n", timeit(full3));
	
return 0;
}
