//******************************************************************************
//
// File:    Test01.cu
// Package: edu.rit.gpu.test
// Unit:    Test01 kernel function
//
// This C/CUDA source file is copyright (C) 2014 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This C/CUDA source file is part of the Parallel Java 2 Library ("PJ2"). PJ2
// is free software; you can redistribute it and/or modify it under the terms of
// the GNU General Public License as published by the Free Software Foundation;
// either version 3 of the License, or (at your option) any later version.
//
// PJ2 is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// A copy of the GNU General Public License is provided in the file gpl.txt. You
// may also obtain a copy of the GNU General Public License on the World Wide
// Web at http://www.gnu.org/licenses/gpl.html.
//
//******************************************************************************

/**
 * Compute the sum of two integer vectors.
 * <P>
 * Called with a 1-D grid of 1-D blocks. Each thread computes one element of the
 * output vector.
 *
 * @param  a    First input vector.
 * @param  b    Second input vector.
 * @param  c    Output vector.
 * @param  len  Vector length.
 *
 * @author  Alan Kaminsky
 * @version 20-Feb-2014
 */
extern "C" __global__ void addIntVectors
	(int *a,
	 int *b,
	 int *c,
	 int len)
	{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < len)
		c[i] = a[i] + b[i];
	}
