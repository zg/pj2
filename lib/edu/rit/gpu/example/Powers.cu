//******************************************************************************
//
// File:    Powers.cu
// Package: edu.rit.gpu.example
// Unit:    Powers kernel function
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

// Struct containing results of one calculation.
typedef struct
	{
	int x;
	double sqrt_x;
	double curt_x;
	double x_sq;
	double x_cu;
	}
	powers_t;

/**
 * Device kernel to compute powers of x.
 * <P>
 * Called with a one-dimensional grid of one-dimensional blocks, NB blocks, NT
 * threads per block.
 *
 * @param  powers  Pointer to array of powers structs.
 * @param  N       Array length.
 *
 * @author  Alan Kaminsky
 * @version 03-Jun-2014
 */
extern "C" __global__ void computePowers
	(powers_t *powers,
	 int N)
	{
	int thr, size, rank, i;
	double log_x;

	// Determine number of threads and this thread's rank.
	thr = threadIdx.x;
	size = gridDim.x*blockDim.x;
	rank = blockIdx.x*blockDim.x + thr;

	// Compute powers.
	for (i = rank; i < N; i += size)
		{
		log_x = log((double)(powers[i].x));
		powers[i].sqrt_x = exp(log_x/2);
		powers[i].curt_x = exp(log_x/3);
		powers[i].x_sq = exp(log_x*2);
		powers[i].x_cu = exp(log_x*3);
		}
	}
