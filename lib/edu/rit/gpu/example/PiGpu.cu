//******************************************************************************
//
// File:    PiGpu.cu
// Package: edu.rit.gpu.example
// Unit:    PiGpu kernel function
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

#include "Prng.cu"

// Number of threads per block.
#define NT 1024

// Overall counter variable in global memory.
__device__ unsigned long long int devCount;

// Per-thread counter variables in shared memory.
__shared__ unsigned long long int shrCount [NT];

/**
 * Device kernel to compute random points.
 * <P>
 * Called with a one-dimensional grid of one-dimensional blocks, NB blocks, NT
 * threads per block. NT must be a power of 2.
 *
 * @param  seed  Pseudorandom number generator seed.
 * @param  N     Number of points.
 *
 * @author  Alan Kaminsky
 * @version 04-Jun-2014
 */
extern "C" __global__ void computeRandomPoints
	(unsigned long long int seed,
	 unsigned long long int N)
	{
	int thr, size, rank;
	unsigned long long int count;
	prng_t prng;

	// Determine number of threads and this thread's rank.
	thr = threadIdx.x;
	size = gridDim.x*NT;
	rank = blockIdx.x*NT + thr;

	// Initialize per-thread prng and count.
	prng_setSeed (&prng, seed + rank);
	count = 0;

	// Compute random points.
	for (unsigned long long int i = rank; i < N; i += size)
		{
		double x = prng_nextDouble (&prng);
		double y = prng_nextDouble (&prng);
		if (x*x + y*y <= 1.0) ++ count;
		}

	// Shared memory parallel reduction within thread block.
	shrCount[thr] = count;
	__syncthreads();
	for (int i = NT/2; i > 0; i >>= 1)
		{
		if (thr < i)
			shrCount[thr] += shrCount[thr+i];
		__syncthreads();
		}

	// Atomic reduction into overall counter.
	if (thr == 0)
		atomicAdd (&devCount, shrCount[0]);
	}
