//******************************************************************************
//
// File:    KnapsackExhGpu.cu
// Package: edu.rit.gpu.example
// Unit:    KnapsackExhGpu kernel function
//
// This C/CUDA source file is copyright (C) 2016 by Alan Kaminsky. All rights
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

#include "WV.cu"

/**
 * Module KnapsackExhGpu provides GPU data types and operations for solving a
 * knapsack problem using an exhaustive search. The kernel can solve knapsack
 * problems with up to 63 items.
 *
 * @author  Alan Kaminsky
 * @version 16-Mar-2016
 */

// Thread rank that found the best-of-best solution.
__device__ int bobRank;

// For shared memory reduction of thread ranks.
__shared__ int shrRank [1024];

/**
 * Kernel function to solve a knapsack problem using an exhaustive search.
 * <P>
 * Called with a one-dimensional grid of one-dimensional blocks, NB blocks, NT
 * threads per block, NG = NB*NT threads total.
 * <P>
 * Each thread evaluates certain subsets of items and keeps track of the best
 * solution found. The kernel outputs the best of the per-thread solutions.
 *
 * @param  C
 *     Knapsack capacity. (input)
 * @param  N
 *     Number of items. (input)
 * @param  item_wv
 *     Array of items' weights/values, indexed 0 through N-1. (input)
 * @param  soln_lb
 *     Inclusive lower bound of solutions to search.
 * @param  soln_ub
 *     Inclusive upper bound of solutions to search.
 * @param  best_wv
 *     Array of total weights/values for the threads' best solutions, indexed 0
 *     through NG-1. (input/output)
 * @param  best_soln
 *     Array of the threads' best solutions, indexed 0 through NG-1. Each
 *     element is a long integer bitmap of the items in the solution.
 *     (input/output)
 */
extern "C" __global__ void search
	(unsigned long long int C,
	 int N,
	 wv_t *item_wv,
	 unsigned long long int soln_lb,
	 unsigned long long int soln_ub,
	 wv_t *best_wv,
	 unsigned long long int *best_soln)
	{
	int NG, r, i, j, oldRank, newRank;
	unsigned long long int best_soln_r, curr_soln_r;
	wv_t best_wv_r, curr_wv_r;

	// Determine total number of threads in the grid.
	NG = gridDim.x*blockDim.x;

	// Determine this thread's rank in the grid.
	r = blockIdx.x*blockDim.x + threadIdx.x;

	// Initialize this thread's data.
	wv_init (&best_wv_r);
	best_soln_r = 0;

	// Search all possible solutions in parallel in a leapfrog fashion.
	for (curr_soln_r = soln_lb + r; curr_soln_r <= soln_ub; curr_soln_r += NG)
		{
		// Compute total weight and value of current solution.
		wv_init (&curr_wv_r);
		for (i = 0; i < N && curr_wv_r.weight < C; ++ i)
			if (curr_soln_r & (1ULL << i))
				wv_add (&curr_wv_r, &item_wv[i]);

		// Keep previous best solution or current solution, whichever is better.
		if (i == N && wv_isBetterThan (&curr_wv_r, &best_wv_r, C))
			{
			best_wv_r = curr_wv_r;
			best_soln_r = curr_soln_r;
			}
		}

	// Record weight/value of best solution.
	best_wv[r] = best_wv_r;
	best_soln[r] = best_soln_r;
	__threadfence();

	// Shared memory reduction to determine thread rank with best-of-best
	// solution in this block.
	i = threadIdx.x;
	shrRank[i] = r;
	__syncthreads();
	j = 1;
	while (j < blockDim.x) j <<= 1;
	j >>= 1;
	while (j != 0)
		{
		if (i < j && i + j < blockDim.x &&
			wv_isBetterThan (&best_wv[shrRank[i+j]], &best_wv[shrRank[i]], C))
				shrRank[i] = shrRank[i+j];
		__syncthreads();
		j >>= 1;
		}

	// Global memory reduction to determine thread rank with best-of-best
	// solution across all blocks.
	if (i == 0)
		do
			{
			oldRank = bobRank;
			newRank =
				oldRank == -1 ||
				wv_isBetterThan (&best_wv[shrRank[0]], &best_wv[oldRank], C) ?
					shrRank[0] : oldRank;
			}
		while (atomicCAS (&bobRank, oldRank, newRank) != oldRank);
	}
