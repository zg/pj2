//******************************************************************************
//
// File:    WalkSackGpu.cu
// Package: edu.rit.gpu.example
// Unit:    WalkSackGpu kernel function
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

#include "Prng.cu"
#include "WV.cu"

/**
 * Module WalkSackGpu provides GPU data types and operations for solving a
 * knapsack problem using a random walk.
 *
 * @author  Alan Kaminsky
 * @version 16-Mar-2016
 */

// Thread rank that found the best-of-best solution.
__device__ int bobRank;

// For shared memory reduction of thread ranks.
__shared__ int shrRank [1024];

// Swap elements a and b in array x.
__device__ void swap
	(int *x,
	 int a,
	 int b)
	{
	int t = x[a];
	x[a] = x[b];
	x[b] = t;
	}

/**
 * Kernel function to solve a knapsack problem.
 * <P>
 * Called with a one-dimensional grid of one-dimensional blocks, NB blocks, NT
 * threads per block, NG = NB*NT threads total.
 * <P>
 * Each thread performs an independent random walk through the space of subsets
 * of items and keeps track of the best solution found. After doing a certain
 * number of steps, the kernel outputs the best of the per-thread solutions.
 *
 * @param  C
 *     Knapsack capacity. (input)
 * @param  N
 *     Number of items. (input)
 * @param  item_wv
 *     Array of items' weights/values, indexed 0 through N-1. (input)
 * @param  curr_soln
 *     Array of the threads' current solutions, indexed 0 through NG-1. Each
 *     element in turn is an array of integers, indexed 0 through N. For thread
 *     i, curr_soln[i][0] is the number of items in the knapsack, M;
 *     curr_soln[i][1..M] are the indexes of the items in knapsack in no
 *     particular order; curr_soln[i][M+1..N] are the indexes of the items not
 *     in the knapsack in no particular order. (input/output)
 * @param  best_wv
 *     Array of total weights/values for the threads' best solutions, indexed 0
 *     through NG-1. (input/output)
 * @param  best_soln
 *     Array of the threads' best solutions, indexed 0 through NG-1. Each
 *     element in turn is an array of integers, indexed 0 through N, the same as
 *     curr_soln. (input/output)
 * @param  best_step
 *     Array of steps at which each thread achieved its best solution, indexed
 *     0 through NG-1. (output)
 * @param  seed
 *     Random seed.
 * @param  steps
 *     Number of steps. (input)
 */
extern "C" __global__ void walk
	(unsigned long long int C,
	 int N,
	 wv_t *item_wv,
	 int **curr_soln,
	 wv_t *best_wv,
	 int **best_soln,
	 unsigned long long int *best_step,
	 unsigned long long int seed,
	 unsigned long long int steps)
	{
	int r, i, j, M, oldRank, newRank;
	unsigned long long int s;
	wv_t curr_wv_r;
	wv_t best_wv_r;
	int *curr_soln_r;
	int *best_soln_r;
	prng_t prng;

	// Determine this thread's rank in the grid.
	r = blockIdx.x*blockDim.x + threadIdx.x;

	// Set up pointers to this thread's data.
	curr_soln_r = curr_soln[r]; // Items in current solution
	best_soln_r = best_soln[r]; // Items in best solution

	// Initialize this thread's data.
	wv_init (&curr_wv_r);       // Weight/value of current solution
	wv_init (&best_wv_r);       // Weight/value of best solution
	M = 0;                      // Number of items in current solution
	for (i = 1; i <= N; ++ i)   // Items in current solution
		curr_soln_r[i] = i - 1;
	prng_setSeed (&prng, seed + r);

	// Do <steps> random steps.
	for (s = 0; s < steps; ++ s)
		{
		// If current solution's weight < knapsack capacity, add a random item.
		if (curr_wv_r.weight < C)
			{
			i = M + 1;
			j = M + 1 + prng_nextInt (&prng, N - M);
			swap (curr_soln_r, i, j);
			wv_add (&curr_wv_r, &item_wv[curr_soln_r[i]]);
			++ M;
			}

		// If current solution's weight >= knapsack capacity, subtract a random
		// item.
		else
			{
			i = M;
			j = 1 + prng_nextInt (&prng, M);
			swap (curr_soln_r, i, j);
			wv_subtract (&curr_wv_r, &item_wv[curr_soln_r[i]]);
			-- M;
			}

		// If new current solution is better than best solution, record new best
		// solution.
		if (wv_isBetterThan (&curr_wv_r, &best_wv_r, C))
			{
			best_wv_r = curr_wv_r;
			best_soln_r[0] = M;
			for (i = 1; i <= M; ++ i)
				best_soln_r[i] = curr_soln_r[i];
			best_step[r] = s + 1;
			}
		}

	// Record weight/value of best solution.
	best_wv[r] = best_wv_r;
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
