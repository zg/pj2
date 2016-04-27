//******************************************************************************
//
// File:    KnapsackBnbGpu.cu
// Package: edu.rit.gpu.example
// Unit:    KnapsackBnbGpu kernel function
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
 * Module KnapsackBnbGpu provides GPU data types and operations for solving a
 * knapsack problem using a branch-and-bound search. The kernel can solve
 * knapsack problems with up to 63 items.
 * <P>
 * The search tree is partitioned into chunks. Each chunk consists of a
 * particular subset of the first <I>level</I> items, where <I>level</I> is a
 * kernel function argument, along with all possible subsets of the remaining
 * items. These chunks are partitioned among <I>NA</I> GPU accelerators in a
 * leapfrog fashion, where <I>NA</I> is the number of GPUs participating in the
 * search, also a kernel function argument. Within each GPU, the chunks are
 * partitioned among the kernel threads in a dynamic fashion.
 *
 * @author  Alan Kaminsky
 * @version 21-Mar-2016
 */

// For dynamic scheduling of search tree chunks.
__device__ unsigned long long int chunk_lb;

// Get the next available search tree chunk.
__device__ unsigned long long int nextChunk
	(int NA)
	{
	unsigned long long int oldChunk, newChunk;
	do
		{
		oldChunk = chunk_lb;
		newChunk = oldChunk + NA;
		}
	while (atomicCAS (&chunk_lb, oldChunk, newChunk) != oldChunk);
	return oldChunk;
	}

// Thread rank that found the best-of-best solution.
__device__ int bobRank;

// For shared memory reduction of thread ranks.
__shared__ int shrRank [1024];

// Clear the given bit in the given value.
__device__ unsigned long long int clearBit
	(unsigned long long int value,
	 int bit)
	{
	return value & ~(1ULL << bit);
	}

// Set the given bit in the given value.
__device__ unsigned long long int setBit
	(unsigned long long int value,
	 int bit)
	{
	return value | (1ULL << bit);
	}

// Test the given bit in the given value.
__device__ int testBit
	(unsigned long long int value,
	 int bit)
	{
	return (int)(value >> bit) & 1;
	}

/**
 * Kernel function to solve a knapsack problem using a branch-and-bound search.
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
 * @param  level
 *     Search tree level for chunking. (input)
 * @param  NA
 *     Number of GPU accelerators doing the search. (input)
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
	 int level,
	 int NA,
	 wv_t *best_wv,
	 unsigned long long int *best_soln)
	{
	int r, i, j, oldRank, newRank;
	unsigned long long int chunk_ub, best_soln_r, curr_soln_r, state_r;
	wv_t best_wv_r, curr_wv_r;

	// Determine this thread's rank in the grid.
	r = blockIdx.x*blockDim.x + threadIdx.x;

	// Initialize this thread's best solution.
	wv_init (&best_wv_r);
	best_soln_r = 0;

	// Compute chunks in parallel in a dynamic fashion.
	chunk_ub = (1ULL << level) - 1ULL;
	curr_soln_r = nextChunk (NA);
	while (curr_soln_r <= chunk_ub)
		{
		// Compute total weight and value of the first <level> items.
		wv_init (&curr_wv_r);
		for (i = 0; i < level && curr_wv_r.weight <= C; ++ i)
			if (testBit (curr_soln_r, i))
				wv_add (&curr_wv_r, &item_wv[i]);

		// If capacity not exceeded, do depth first traversal of search tree.
		if (curr_wv_r.weight <= C)
			{
			state_r = 0;
			while (i >= level)
				{
				// Base case: Update best solution; backtrack.
				if (i == N)
					{
					if (curr_wv_r.value > best_wv_r.value)
						{
						best_wv_r = curr_wv_r;
						best_soln_r = curr_soln_r;
						}
					-- i;
					}

				// Recursive cases: Item i not yet in knapsack.
				else if (! testBit (curr_soln_r, i))
					{
					// Test omitting item i from knapsack.
					if (! testBit (state_r, i))
						{
						state_r = setBit (state_r, i);
						++ i;
						}

					// Backtracking from test omitting item i from knapsack.
					// Test adding item i to knapsack, if capacity would not be
					// exceeded.
					else if (curr_wv_r.weight + item_wv[i].weight <= C)
						{
						curr_soln_r = setBit (curr_soln_r, i);
						wv_add (&curr_wv_r, &item_wv[i]);
						++ i;
						}

					// Prune search and backtrack, if capacity would be
					// exceeded.
					else
						{
						state_r = clearBit (state_r, i);
						-- i;
						}
					}

				// Recursive case: Backtracking from test adding item i to
				// knapsack.
				else
					{
					state_r = clearBit (state_r, i);
					curr_soln_r = clearBit (curr_soln_r, i);
					wv_subtract (&curr_wv_r, &item_wv[i]);
					-- i;
					}
				}
			}

		curr_soln_r = nextChunk (NA);
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
			best_wv[shrRank[i+j]].value > best_wv[shrRank[i]].value)
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
				best_wv[shrRank[0]].value > best_wv[oldRank].value ?
					shrRank[0] : oldRank;
			}
		while (atomicCAS (&bobRank, oldRank, newRank) != oldRank);
	}
