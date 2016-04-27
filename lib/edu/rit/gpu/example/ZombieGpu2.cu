//******************************************************************************
//
// File:    ZombieGpu2.cu
// Package: edu.rit.gpu.example
// Unit:    ZombieGpu2 kernel function
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

// Number of threads per block.
#define NT 256

// Structure for a 2-D vector.
typedef struct
	{
	double x;
	double y;
	}
	vector_t;

// Vector addition; a = b + c. Returns a.
__device__ vector_t *vectorAdd
	(vector_t *a,
	 vector_t *b,
	 vector_t *c)
	{
	a->x = b->x + c->x;
	a->y = b->y + c->y;
	return a;
	}

// Vector subtraction; a = b - c. Returns a.
__device__ vector_t *vectorSubtract
	(vector_t *a,
	 vector_t *b,
	 vector_t *c)
	{
	a->x = b->x - c->x;
	a->y = b->y - c->y;
	return a;
	}

// Scalar product; a = a*b. Returns a.
__device__ vector_t *scalarProduct
	(vector_t *a,
	 double b)
	{
	a->x *= b;
	a->y *= b;
	return a;
	}

// Returns the magnitude of a.
__device__ double vectorMagnitude
	(vector_t *a)
	{
	return sqrt (a->x*a->x + a->y*a->y);
	}

// Variables in global memory.
__device__ double devDelta;

// Per-thread variables in shared memory.
__shared__ vector_t shrVel [NT];

/**
 * Atomically set double variable v to the sum of itself and value.
 *
 * @param  v      Pointer to double variable.
 * @param  value  Value.
 */
__device__ void atomicAdd
	(double *v,
	 double value)
	{
	double oldval, newval;
	do
		{
		oldval = *v;
		newval = oldval + value;
		}
	while (atomicCAS
		((unsigned long long int *)v,
		 __double_as_longlong (oldval),
		 __double_as_longlong (newval))
			!= __double_as_longlong (oldval));
	}

/**
 * Device kernel to update zombie positions after one time step.
 * <P>
 * Called with a one-dimensional grid of one-dimensional blocks, N blocks, NT
 * threads per block. N = number of zombies. Each block updates one zombie. Each
 * thread within a block computes the velocity with respect to one other zombie.
 *
 * @param  pos   Array of zombies' current positions.
 * @param  next  Array of zombies' next positions.
 * @param  N     Number of zombies.
 * @param  G     Parameter in velocity formula.
 * @param  L     Parameter in velocity formula.
 * @param  dt    Time step.
 *
 * @author  Alan Kaminsky
 * @version 28-Oct-2014
 */
extern "C" __global__ void timeStep
	(vector_t *pos,
	 vector_t *next,
	 int N,
	 double G,
	 double L,
	 double dt)
	{
	int i = blockIdx.x;        // Index of this block's zombie
	int j = threadIdx.x;       // Index of this thread within block
	vector_t pos_i = pos[i];   // This zombie's current position
	vector_t vel = {0.0, 0.0}; // This zombie's velocity
	int k;
	vector_t posdiff;
	double d, v;

	// Compute and accumulate velocity w.r.t. every other zombie.
	for (k = j; k < N; k += NT)
		{
		if (k == i) continue;
		vectorSubtract (&posdiff, &pos[k], &pos_i);
		d = vectorMagnitude (&posdiff);
		v = G*exp(-d/L) - exp(-d);
		vectorAdd (&vel, &vel, scalarProduct (&posdiff, v/d));
		}

	// Compute net velocity via shared memory parallel reduction.
	shrVel[j] = vel;
	__syncthreads();
	for (k = NT/2; k > 0; k >>= 1)
		{
		if (j < k)
			vectorAdd (&shrVel[j], &shrVel[j], &shrVel[j+k]);
		__syncthreads();
		}

	// Single threaded section.
	if (j == 0)
		{
		// Get net velocity.
		vel = shrVel[0];

		// Move zombie in the direction of its velocity.
		vectorAdd (&next[i], &pos_i, scalarProduct (&vel, dt));

		// Accumulate position delta.
		atomicAdd (&devDelta, abs(vel.x) + abs(vel.y));
		}
	}
