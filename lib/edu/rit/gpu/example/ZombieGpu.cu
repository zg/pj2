//******************************************************************************
//
// File:    ZombieGpu.cu
// Package: edu.rit.gpu.example
// Unit:    ZombieGpu kernel function
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

// Variables in global memory.
__device__ double devDelta;

// Per-thread variables in shared memory.
__shared__ double shrVx [NT];
__shared__ double shrVy [NT];

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
 * @param  xpos   Array of zombies' current X coordinates.
 * @param  ypos   Array of zombies' current Y coordinates.
 * @param  xnext  Array of zombies' next X coordinates.
 * @param  ynext  Array of zombies' next Y coordinates.
 * @param  N      Number of zombies.
 * @param  G      Parameter in velocity formula.
 * @param  L      Parameter in velocity formula.
 * @param  dt     Time step.
 *
 * @author  Alan Kaminsky
 * @version 25-Mar-2014
 */
extern "C" __global__ void timeStep
	(double *xpos,
	 double *ypos,
	 double *xnext,
	 double *ynext,
	 int N,
	 double G,
	 double L,
	 double dt)
	{
	int i = blockIdx.x;      // Index of this block's zombie
	int j = threadIdx.x;     // Index of this thread within block
	double xpos_i = xpos[i]; // This zombie's current X position
	double ypos_i = ypos[i]; // This zombie's current X position
	double vx = 0.0;         // This zombie's X velocity
	double vy = 0.0;         // This zombie's Y velocity
	int k;
	double dx, dy, d, v;

	// Compute and accumulate velocity w.r.t. every other zombie.
	for (k = j; k < N; k += NT)
		{
		if (k == i) continue;
		dx = xpos[k] - xpos_i;
		dy = ypos[k] - ypos_i;
		d = sqrt(dx*dx + dy*dy);
		v = G*exp(-d/L) - exp(-d);
		vx += v*dx/d;
		vy += v*dy/d;
		}

	// Compute net velocity via shared memory parallel reduction.
	shrVx[j] = vx;
	shrVy[j] = vy;
	__syncthreads();
	for (k = NT/2; k > 0; k >>= 1)
		{
		if (j < k)
			{
			shrVx[j] += shrVx[j+k];
			shrVy[j] += shrVy[j+k];
			}
		__syncthreads();
		}

	// Single threaded section.
	if (j == 0)
		{
		// Get net velocity.
		vx = shrVx[0];
		vy = shrVy[0];

		// Move zombie in the direction of its velocity.
		dx = vx*dt;
		dy = vy*dt;
		xnext[i] = xpos_i + dx;
		ynext[i] = ypos_i + dy;

		// Accumulate position delta.
		atomicAdd (&devDelta, abs(dx) + abs(dy));
		}
	}
