//******************************************************************************
//
// File:    WV.cu
// Package: edu.rit.gpu.example
// Unit:    Knapsack problem weight/value functions
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

#ifndef __WV_CU_INCLUDED__
#define __WV_CU_INCLUDED__

/**
 * This file contains CUDA functions for a weight/value structure for a knapsack
 * problem. This file is intended to be #included into a program source file.
 *
 * @author  Alan Kaminsky
 * @version 16-Mar-2016
 */

// Weight and value of one item. Also used for total weight and total value of
// the knapsack.
typedef struct
	{
	unsigned long long int weight;
	unsigned long long int value;
	}
	wv_t;

// Initialize weight/value a to 0/0.
__device__ void wv_init
	(wv_t *a)
	{
	a->weight = 0;
	a->value = 0;
	}

// Add weight/value b to weight/value a.
__device__ void wv_add
	(wv_t *a,
	 wv_t *b)
	{
	a->weight += b->weight;
	a->value += b->value;
	}

// Subtract weight/value b from weight/value a.
__device__ void wv_subtract
	(wv_t *a,
	 wv_t *b)
	{
	a->weight -= b->weight;
	a->value -= b->value;
	}

// Determine which weight/value is better, a or b, with respect to capacity C.
// Returns 1 if a is better, 0 if b is better.
__device__ int wv_isBetterThan
	(wv_t *a,
	 wv_t *b,
	 unsigned long long int C)
	{
	if (a->weight > C && b->weight > C)
		return a->weight < b->weight;
	else if (a->weight > C)
		return 0;
	else if (b->weight > C)
		return 1;
	else
		return a->value > b->value;
	}

#endif
