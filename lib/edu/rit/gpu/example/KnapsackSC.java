//******************************************************************************
//
// File:    KnapsackSC.java
// Package: edu.rit.gpu.example
// Unit:    Class edu.rit.gpu.example.KnapsackSC
//
// This Java source file is copyright (C) 2016 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This Java source file is part of the Parallel Java 2 Library ("PJ2"). PJ2 is
// free software; you can redistribute it and/or modify it under the terms of
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

package edu.rit.gpu.example;

import edu.rit.util.Random;
import java.util.NoSuchElementException;

/**
 * Class KnapsackSC provides an object that defines a <I>strongly correlated</I>
 * knapsack problem. The definition consists of the knapsack capacity, the
 * number of items, and the weight and value of each item. Each item's weight is
 * chosen uniformly at random in the range 1 through <I>maxW</I> inclusive. Each
 * item's value is equal to its weight plus <I>deltaV</I>, where <I>deltaV</I>
 * may be positive, zero, or negative.
 *
 * @author  Alan Kaminsky
 * @version 15-Feb-2016
 */
public class KnapsackSC
	implements KnapsackProblem
	{

// Hidden data members.

	private long C;
	private int N;
	private long maxW;
	private long deltaV;
	private Random prng;
	private int count;

// Exported constructors.

	/**
	 * Construct a new strongly correlated knapsack problem.
	 *
	 * @param  C       Knapsack capacity &ge; 1.
	 * @param  N       Number of items &ge; 1.
	 * @param  maxW    Maximum item weight &ge; 1.
	 * @param  deltaV  Item value delta.
	 * @param  seed    Pseudorandom number generator seed.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if any argument is not in the stated
	 *     range.
	 */
	public KnapsackSC
		(long C,
		 int N,
		 long maxW,
		 long deltaV,
		 long seed)
		{
		if (C < 1)
			throw new IllegalArgumentException (String.format
				("KnapsackSC(): C = %d illegal", C));
		if (N < 1)
			throw new IllegalArgumentException (String.format
				("KnapsackSC(): N = %d illegal", N));
		if (maxW < 1)
			throw new IllegalArgumentException (String.format
				("KnapsackSC(): maxW = %d illegal", maxW));

		this.C = C;
		this.N = N;
		this.maxW = maxW;
		this.deltaV = deltaV;
		this.prng = new Random (seed);
		this.count = 0;
		}

// Exported operations.

	/**
	 * Get the knapsack capacity.
	 *
	 * @return  Capacity.
	 */
	public long capacity()
		{
		return C;
		}

	/**
	 * Get the number of items, <I>N.</I>
	 *
	 * @return  Number of items.
	 */
	public int itemCount()
		{
		return N;
		}

	/**
	 * Get the weight and value of the next item. This method must be called
	 * <I>N</I> times to get all the items. Each method call returns a new
	 * {@linkplain WV} object.
	 *
	 * @return  Weight/value.
	 *
	 * @exception  NoSuchElementException
	 *     (unchecked exception) Thrown if this method is called more than
	 *     <I>N</I> times.
	 */
	public WV next()
		{
		if (count >= N)
			throw new NoSuchElementException
				("KnapsackSC.next(): Called too many times");

		-- count;
		long W = (long)(maxW*prng.nextDouble() + 1);
		return new WV (W, W + deltaV);
		}

	}
