//******************************************************************************
//
// File:    TotientSmp.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.TotientSmp
//
// This Java source file is copyright (C) 2013 by Alan Kaminsky. All rights
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

package edu.rit.pj2.example;

import edu.rit.pj2.LongLoop;
import edu.rit.pj2.Task;
import edu.rit.pj2.vbl.LongVbl;
import edu.rit.util.LongList;

/**
 * Class TotientSmp is an SMP parallel program that computes the Euler totient
 * of a number. &Phi;(<I>n</I>) is the number of numbers in the range 1 through
 * <I>n</I>&minus;1 that are relatively prime to <I>n</I>.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.example.TotientSmp <I>n</I></TT>
 *
 * @author  Alan Kaminsky
 * @version 28-Dec-2013
 */
public class TotientSmp
	extends Task
	{

// Program variables.

	long n;
	LongVbl phi;
	LongList nFactors = new LongList();

// Main program.

	/**
	 * Main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Validate command line arguments.
		if (args.length != 1) usage();
		n = Long.parseLong (args[0]);

		// Compute totient.
		phi = new LongVbl.Sum (0);
		factorize (n, nFactors);
		parallelFor (2, n - 1) .exec (new LongLoop()
			{
			LongList iFactors;
			LongVbl thrPhi;
			public void start()
				{
				iFactors = new LongList();
				thrPhi = threadLocal (phi);
				}
			public void run (long i)
				{
				if (relativelyPrime (factorize (i, iFactors), nFactors))
					++ thrPhi.item;
				}
			});

		// Print totient.
		System.out.printf ("%d%n", phi.item + 1);
		}

// Hidden operations.

	/**
	 * Store a list of the prime factors of <I>x</I> in ascending order in the
	 * given list.
	 *
	 * @param  x     Number to factorize.
	 * @param  list  List of prime factors (output).
	 *
	 * @param  The given list is returned.
	 */
	private static LongList factorize
		(long x,
		 LongList list)
		{
		list.clear();
		long p = 2;
		long psqr = p*p;
		while (psqr <= x)
			{
			if (x % p == 0)
				{
				list.addLast (p);
				x /= p;
				}
			else
				{
				p = p == 2 ? 3 : p + 2;
				psqr = p*p;
				}
			}
		if (x != 1)
			list.addLast (x);
		return list;
		}

	/**
	 * Determine whether two numbers are relatively prime, given their lists of
	 * factors.
	 *
	 * @param  xFactors  List of factors of first number.
	 * @param  yFactors  List of factors of second number.
	 *
	 * @return  True if the numbers are relatively prime (have no common
	 *          factors), false otherwise.
	 */
	private static boolean relativelyPrime
		(LongList xFactors,
		 LongList yFactors)
		{
		int xSize = xFactors.size();
		int ySize = yFactors.size();
		int ix = 0;
		int iy = 0;
		long x, y;
		while (ix < xSize && iy < ySize)
			{
			x = xFactors.get (ix);
			y = yFactors.get (iy);
			if (x == y) return false;
			else if (x < y) ++ ix;
			else ++ iy;
			}
		return true;
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.TotientSmp <n>");
		throw new IllegalArgumentException();
		}

	}
