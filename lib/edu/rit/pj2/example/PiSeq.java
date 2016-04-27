//******************************************************************************
//
// File:    PiSeq.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.PiSeq
//
// This Java source file is copyright (C) 2014 by Alan Kaminsky. All rights
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

import edu.rit.pj2.Task;
import edu.rit.util.Random;

/**
 * Class PiSeq is a sequential program that calculates an approximate value for
 * &pi; using a Monte Carlo technique. The program generates a number of random
 * points in the unit square (0,0) to (1,1) and counts how many of them lie
 * within a circle of radius 1 centered at the origin. The fraction of the
 * points within the circle is approximately &pi;/4.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.example.PiSeq <I>seed</I> <I>N</I></TT>
 * <BR><TT><I>seed</I></TT> = Random seed
 * <BR><TT><I>N</I></TT> = Number of random points
 * <P>
 * The program uses class {@linkplain edu.rit.util.Random} for its pseudorandom
 * number generator.
 *
 * @author  Alan Kaminsky
 * @version 30-May-2014
 */
public class PiSeq
	extends Task
	{

// Program shared variables.

	// Command line arguments.
	long seed;
	long N;

	// Pseudorandom number generator.
	Random prng;

	// Number of points within the unit circle.
	long count;

// Main program.

	/**
	 * Main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Validate command line arguments.
		if (args.length != 2) usage();
		seed = Long.parseLong (args[0]);
		N = Long.parseLong (args[1]);

		// Set up PRNG.
		prng = new Random (seed);

		// Generate n random points in the unit square, count how many are in
		// the unit circle.
		count = 0;
		for (long i = 0; i < N; ++ i)
			{
			double x = prng.nextDouble();
			double y = prng.nextDouble();
			if (x*x + y*y <= 1.0) ++ count;
			}

		// Print results.
		System.out.printf ("pi = 4*%d/%d = %.9f%n", count, N, 4.0*count/N);
		}

// Hidden operations.

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.PiSeq <seed> <N>");
		System.err.println ("<seed> = Random seed");
		System.err.println ("<N> = Number of random points");
		throw new IllegalArgumentException();
		}

	/**
	 * Specify that this task requires one core.
	 */
	protected static int coresRequired()
		{
		return 1;
		}

	}
