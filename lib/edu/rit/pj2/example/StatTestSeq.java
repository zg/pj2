//******************************************************************************
//
// File:    StatTestSeq.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.StatTestSeq
//
// This Java source file is copyright (C) 2015 by Alan Kaminsky. All rights
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

import edu.rit.numeric.Histogram;
import edu.rit.pj2.Task;
import edu.rit.util.Random;

/**
 * Class StatTestSeq is a sequential program that performs a statistical test on
 * a pseudorandom number generator.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.example.StatTestSeq <I>seed</I> <I>B</I>
 * <I>N</I></TT>
 * <BR><TT><I>seed</I></TT> = Random seed
 * <BR><TT><I>B</I></TT> = Number of histogram bins
 * <BR><TT><I>N</I></TT> = Number of trials
 * <P>
 * The program uses class {@linkplain edu.rit.util.Random} for its pseudorandom
 * number generator.
 *
 * @author  Alan Kaminsky
 * @version 26-Mar-2015
 */
public class StatTestSeq
	extends Task
	{
	// Command line arguments.
	long seed;
	int B;
	long N;

	// Pseudorandom number generator.
	Random prng;

	// Histogram of random numbers.
	Histogram hist;

	// Main program.
	public void main
		(String[] args)
		throws Exception
		{
		// Validate command line arguments.
		if (args.length != 3) usage();
		seed = Long.parseLong (args[0]);
		B = Integer.parseInt (args[1]);
		N = Long.parseLong (args[2]);

		// Set up PRNG.
		prng = new Random (seed);

		// Set up histogram.
		hist = new Histogram (B)
			{
			public void accumulate (double x)
				{
				increment ((int)(x*size()));
				}
			};

		// Do N trials.
		for (long i = 0; i < N; ++ i)
			hist.accumulate (prng.nextDouble());

		// Print results.
		System.out.printf ("Bin\tCount%n");
		for (int i = 0; i < B; ++ i)
			System.out.printf ("%d\t%d%n", i, hist.count (i));
		double chisqr = hist.chisqr();
		System.out.printf ("Chisqr = %.5g%n", chisqr);
		System.out.printf ("Pvalue = %.5g%n", hist.pvalue (chisqr));
		}

	// Print a usage message and exit.
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.StatTestSeq <seed> <B> <N>");
		System.err.println ("<seed> = Random seed");
		System.err.println ("<B> = Number of histogram bins");
		System.err.println ("<N> = Number of trials");
		throw new IllegalArgumentException();
		}

	// Specify that this task requires one core.
	protected static int coresRequired()
		{
		return 1;
		}
	}
