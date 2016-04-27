//******************************************************************************
//
// File:    StatTestSmp.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.StatTestSmp
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
import edu.rit.pj2.LongLoop;
import edu.rit.pj2.Task;
import edu.rit.pj2.vbl.HistogramVbl;
import edu.rit.util.Random;

/**
 * Class StatTestSmp is a multicore parallel program that performs a statistical
 * test on a pseudorandom number generator.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.example.StatTestSmp <I>seed</I> <I>B</I>
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
public class StatTestSmp
	extends Task
	{
	// Command line arguments.
	long seed;
	int B;
	long N;

	// Global histogram of random numbers.
	HistogramVbl histvbl;

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

		// Set up global histogram.
		histvbl = new HistogramVbl (new Histogram (B)
			{
			public void accumulate (double x)
				{
				increment ((int)(x*size()));
				}
			});

		// Do N trials.
		parallelFor (0, N - 1) .exec (new LongLoop()
			{
			HistogramVbl thrHistvbl;
			Random prng;
			int leap;
			public void start()
				{
				thrHistvbl = threadLocal (histvbl);
				prng = new Random (seed);
				prng.skip (rank());
				leap = threads() - 1;
				}
			public void run (long i)
				{
				thrHistvbl.hist.accumulate (prng.nextDouble());
				prng.skip (leap);
				}
			});

		// Print results.
		System.out.printf ("Bin\tCount%n");
		for (int i = 0; i < B; ++ i)
			System.out.printf ("%d\t%d%n", i, histvbl.hist.count (i));
		double chisqr = histvbl.hist.chisqr();
		System.out.printf ("Chisqr = %.5g%n", chisqr);
		System.out.printf ("Pvalue = %.5g%n", histvbl.hist.pvalue (chisqr));
		}

	// Print a usage message and exit.
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.StatTestSmp <seed> <B> <N>");
		System.err.println ("<seed> = Random seed");
		System.err.println ("<B> = Number of histogram bins");
		System.err.println ("<N> = Number of trials");
		throw new IllegalArgumentException();
		}
	}
