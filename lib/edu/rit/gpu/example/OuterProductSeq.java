//******************************************************************************
//
// File:    OuterProductSeq.java
// Package: edu.rit.gpu.example
// Unit:    Class edu.rit.gpu.example.OuterProductSeq
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

package edu.rit.gpu.example;

import edu.rit.pj2.Task;
import edu.rit.util.Random;

/**
 * Class OuterProductSeq is a sequential program that calculates the outer
 * product matrix of two vectors. The vector elements are set to random values
 * between -10.0 and +10.0. To avert enormous printouts, only selected elements
 * of the vectors and the matrix are printed.
 * <P>
 * Usage: <TT>java pj2 edu.rit.gpu.example.OuterProductSeq <I>seed</I>
 * <I>N</I></TT>
 * <BR><TT><I>seed</I></TT> = Random seed
 * <BR><TT><I>N</I></TT> = Vector length
 *
 * @author  Alan Kaminsky
 * @version 30-May-2014
 */
public class OuterProductSeq
	extends Task
	{

	/**
	 * Task main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		long t1 = System.currentTimeMillis();

		// Validate command line arguments.
		if (args.length != 2) usage();
		long seed = Long.parseLong (args[0]);
		int N = Integer.parseInt (args[1]);

		// Set up input vectors and output matrix.
		Random prng = new Random (seed);
		double[] a = new double [N];
		double[] b = new double [N];
		double[][] c = new double [N] [N];
		for (int i = 0; i < N; ++ i)
			{
			a[i] = prng.nextDouble()*20.0 - 10.0;
			b[i] = prng.nextDouble()*20.0 - 10.0;
			}

		// Compute outer product.
		long t2 = System.currentTimeMillis();
		for (int i = 0; i < N; ++ i)
			for (int j = 0; j < N; ++ j)
				c[i][j] = a[i]*b[j];
		long t3 = System.currentTimeMillis();

		// Print results.
		System.out.printf ("a[%d] = %.5f%n", 0,   a[0  ]);
		System.out.printf ("a[%d] = %.5f%n", 1,   a[1  ]);
		System.out.printf ("a[%d] = %.5f%n", N-2, a[N-2]);
		System.out.printf ("a[%d] = %.5f%n", N-1, a[N-1]);

		System.out.printf ("b[%d] = %.5f%n", 0,   b[0  ]);
		System.out.printf ("b[%d] = %.5f%n", 1,   b[1  ]);
		System.out.printf ("b[%d] = %.5f%n", N-2, b[N-2]);
		System.out.printf ("b[%d] = %.5f%n", N-1, b[N-1]);

		System.out.printf ("c[%d][%d] = %.5f%n", 0,   0,   c[0  ][0  ]);
		System.out.printf ("c[%d][%d] = %.5f%n", 0,   1,   c[0  ][1  ]);
		System.out.printf ("c[%d][%d] = %.5f%n", 0,   N-2, c[0  ][N-2]);
		System.out.printf ("c[%d][%d] = %.5f%n", 0,   N-1, c[0  ][N-1]);
		System.out.printf ("c[%d][%d] = %.5f%n", 1,   0,   c[1  ][0  ]);
		System.out.printf ("c[%d][%d] = %.5f%n", 1,   1,   c[1  ][1  ]);
		System.out.printf ("c[%d][%d] = %.5f%n", 1,   N-2, c[1  ][N-2]);
		System.out.printf ("c[%d][%d] = %.5f%n", 1,   N-1, c[1  ][N-1]);
		System.out.printf ("c[%d][%d] = %.5f%n", N-2, 0,   c[N-2][0  ]);
		System.out.printf ("c[%d][%d] = %.5f%n", N-2, 1,   c[N-2][1  ]);
		System.out.printf ("c[%d][%d] = %.5f%n", N-2, N-2, c[N-2][N-2]);
		System.out.printf ("c[%d][%d] = %.5f%n", N-2, N-1, c[N-2][N-1]);
		System.out.printf ("c[%d][%d] = %.5f%n", N-1, 0,   c[N-1][0  ]);
		System.out.printf ("c[%d][%d] = %.5f%n", N-1, 1,   c[N-1][1  ]);
		System.out.printf ("c[%d][%d] = %.5f%n", N-1, N-2, c[N-1][N-2]);
		System.out.printf ("c[%d][%d] = %.5f%n", N-1, N-1, c[N-1][N-1]);

		// Print running times.
		long t4 = System.currentTimeMillis();
		System.out.printf ("%d msec pre%n", t2 - t1);
		System.out.printf ("%d msec calc%n", t3 - t2);
		System.out.printf ("%d msec post%n", t4 - t3);
		System.out.printf ("%d msec total%n", t4 - t1);
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.gpu.example.OuterProductSeq <seed> <N>");
		System.err.println ("<seed> = Random seed");
		System.err.println ("<N> = Vector length");
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
