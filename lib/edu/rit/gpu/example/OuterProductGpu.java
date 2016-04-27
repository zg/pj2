//******************************************************************************
//
// File:    OuterProductGpu.java
// Package: edu.rit.gpu.example
// Unit:    Class edu.rit.gpu.example.OuterProductGpu
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

import edu.rit.gpu.CacheConfig;
import edu.rit.gpu.Kernel;
import edu.rit.gpu.Gpu;
import edu.rit.gpu.GpuDoubleArray;
import edu.rit.gpu.GpuDoubleMatrix;
import edu.rit.gpu.Module;
import edu.rit.pj2.Task;
import edu.rit.util.Random;

/**
 * Class OuterProductGpu is a GPU parallel program that calculates the outer
 * product matrix of two vectors. The vector elements are set to random values
 * between -10.0 and +10.0. To avert enormous printouts, only selected elements
 * of the vectors and the matrix are printed.
 * <P>
 * Usage: <TT>java pj2 edu.rit.gpu.example.OuterProductGpu <I>seed</I>
 * <I>N</I></TT>
 * <BR><TT><I>seed</I></TT> = Random seed
 * <BR><TT><I>N</I></TT> = Vector length
 *
 * @author  Alan Kaminsky
 * @version 30-May-2014
 */
public class OuterProductGpu
	extends Task
	{

	/**
	 * Kernel function interface.
	 */
	private static interface OuterProductKernel
		extends Kernel
		{
		public void outerProduct
			(GpuDoubleArray a,
			 GpuDoubleArray b,
			 GpuDoubleMatrix c,
			 int N);
		}

	/**
	 * GPU kernel block dimensions = NT x NT threads.
	 */
	private static final int NT = 32;

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

		// Initialize GPU.
		Gpu gpu = Gpu.gpu();
		gpu.ensureComputeCapability (2, 0);

		// Set up input vectors and output matrix.
		Random prng = new Random (seed);
		GpuDoubleArray a = gpu.getDoubleArray (N);
		GpuDoubleArray b = gpu.getDoubleArray (N);
		GpuDoubleMatrix c = gpu.getDoubleMatrix (N, N);
		for (int i = 0; i < N; ++ i)
			{
			a.item[i] = prng.nextDouble()*20.0 - 10.0;
			b.item[i] = prng.nextDouble()*20.0 - 10.0;
			}
		a.hostToDev();
		b.hostToDev();

		// Compute outer product.
		Module module =
			gpu.getModule ("edu/rit/gpu/example/OuterProductGpu.cubin");
		OuterProductKernel kernel =
			module.getKernel (OuterProductKernel.class);
		kernel.setBlockDim (NT, NT);
		kernel.setGridDim ((N + NT - 1)/NT, (N + NT - 1)/NT); 
		kernel.setCacheConfig (CacheConfig.CU_FUNC_CACHE_PREFER_L1);
		long t2 = System.currentTimeMillis();
		kernel.outerProduct (a, b, c, N);
		long t3 = System.currentTimeMillis();

		// Print results.
		c.devToHost();

		System.out.printf ("a[%d] = %.5f%n", 0,   a.item[0  ]);
		System.out.printf ("a[%d] = %.5f%n", 1,   a.item[1  ]);
		System.out.printf ("a[%d] = %.5f%n", N-2, a.item[N-2]);
		System.out.printf ("a[%d] = %.5f%n", N-1, a.item[N-1]);

		System.out.printf ("b[%d] = %.5f%n", 0,   b.item[0  ]);
		System.out.printf ("b[%d] = %.5f%n", 1,   b.item[1  ]);
		System.out.printf ("b[%d] = %.5f%n", N-2, b.item[N-2]);
		System.out.printf ("b[%d] = %.5f%n", N-1, b.item[N-1]);

		System.out.printf ("c[%d][%d] = %.5f%n", 0,   0,   c.item[0  ][0  ]);
		System.out.printf ("c[%d][%d] = %.5f%n", 0,   1,   c.item[0  ][1  ]);
		System.out.printf ("c[%d][%d] = %.5f%n", 0,   N-2, c.item[0  ][N-2]);
		System.out.printf ("c[%d][%d] = %.5f%n", 0,   N-1, c.item[0  ][N-1]);
		System.out.printf ("c[%d][%d] = %.5f%n", 1,   0,   c.item[1  ][0  ]);
		System.out.printf ("c[%d][%d] = %.5f%n", 1,   1,   c.item[1  ][1  ]);
		System.out.printf ("c[%d][%d] = %.5f%n", 1,   N-2, c.item[1  ][N-2]);
		System.out.printf ("c[%d][%d] = %.5f%n", 1,   N-1, c.item[1  ][N-1]);
		System.out.printf ("c[%d][%d] = %.5f%n", N-2, 0,   c.item[N-2][0  ]);
		System.out.printf ("c[%d][%d] = %.5f%n", N-2, 1,   c.item[N-2][1  ]);
		System.out.printf ("c[%d][%d] = %.5f%n", N-2, N-2, c.item[N-2][N-2]);
		System.out.printf ("c[%d][%d] = %.5f%n", N-2, N-1, c.item[N-2][N-1]);
		System.out.printf ("c[%d][%d] = %.5f%n", N-1, 0,   c.item[N-1][0  ]);
		System.out.printf ("c[%d][%d] = %.5f%n", N-1, 1,   c.item[N-1][1  ]);
		System.out.printf ("c[%d][%d] = %.5f%n", N-1, N-2, c.item[N-1][N-2]);
		System.out.printf ("c[%d][%d] = %.5f%n", N-1, N-1, c.item[N-1][N-1]);

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
		System.err.println ("Usage: java pj2 edu.rit.gpu.example.OuterProductGpu <seed> <N>");
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

	/**
	 * Specify that this task requires one GPU accelerator.
	 */
	protected static int gpusRequired()
		{
		return 1;
		}

	}
