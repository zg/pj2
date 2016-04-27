//******************************************************************************
//
// File:    PiGpu2.java
// Package: edu.rit.gpu.example
// Unit:    Class edu.rit.gpu.example.PiGpu2
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

import edu.rit.gpu.Kernel;
import edu.rit.gpu.Gpu;
import edu.rit.gpu.GpuLongVbl;
import edu.rit.gpu.Module;
import edu.rit.pj2.LongChunk;
import edu.rit.pj2.Section;
import edu.rit.pj2.Task;
import edu.rit.pj2.vbl.LongVbl;

/**
 * Class PiGpu2 is a GPU parallel program that calculates an approximate value
 * for &pi; using a Monte Carlo technique. The program generates a number of
 * random points in the unit square (0,0) to (1,1) and counts how many of them
 * lie within a circle of radius 1 centered at the origin. The fraction of the
 * points within the circle is approximately &pi;/4. The program runs on a
 * single node, using all the GPUs on the node.
 * <P>
 * Usage: <TT>java pj2 edu.rit.gpu.example.PiGpu2 <I>seed</I> <I>N</I></TT>
 * <BR><TT><I>seed</I></TT> = Random seed
 * <BR><TT><I>N</I></TT> = Number of random points
 *
 * @author  Alan Kaminsky
 * @version 04-Jun-2014
 */
public class PiGpu2
	extends Task
	{

	private static long seed;
	private static long N;
	private static LongVbl count;

	/**
	 * Kernel function interface.
	 */
	private static interface PiKernel
		extends Kernel
		{
		public void computeRandomPoints
			(long seed,
			 long N);
		}

	/**
	 * Task main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Validate command line arguments.
		if (args.length != 2) usage();
		seed = Long.parseLong (args[0]);
		N = Long.parseLong (args[1]);

		// Set up global counter variable.
		count = new LongVbl.Sum (0);

		// Run one CPU thread for each GPU on the node.
		parallelDo (Gpu.allowedDeviceCount(), new Section()
			{
			public void run() throws Exception
				{
				// Set up per-thread counter variable.
				LongVbl thrCount = threadLocal (count);

				// Initialize per-thread GPU.
				Gpu gpu = Gpu.gpu();
				gpu.ensureComputeCapability (2, 0);

				// Set up GPU counter variable.
				Module module = gpu.getModule
					("edu/rit/gpu/example/PiGpu.cubin");
				GpuLongVbl devCount = module.getLongVbl ("devCount");

				// Determine how many of the N points this thread will compute.
				long thrN = LongChunk.partition (0, N - 1, threads(), rank())
					.length() .longval();

				// Generate thrN random points in the unit square, count how
				// many are in the unit circle.
				devCount.item = 0;
				devCount.hostToDev();
				PiKernel kernel = module.getKernel (PiKernel.class);
				kernel.setBlockDim (1024);
				kernel.setGridDim (gpu.getMultiprocessorCount());
				kernel.computeRandomPoints (seed + 1000000L*rank(), thrN);

				// Get per-thread count, automatically reduced into global
				// count.
				devCount.devToHost();
				thrCount.item = devCount.item;
				}
			});

		// Print results.
		System.out.printf ("pi = 4*%d/%d = %.9f%n", count.item, N,
			4.0*count.item/N);
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.gpu.example.PiGpu2 <seed> <N>");
		System.err.println ("<seed> = Random seed");
		System.err.println ("<N> = Number of random points");
		throw new IllegalArgumentException();
		}

	/**
	 * Specify that this task requires one core. (Parallel team threads will
	 * share the core.)
	 */
	protected static int coresRequired()
		{
		return 1;
		}

	/**
	 * Specify that this task requires all GPU accelerators on the node.
	 */
	protected static int gpusRequired()
		{
		return ALL_GPUS;
		}

	}
