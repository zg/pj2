//******************************************************************************
//
// File:    PiGpu.java
// Package: edu.rit.gpu.example
// Unit:    Class edu.rit.gpu.example.PiGpu
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
import edu.rit.pj2.Task;

/**
 * Class PiGpu is a GPU parallel that calculates an approximate value for
 * &pi; using a Monte Carlo technique. The program generates a number of random
 * points in the unit square (0,0) to (1,1) and counts how many of them lie
 * within a circle of radius 1 centered at the origin. The fraction of the
 * points within the circle is approximately &pi;/4.
 * <P>
 * Usage: <TT>java pj2 edu.rit.gpu.example.PiGpu <I>seed</I> <I>N</I></TT>
 * <BR><TT><I>seed</I></TT> = Random seed
 * <BR><TT><I>N</I></TT> = Number of random points
 *
 * @author  Alan Kaminsky
 * @version 30-May-2014
 */
public class PiGpu
	extends Task
	{

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
		long seed = Long.parseLong (args[0]);
		long N = Long.parseLong (args[1]);

		// Initialize GPU.
		Gpu gpu = Gpu.gpu();
		gpu.ensureComputeCapability (2, 0);

		// Set up GPU counter variable.
		Module module = gpu.getModule ("edu/rit/gpu/example/PiGpu.cubin");
		GpuLongVbl count = module.getLongVbl ("devCount");

		// Generate n random points in the unit square, count how many are in
		// the unit circle.
		count.item = 0;
		count.hostToDev();
		PiKernel kernel = module.getKernel (PiKernel.class);
		kernel.setBlockDim (1024);
		kernel.setGridDim (gpu.getMultiprocessorCount());
		kernel.computeRandomPoints (seed, N);

		// Print results.
		count.devToHost();
		System.out.printf ("pi = 4*%d/%d = %.9f%n", count.item, N,
			4.0*count.item/N);
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.gpu.example.PiGpu <seed> <N>");
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

	/**
	 * Specify that this task requires one GPU accelerator.
	 */
	protected static int gpusRequired()
		{
		return 1;
		}

	}
