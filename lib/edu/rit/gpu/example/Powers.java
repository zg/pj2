//******************************************************************************
//
// File:    Powers.java
// Package: edu.rit.gpu.example
// Unit:    Class edu.rit.gpu.example.Powers
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
import edu.rit.gpu.GpuStructArray;
import edu.rit.gpu.Module;
import edu.rit.gpu.Struct;
import edu.rit.pj2.Task;
import java.nio.ByteBuffer;

/**
 * Class Powers is a GPU parallel program that calculates the square root, cube
 * root, square, and cube of the numbers from 1 to <I>N</I>. The program
 * illustrates the use of C structs in Java GPU programs.
 * <P>
 * Usage: <TT>java pj2 edu.rit.gpu.example.Powers <I>N</I></TT>
 *
 * @author  Alan Kaminsky
 * @version 03-Jun-2014
 */
public class Powers
	extends Task
	{

	/**
	 * Java object for C struct containing results of one calculation.
	 */
	private static class powers_t
		extends Struct
		{
		public int x;          // 4 bytes
		                       // 4 bytes padding for alignment
		public double sqrt_x;  // 8 bytes
		public double curt_x;  // 8 bytes
		public double x_sq;    // 8 bytes
		public double x_cu;    // 8 bytes
		                       // 40 bytes total

		public powers_t (int x)
			{
			this.x = x;
			}

		public static long sizeof()
			{
			return 40L;
			}

		public void toStruct (ByteBuffer buf)
			{
			buf.putInt (x);
			buf.putInt (0); // Padding
			buf.putDouble (sqrt_x);
			buf.putDouble (curt_x);
			buf.putDouble (x_sq);
			buf.putDouble (x_cu);
			}

		public void fromStruct (ByteBuffer buf)
			{
			x = buf.getInt();
			buf.getInt(); // Padding
			sqrt_x = buf.getDouble();
			curt_x = buf.getDouble();
			x_sq = buf.getDouble();
			x_cu = buf.getDouble();
			}
		}

	/**
	 * Kernel function interface.
	 */
	private static interface PowersKernel
		extends Kernel
		{
		public void computePowers
			(GpuStructArray<powers_t> powers,
			 int N);
		}

	/**
	 * Task main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Validate command line arguments.
		if (args.length != 1) usage();
		int N = Integer.parseInt (args[0]);

		// Initialize GPU.
		Gpu gpu = Gpu.gpu();
		gpu.ensureComputeCapability (2, 0);

		// Initialize results array.
		GpuStructArray<powers_t> powers =
			gpu.getStructArray (powers_t.class, N);
		for (int i = 0; i < N; ++ i)
			powers.item[i] = new powers_t (i + 1);
		powers.hostToDev();

		// Compute results.
		Module module = gpu.getModule ("edu/rit/gpu/example/Powers.cubin");
		PowersKernel kernel = module.getKernel (PowersKernel.class);
		kernel.setBlockDim (1024);
		kernel.setGridDim (gpu.getMultiprocessorCount());
		kernel.computePowers (powers, N);

		// Print results.
		powers.devToHost();
		System.out.printf ("x\tsqrt(x)\tcurt(x)\tx^2\tx^3%n");
		for (int i = 0; i < N; ++ i)
			{
			powers_t p = powers.item[i];
			System.out.printf ("%d\t%.5e\t%.5e\t%.5e\t%.5e%n",
				p.x, p.sqrt_x, p.curt_x, p.x_sq, p.x_cu);
			}
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.gpu.example.Powers <N>");
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
