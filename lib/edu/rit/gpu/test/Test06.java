//******************************************************************************
//
// File:    Test06.java
// Package: edu.rit.gpu.test
// Unit:    Class edu.rit.gpu.test.Test06
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

package edu.rit.gpu.test;

import edu.rit.gpu.Kernel;
import edu.rit.gpu.Gpu;
import edu.rit.gpu.GpuDoubleMatrix;
import edu.rit.gpu.Module;
import edu.rit.pj2.Task;
import edu.rit.pj2.TerminateException;

/**
 * Class Test06 is a unit test main program that adds two double matrices on the
 * GPU.
 * <P>
 * Usage: <TT>java pj2 edu.rit.gpu.test.Test06 <I>rows</I> <I>cols</I></TT>
 * <BR><TT><I>rows</I></TT> = Number of matrix rows
 * <BR><TT><I>cols</I></TT> = Number of matrix columns
 *
 * @author  Alan Kaminsky
 * @version 30-May-2014
 */
public class Test06
	extends Task
	{

	/**
	 * GPU kernel interface.
	 */
	private static interface Test06Kernel
		extends Kernel
		{
		public void addDoubleMatrices
			(GpuDoubleMatrix a,
			 GpuDoubleMatrix b,
			 GpuDoubleMatrix c,
			 int rows,
			 int cols);
		}

	/**
	 * Task main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Verify command line arguments.
		if (args.length != 2) usage();
		int rows = Integer.parseInt (args[0]);
		int cols = Integer.parseInt (args[1]);

		// Allocate matrices in GPU memory.
		Gpu gpu = Gpu.gpu();
		GpuDoubleMatrix a = gpu.getDoubleMatrix (rows, cols);
		GpuDoubleMatrix b = gpu.getDoubleMatrix (rows, cols);
		GpuDoubleMatrix c = gpu.getDoubleMatrix (rows, cols);

		// Initialize input matrices.
		for (int row = 0; row < rows; ++ row)
			for (int col = 0; col < cols; ++ col)
				{
				a.item[row][col] = 1.0*(row + 1) + 0.001*(col + 1);
				b.item[row][col] = 2.0*(row + 1) + 0.002*(col + 1);
				}
		a.hostToDev();
		b.hostToDev();

		// Launch GPU kernel.
		Module module = gpu.getModule ("edu/rit/gpu/test/Test06.ptx");
		Test06Kernel kernel = module.getKernel (Test06Kernel.class);
		kernel.setBlockDim (16, 16);
		kernel.setGridDim ((cols + 15)/16, (rows + 15)/16);
		kernel.addDoubleMatrices (a, b, c, rows, cols);

		// Print results.
		c.devToHost();
		for (int row = 0; row < rows; ++ row)
			for (int col = 0; col < cols; ++ col)
				System.out.printf
					("a[%d][%d]=%.3f\tb[%d][%d]=%.3f\tc[%d][%d]=%.3f%n",
					row, col, a.item[row][col],
					row, col, b.item[row][col],
					row, col, c.item[row][col]);
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.gpu.test.Test06 <rows> <cols>");
		System.err.println ("<rows> = Number of matrix rows");
		System.err.println ("<cols> = Number of matrix columns");
		throw new TerminateException();
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
