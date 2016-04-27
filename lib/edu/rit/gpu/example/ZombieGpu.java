//******************************************************************************
//
// File:    ZombieGpu.java
// Package: edu.rit.gpu.example
// Unit:    Class edu.rit.gpu.example.ZombieGpu
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
import edu.rit.gpu.Gpu;
import edu.rit.gpu.GpuDoubleArray;
import edu.rit.gpu.GpuDoubleVbl;
import edu.rit.gpu.Kernel;
import edu.rit.gpu.Module;
import edu.rit.pj2.Task;
import edu.rit.util.Random;

/**
 * Class ZombieGpu is a GPU parallel program to compute the motion of a group of
 * zombies. This is a so-called "<I>N</I>-bodies" problem.
 * <P>
 * Usage: <TT>java pj2 edu.rit.gpu.example.ZombieGpu <I>seed</I> <I>N</I>
 * <I>W</I> <I>G</I> <I>L</I> <I>dt</I> <I>eps</I> <I>steps</I> <I>snap</I></TT>
 * <BR><TT><I>seed</I></TT> = Random seed
 * <BR><TT><I>N</I></TT> = Number of bodies
 * <BR><TT><I>W</I></TT> = Region size
 * <BR><TT><I>G</I></TT> = Attraction factor
 * <BR><TT><I>L</I></TT> = Attraction length scale
 * <BR><TT><I>dt</I></TT> = Time step size
 * <BR><TT><I>eps</I></TT> = Convergence threshold
 * <BR><TT><I>steps</I></TT> = Number of time steps (0 = until convergence)
 * <BR><TT><I>snap</I></TT> = Snapshot interval (0 = none)
 *
 * @author  Alan Kaminsky
 * @version 10-Oct-2014
 */
public class ZombieGpu
	extends Task
	{

	// Command line arguments.
	long seed;
	int N;
	double W;
	double G;
	double L;
	double dt;
	double eps;
	int steps;
	int snap;

	// Current body positions.
	GpuDoubleArray xpos;
	GpuDoubleArray ypos;

	// Next body positions.
	GpuDoubleArray xnext;
	GpuDoubleArray ynext;

	// For detecting convergence.
	GpuDoubleVbl delta;

	/**
	 * Kernel function interface.
	 */
	private static interface ZombieKernel
		extends Kernel
		{
		public void timeStep
			(GpuDoubleArray xpos,
			 GpuDoubleArray ypos,
			 GpuDoubleArray xnext,
			 GpuDoubleArray ynext,
			 int N,
			 double G,
			 double L,
			 double dt);
		}

	/**
	 * Task main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Parse command line arguments.
		if (args.length != 9) usage();
		seed = Long.parseLong (args[0]);
		N = Integer.parseInt (args[1]);
		W = Double.parseDouble (args[2]);
		G = Double.parseDouble (args[3]);
		L = Double.parseDouble (args[4]);
		dt = Double.parseDouble (args[5]);
		eps = Double.parseDouble (args[6]);
		steps = Integer.parseInt (args[7]);
		snap = Integer.parseInt (args[8]);

		// Initialize GPU.
		Gpu gpu = Gpu.gpu();
		gpu.ensureComputeCapability (2, 0);

		// Set up GPU variables.
		Module module = gpu.getModule ("edu/rit/gpu/example/ZombieGpu.cubin");
		xpos = gpu.getDoubleArray (N);
		ypos = gpu.getDoubleArray (N);
		xnext = gpu.getDoubleArray (N);
		ynext = gpu.getDoubleArray (N);
		delta = module.getDoubleVbl ("devDelta");

		// Set up GPU kernel.
		ZombieKernel kernel = module.getKernel (ZombieKernel.class);
		kernel.setBlockDim (256);
		kernel.setGridDim (N);
		kernel.setCacheConfig (CacheConfig.CU_FUNC_CACHE_PREFER_L1);

		// Set zombies' initial (x,y) coordinates at random in a WxW square
		// region.
		Random prng = new Random (seed);
		for (int i = 0; i < N; ++ i)
			{
			xpos.item[i] = prng.nextDouble()*W;
			ypos.item[i] = prng.nextDouble()*W;
			}
		xpos.hostToDev();
		ypos.hostToDev();

		// Snapshot all bodies' initial positions.
		int t = 0;
		snapshot (t);

		// Do repeated time steps.
		for (;;)
			{
			// Do one time step.
			delta.item = 0.0;
			delta.hostToDev();
			kernel.timeStep (xpos, ypos, xnext, ynext, N, G, L, dt);

			// Advance to next time step.
			++ t;

			// Update positions.
			GpuDoubleArray tmp;
			tmp = xpos; xpos = xnext; xnext = tmp;
			tmp = ypos; ypos = ynext; ynext = tmp;

			// Stop when position delta is less than convergence threshold or
			// when the specified number of time steps have occurred.
			delta.devToHost();
			if ((steps == 0 && delta.item < eps) || (steps != 0 && t == steps))
				break;

			// Snapshot all bodies' positions every <snap> time steps.
			if (snap > 0 && (t % snap) == 0)
				snapshot (t);
			}

		// Snapshot all bodies' final positions.
		snapshot (t);
		}

	/**
	 * Snapshot all bodies' positions.
	 */
	private void snapshot
		(int t)
		{
		xpos.devToHost();
		ypos.devToHost();
		for (int i = 0; i < N; ++ i)
			System.out.printf ("%d\t%d\t%g\t%g%n",
				t, i, xpos.item[i], ypos.item[i]);
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.ZombieGpu <seed> <N> <W> <G> <L> <dt> <eps> <steps> <snap>");
		System.err.println ("<seed> = Random seed");
		System.err.println ("<N> = Number of bodies");
		System.err.println ("<W> = Region size");
		System.err.println ("<G> = Attraction factor");
		System.err.println ("<L> = Attraction length scale");
		System.err.println ("<dt> = Time step size");
		System.err.println ("<eps> = Convergence threshold");
		System.err.println ("<steps> = Number of time steps (0 = until convergence)");
		System.err.println ("<snap> = Snapshot interval (0 = none)");
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
