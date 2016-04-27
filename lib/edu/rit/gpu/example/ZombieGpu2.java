//******************************************************************************
//
// File:    ZombieGpu2.java
// Package: edu.rit.gpu.example
// Unit:    Class edu.rit.gpu.example.ZombieGpu2
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
import edu.rit.gpu.GpuDoubleVbl;
import edu.rit.gpu.GpuStructArray;
import edu.rit.gpu.Kernel;
import edu.rit.gpu.Module;
import edu.rit.gpu.Struct;
import edu.rit.pj2.Task;
import edu.rit.util.Random;
import java.nio.ByteBuffer;

/**
 * Class ZombieGpu2 is a GPU parallel program to compute the motion of a group
 * of zombies. This is a so-called "<I>N</I>-bodies" problem. Class ZombieGpu2
 * uses a two-dimensional vector object to represent the zombies' positions and
 * velocities.
 * <P>
 * Usage: <TT>java pj2 edu.rit.gpu.example.ZombieGpu2 <I>seed</I> <I>N</I>
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
 * @version 28-Oct-2014
 */
public class ZombieGpu2
	extends Task
	{

	// Structure for a 2-D vector.
	private static class Vector
		extends Struct
		{
		public double x;
		public double y;

		// Construct a new vector.
		public Vector
			(double x,
			 double y)
			{
			this.x = x;
			this.y = y;
			}

		// Returns the size in bytes of the C struct.
		public static long sizeof()
			{
			return 16;
			}

		// Write this Java object to the given byte buffer as a C struct.
		public void toStruct
			(ByteBuffer buf)
			{
			buf.putDouble (x);
			buf.putDouble (y);
			}

		// Read this Java object from the given byte buffer as a C struct.
		public void fromStruct
			(ByteBuffer buf)
			{
			x = buf.getDouble();
			y = buf.getDouble();
			}
		}

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
	GpuStructArray<Vector> pos;

	// Next body positions.
	GpuStructArray<Vector> next;

	// For detecting convergence.
	GpuDoubleVbl delta;

	/**
	 * Kernel function interface.
	 */
	private static interface ZombieKernel
		extends Kernel
		{
		public void timeStep
			(GpuStructArray<Vector> pos,
			 GpuStructArray<Vector> next,
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
		Module module = gpu.getModule ("edu/rit/gpu/example/ZombieGpu2.cubin");
		pos = gpu.getStructArray (Vector.class, N);
		next = gpu.getStructArray (Vector.class, N);
		delta = module.getDoubleVbl ("devDelta");

		// Set up GPU kernel.
		ZombieKernel kernel = module.getKernel (ZombieKernel.class);
		kernel.setBlockDim (256);
		kernel.setGridDim (N);
		kernel.setCacheConfig (CacheConfig.CU_FUNC_CACHE_PREFER_L1);

		// Set zombies' initial (x,y) coordinates at random in a WxW square
		// region. Also allocate zombies' next positions.
		Random prng = new Random (seed);
		for (int i = 0; i < N; ++ i)
			{
			pos.item[i] = new Vector (prng.nextDouble()*W, prng.nextDouble()*W);
			next.item[i] = new Vector (0, 0);
			}
		pos.hostToDev();

		// Snapshot all bodies' initial positions.
		int t = 0;
		snapshot (t);

		// Do repeated time steps.
		for (;;)
			{
			// Do one time step.
			delta.item = 0.0;
			delta.hostToDev();
			kernel.timeStep (pos, next, N, G, L, dt);

			// Advance to next time step.
			++ t;

			// Update positions.
			GpuStructArray<Vector> tmp;
			tmp = pos; pos = next; next = tmp;

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
		pos.devToHost();
		for (int i = 0; i < N; ++ i)
			System.out.printf ("%d\t%d\t%g\t%g%n",
				t, i, pos.item[i].x, pos.item[i].y);
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.ZombieGpu2 <seed> <N> <W> <G> <L> <dt> <eps> <steps> <snap>");
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
