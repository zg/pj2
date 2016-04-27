//******************************************************************************
//
// File:    ZombieClu.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.ZombieClu
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

package edu.rit.pj2.example;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.pj2.Chunk;
import edu.rit.pj2.Job;
import edu.rit.pj2.Loop;
import edu.rit.pj2.Task;
import edu.rit.pj2.Tuple;
import edu.rit.pj2.vbl.DoubleVbl;
import edu.rit.util.Random;
import java.io.IOException;
import static java.lang.Math.*;

/**
 * Class ZombieClu is a cluster parallel program to compute the motion of a
 * group of zombies. This is a so-called "<I>N</I>-bodies" problem.
 * <P>
 * Usage: <TT>java pj2 [workers=<I>K</I>] edu.rit.pj2.example.ZombieClu
 * <I>seed</I> <I>N</I> <I>W</I> <I>G</I> <I>L</I> <I>dt</I> <I>eps</I>
 * <I>steps</I> <I>snap</I></TT>
 * <BR><TT><I>K</I></TT> = Number of worker tasks (default: 1)
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
 * @version 08-Jul-2014
 */
public class ZombieClu
	extends Job
	{

	/**
	 * Job main program.
	 */
	public void main
		(String[] args)
		{
		// Parse command line arguments.
		if (args.length != 9) usage();
		long seed = Long.parseLong (args[0]);
		int N = Integer.parseInt (args[1]);
		double W = Double.parseDouble (args[2]);
		double G = Double.parseDouble (args[3]);
		double L = Double.parseDouble (args[4]);
		double dt = Double.parseDouble (args[5]);
		double eps = Double.parseDouble (args[6]);
		int steps = Integer.parseInt (args[7]);
		int snap = Integer.parseInt (args[8]);

		// Set up a task group of K worker tasks.
		int K = workers();
		if (K == DEFAULT_WORKERS) K = 1;
		rule() .task (K, WorkerTask.class) .args (args);

		// Set up snapshot task.
		rule() .task (SnapshotTask.class) .args (args) .args (""+K)
			.runInJobProcess();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 [workers=<K>] edu.rit.pj2.example.ZombieClu <seed> <N> <W> <G> <L> <dt> <eps> <steps> <snap>");
		System.err.println ("<K> = Number of worker tasks (default: 1)");
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
	 * Tuple with results of one time step.
	 */
	private static class ZombieTuple
		extends Tuple
		{
		public int rank;     // Worker task rank
		public int step;     // Time step number
		public int lb;       // Lower bound zombie index
		public double[] x;   // Zombie X coordinates
		public double[] y;   // Zombie Y coordinates
		public double delta; // Zombie position delta

		public ZombieTuple()
			{
			}

		public ZombieTuple
			(int rank,
			 int step,
			 int lb,
			 double[] x,
			 double[] y,
			 double delta)
			{
			this.rank = rank;
			this.step = step;
			this.lb = lb;
			this.x = x;
			this.y = y;
			this.delta = delta;
			}

		public boolean matchContent
			(Tuple target)
			{
			ZombieTuple t = (ZombieTuple) target;
			return this.rank == t.rank && this.step == t.step;
			}

		public void writeOut
			(OutStream out)
			throws IOException
			{
			out.writeInt (rank);
			out.writeInt (step);
			out.writeInt (lb);
			out.writeDoubleArray (x);
			out.writeDoubleArray (y);
			out.writeDouble (delta);
			}

		public void readIn
			(InStream in)
			throws IOException
			{
			rank = in.readInt();
			step = in.readInt();
			lb = in.readInt();
			x = in.readDoubleArray();
			y = in.readDoubleArray();
			delta = in.readDouble();
			}
		}

	/**
	 * Worker task class.
	 */
	private static class WorkerTask
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

		// Task group size and worker rank.
		int size;
		int rank;

		// Current zombie positions.
		double[] x;
		double[] y;

		// Next zombie positions.
		int lb, ub, len;
		double[] xnext;
		double[] ynext;

		// For detecting convergence.
		DoubleVbl delta = new DoubleVbl.Sum();

		// Task main program.
		public void main
			(String[] args)
			throws Exception
			{
			// Parse command line arguments.
			seed = Long.parseLong (args[0]);
			N = Integer.parseInt (args[1]);
			W = Double.parseDouble (args[2]);
			G = Double.parseDouble (args[3]);
			L = Double.parseDouble (args[4]);
			dt = Double.parseDouble (args[5]);
			eps = Double.parseDouble (args[6]);
			steps = Integer.parseInt (args[7]);

			// Get task group size and worker rank.
			size = groupSize();
			rank = taskRank();

			// Allocate storage for current positions for all zombies.
			x = new double [N];
			y = new double [N];

			// Allocate storage for next positions for just this worker's slice
			// of zombies.
			Chunk slice = Chunk.partition (0, N - 1, size, rank);
			lb = slice.lb();
			ub = slice.ub();
			len = (int) slice.length();
			xnext = new double [len];
			ynext = new double [len];

			// Initialize zombies' (x,y) coordinates.
			Random prng = new Random (seed);
			for (int i = 0; i < N; ++ i)
				{
				x[i] = prng.nextDouble()*W;
				y[i] = prng.nextDouble()*W;
				}

			// Do time steps.
			int t = 0;
			ZombieTuple template = new ZombieTuple();
			ZombieTuple zt = null;
			for (;;)
				{
				delta.item = 0.0;

				// Do each zombie i for this worker's slice of zombies.
				parallelFor (lb, ub) .exec (new Loop()
					{
					DoubleVbl thrDelta;

					public void start()
						{
						thrDelta = threadLocal (delta);
						}

					public void run (int i)
						{
						double vx = 0.0;
						double vy = 0.0;
						double dx, dy, d, v;

						// Accumulate velocity due to every other body j.
						for (int j = 0; j < N; ++ j)
							{
							if (j == i) continue;
							dx = x[j] - x[i];
							dy = y[j] - y[i];
							d = sqrt(dx*dx + dy*dy);
							v = G*exp(-d/L) - exp(-d);
							vx += v*dx/d;
							vy += v*dy/d;
							}

						// Move body i in the direction of its velocity.
						dx = vx*dt;
						dy = vy*dt;
						xnext[i-lb] = x[i] + dx;
						ynext[i-lb] = y[i] + dy;

						// Accumulate position delta.
						thrDelta.item += abs(dx) + abs(dy);
						}
					});

				// Advance to next time step.
				++ t;

				// Send new zombie positions to the other workers and to the
				// snapshot task.
				putTuple (size, new ZombieTuple
					(rank, t, lb, xnext, ynext, delta.item));

				// Receive new zombie positions from the other workers.
				double totalDelta = 0.0;
				template.step = t;
				for (int i = 0; i < size; ++ i)
					{
					if (i == rank)
						{
						System.arraycopy (xnext, 0, x, lb, len);
						System.arraycopy (ynext, 0, y, lb, len);
						totalDelta += delta.item;
						}
					else
						{
						template.rank = i;
						zt = takeTuple (template);
						System.arraycopy (zt.x, 0, x, zt.lb, zt.x.length);
						System.arraycopy (zt.y, 0, y, zt.lb, zt.y.length);
						totalDelta += zt.delta;
						}
					}

				// Stop when position delta is less than convergence threshold
				// or when the specified number of time steps have occurred.
				if ((steps == 0 && totalDelta < eps) ||
					(steps != 0 && t == steps))
						break;
				}
			}
		}

	/**
	 * Snapshot task class.
	 */
	private static class SnapshotTask
		extends Task
		{
		// Task main program.
		public void main
			(String[] args)
			throws Exception
			{
			// Parse command line arguments.
			long seed = Long.parseLong (args[0]);
			int N = Integer.parseInt (args[1]);
			double W = Double.parseDouble (args[2]);
			double eps = Double.parseDouble (args[6]);
			int steps = Integer.parseInt (args[7]);
			int snap = Integer.parseInt (args[8]);
			int K = Integer.parseInt (args[9]);

			// Print zombies' initial (x,y) coordinates.
			Random prng = new Random (seed);
			for (int i = 0; i < N; ++ i)
				{
				double x_i = prng.nextDouble()*W;
				double y_i = prng.nextDouble()*W;
				System.out.printf ("0\t%d\t%g\t%g%n", i, x_i, y_i);
				System.out.flush();
				}

			// Do time steps.
			int t = 0;
			ZombieTuple template = new ZombieTuple();
			ZombieTuple[] zt = new ZombieTuple [K];
			for (;;)
				{
				// Advance to next time step.
				++ t;

				// Receive and print new zombie positions from the workers.
				double totalDelta = 0.0;
				template.step = t;
				for (int i = 0; i < K; ++ i)
					{
					template.rank = i;
					zt[i] = takeTuple (template);
					totalDelta += zt[i].delta;
					}
				if (snap > 0 && t % snap == 0)
					snapshot (t, zt);

				// Stop when position delta is less than convergence threshold
				// or when the specified number of time steps have occurred.
				if ((steps == 0 && totalDelta < eps) ||
					(steps != 0 && t == steps))
						break;
				}

			// Print zombies' final positions.
			if (snap == 0 || t % snap != 0)
				snapshot (t, zt);
			}
		}

	/**
	 * Print a snapshot of the zombies' positions.
	 */
	private static void snapshot
		(int t,
		 ZombieTuple[] zt)
		{
		for (int i = 0; i < zt.length; ++ i)
			{
			ZombieTuple zt_i = zt[i];
			for (int j = 0; j < zt_i.x.length; ++ j)
				System.out.printf ("%d\t%d\t%g\t%g%n",
					t, zt_i.lb + j, zt_i.x[j], zt_i.y[j]);
			System.out.flush();
			}
		}

	}
