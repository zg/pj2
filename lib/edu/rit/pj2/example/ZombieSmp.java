//******************************************************************************
//
// File:    ZombieSmp.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.ZombieSmp
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

import edu.rit.pj2.Loop;
import edu.rit.pj2.Task;
import edu.rit.pj2.vbl.DoubleVbl;
import edu.rit.util.Random;
import static java.lang.Math.*;

/**
 * Class ZombieSmp is an SMP parallel program to compute the motion of a group
 * of zombies. This is a so-called "<I>N</I>-bodies" problem.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.example.ZombieSmp <I>seed</I> <I>N</I>
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
 * @version 08-Jul-2014
 */
public class ZombieSmp
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
	double[] x;
	double[] y;

	// Next body positions.
	double[] xnext;
	double[] ynext;

	// For detecting convergence.
	DoubleVbl delta = new DoubleVbl.Sum();

	/**
	 * Task main program.
	 */
	public void main
		(String[] args)
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

		// Set up N bodies' initial (x,y) coordinates at random in a WxW square
		// region.
		x = new double [N];
		y = new double [N];
		xnext = new double [N];
		ynext = new double [N];
		Random prng = new Random (seed);
		for (int i = 0; i < N; ++ i)
			{
			x[i] = prng.nextDouble()*W;
			y[i] = prng.nextDouble()*W;
			}

		// Snapshot all bodies' initial positions.
		int t = 0;
		snapshot (t);

		// Do time steps.
		for (;;)
			{
			delta.item = 0.0;

			// Do each body i.
			parallelFor (0, N - 1) .exec (new Loop()
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
					xnext[i] = x[i] + dx;
					ynext[i] = y[i] + dy;

					// Accumulate position delta.
					thrDelta.item += abs(dx) + abs(dy);
					}
				});

			// Advance to next time step.
			++ t;

			// Update positions.
			double[] tmp;
			tmp = x; x = xnext; xnext = tmp;
			tmp = y; y = ynext; ynext = tmp;

			// Stop when position delta is less than convergence threshold or
			// when the specified number of time steps have occurred.
			if ((steps == 0 && delta.item < eps) || (steps != 0 && t == steps))
				break;

			// Snapshot all bodies' positions every <snap> time steps.
			if (snap > 0 && (t % snap) == 0)
				snapshot (t);
			}

		// Snapshot all bodies' final positions.
		if (snap == 0 || t % snap != 0)
			snapshot (t);
		}

	// Snapshot all bodies' positions.
	private void snapshot
		(int t)
		{
		for (int i = 0; i < N; ++ i)
			System.out.printf ("%d\t%d\t%g\t%g%n", t, i, x[i], y[i]);
		System.out.flush();
		}

	// Print a usage message and exit.
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.ZombieSmp <seed> <N> <W> <G> <L> <dt> <eps> <steps> <snap>");
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

	}
