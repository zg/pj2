//******************************************************************************
//
// File:    PiClu.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.PiClu
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

import edu.rit.pj2.Job;
import edu.rit.pj2.LongLoop;
import edu.rit.pj2.Task;
import edu.rit.pj2.vbl.LongVbl;
import edu.rit.util.Random;

/**
 * Class PiClu is a cluster parallel program that calculates an approximate
 * value for &pi; using a Monte Carlo technique. The program generates a number
 * of random points in the unit square (0,0) to (1,1) and counts how many of
 * them lie within a circle of radius 1 centered at the origin. The fraction of
 * the points within the circle is approximately &pi;/4.
 * <P>
 * Usage: <TT>java pj2 [workers=<I>K</I>] edu.rit.pj2.example.PiClu <I>seed</I>
 * <I>N</I></TT>
 * <BR><TT><I>K</I></TT> = Number of worker tasks (default: 1)
 * <BR><TT><I>seed</I></TT> = Random seed
 * <BR><TT><I>N</I></TT> = Number of random points
 * <P>
 * The program uses class {@linkplain edu.rit.util.Random} for its pseudorandom
 * number generator. The computation is performed in <I>K</I> multithreaded
 * {@linkplain edu.rit.pj2.Task Task}s. Each task generates random points using
 * a different seed derived from the given <TT><I>seed</I></TT>. The tasks are
 * performed in parallel as a {@linkplain edu.rit.pj2.Job Job}.
 *
 * @author  Alan Kaminsky
 * @version 24-Dec-2015
 */
public class PiClu
	extends Job
	{

// Job main program.

	/**
	 * Job main program.
	 */
	public void main
		(String[] args)
		{
		// Parse command line arguments.
		if (args.length != 2) usage();
		long seed = Long.parseLong (args[0]);
		long N = Long.parseLong (args[1]);

		// Set up a task group of K worker tasks.
		masterFor (0, N - 1, WorkerTask.class) .args (""+seed);

		// Set up reduction task.
		rule() .atFinish() .task (ReduceTask.class) .args (""+N)
			.runInJobProcess();
		}

// Hidden operations.

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 [workers=<K>] edu.rit.pj2.example.PiClu <seed> <N>");
		System.err.println ("<K> = Number of worker tasks (default: 1)");
		System.err.println ("<seed> = Random seed");
		System.err.println ("<N> = Number of random points");
		throw new IllegalArgumentException();
		}

// Task subclasses.

	/**
	 * Class PiClu.WorkerTask performs part of the computation for the
	 * {@linkplain PiClu PiClu} program.
	 *
	 * @author  Alan Kaminsky
	 * @version 01-Jan-2014
	 */
	private static class WorkerTask
		extends Task
		{
		// Command line arguments.
		long seed;

		// Number of points within the unit circle.
		LongVbl count;

		/**
		 * Main program.
		 */
		public void main
			(String[] args)
			throws Exception
			{
			// Parse command line arguments.
			seed = Long.parseLong (args[0]);

			// Generate n random points in the unit square, count how many are
			// in the unit circle.
			count = new LongVbl.Sum (0);
			workerFor() .exec (new LongLoop()
				{
				Random prng;
				LongVbl thrCount;
				public void start()
					{
					prng = new Random (seed + 1000*taskRank() + rank());
					thrCount = threadLocal (count);
					}
				public void run (long i)
					{
					double x = prng.nextDouble();
					double y = prng.nextDouble();
					if (x*x + y*y <= 1.0) ++ thrCount.item;
					}
				});

			// Report results.
			putTuple (count);
			}
		}

	/**
	 * Class PiClu.ReduceTask combines the worker tasks' results and prints the
	 * overall result for the {@linkplain PiClu PiClu} program.
	 *
	 * @author  Alan Kaminsky
	 * @version 01-Jan-2014
	 */
	private static class ReduceTask
		extends Task
		{
		/**
		 * Reduce task main program.
		 */
		public void main
			(String[] args)
			throws Exception
			{
			long N = Long.parseLong (args[0]);
			LongVbl count = new LongVbl.Sum (0L);
			LongVbl template = new LongVbl();
			LongVbl taskCount;
			while ((taskCount = tryToTakeTuple (template)) != null)
				count.reduce (taskCount);
			System.out.printf ("pi = 4*%d/%d = %.9f%n",
				count.item, N, 4.0*count.item/N);
			}
		}

	}
