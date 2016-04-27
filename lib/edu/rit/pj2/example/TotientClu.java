//******************************************************************************
//
// File:    TotientClu.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.TotientClu
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
import edu.rit.util.LongList;

/**
 * Class TotientClu is a cluster parallel program that computes the Euler
 * totient of a number. &Phi;(<I>n</I>) is the number of numbers in the range 1
 * through <I>n</I>&minus;1 that are relatively prime to <I>n</I>.
 * <P>
 * Usage: <TT>java pj2 [workers=<I>K</I>] edu.rit.pj2.example.TotientClu
 * <I>n</I></TT>
 * <BR><TT><I>K</I></TT> = Number of worker tasks (default: 1)
 * <BR><TT><I>n</I></TT> = Number whose totient to compute
 * <P>
 * The computation is performed in parallel in multiple worker tasks running on
 * separate nodes of the cluster. Each task in turn does its computation in
 * parallel in multiple threads running on separate cores of the node.
 * <P>
 * The default parallel for loop schedule may result in an unbalanced load.
 * Consider specifying the <TT>workerSchedule</TT>, <TT>workerChunk</TT>,
 * <TT>schedule</TT>, and/or <TT>chunk</TT> properties on the <TT>pj2</TT>
 * command line.
 *
 * @author  Alan Kaminsky
 * @version 01-Jan-2014
 */
public class TotientClu
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
		if (args.length != 1) usage();
		long n = Long.parseLong (args[0]);

		// Set up a task group of K worker tasks.
		masterFor (2, n - 1, WorkerTask.class) .args (""+n);

		// Set up reduction task.
		rule() .atFinish() .task (ReduceTask.class) .runInJobProcess();
		}

// Hidden operations.

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 [workers=<K>] edu.rit.pj2.example.TotientClu <n>");
		System.err.println ("<K> = Number of worker tasks (default: 1)");
		System.err.println ("<n> = Number whose totient to compute");
		throw new IllegalArgumentException();
		}

// Task subclasses.

	/**
	 * Class TotientClu.WorkerTask provides a task that computes chunks of
	 * iterations in the totient computation.
	 *
	 * @author  Alan Kaminsky
	 * @version 01-Jan-2014
	 */
	private static class WorkerTask
		extends Task
		{
		long n;
		LongVbl phi;
		LongList nFactors = new LongList();

		/**
		 * Worker task main program.
		 */
		public void main
			(String[] args)
			throws Exception
			{
			n = Long.parseLong (args[0]);

			phi = new LongVbl.Sum();
			factorize (n, nFactors);

			workerFor() .exec (new LongLoop()
				{
				LongList iFactors;
				LongVbl thrPhi;
				public void start()
					{
					iFactors = new LongList();
					thrPhi = threadLocal (phi);
					}
				public void run (long i)
					{
					if (relativelyPrime (factorize (i, iFactors), nFactors))
						++ thrPhi.item;
					}
				});

			// Report result.
			putTuple (phi);
			}

		/**
		 * Store a list of the prime factors of <I>x</I> in ascending order in
		 * the given list.
		 *
		 * @param  x     Number to factorize.
		 * @param  list  List of prime factors (output).
		 *
		 * @param  The given list is returned.
		 */
		private static LongList factorize
			(long x,
			 LongList list)
			{
			list.clear();
			long p = 2;
			long psqr = p*p;
			while (psqr <= x)
				{
				if (x % p == 0)
					{
					list.addLast (p);
					x /= p;
					}
				else
					{
					p = p == 2 ? 3 : p + 2;
					psqr = p*p;
					}
				}
			if (x != 1)
				list.addLast (x);
			return list;
			}

		/**
		 * Determine whether two numbers are relatively prime, given their lists
		 * of factors.
		 *
		 * @param  xFactors  List of factors of first number.
		 * @param  yFactors  List of factors of second number.
		 *
		 * @return  True if the numbers are relatively prime (have no common
		 *          factors), false otherwise.
		 */
		private static boolean relativelyPrime
			(LongList xFactors,
			 LongList yFactors)
			{
			int xSize = xFactors.size();
			int ySize = yFactors.size();
			int ix = 0;
			int iy = 0;
			long x, y;
			while (ix < xSize && iy < ySize)
				{
				x = xFactors.get (ix);
				y = yFactors.get (iy);
				if (x == y) return false;
				else if (x < y) ++ ix;
				else ++ iy;
				}
			return true;
			}
		}

	/**
	 * Class TotientClu.ReduceTask combines the worker tasks' results and prints
	 * the overall result.
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
			LongVbl phi = new LongVbl.Sum();
			LongVbl template = new LongVbl();
			LongVbl taskPhi;
			while ((taskPhi = tryToTakeTuple (template)) != null)
				phi.reduce (taskPhi);
			System.out.printf ("%d%n", phi.item + 1);
			}
		}

	}
