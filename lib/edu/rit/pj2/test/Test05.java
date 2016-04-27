//******************************************************************************
//
// File:    Test05.java
// Package: edu.rit.pj2.test
// Unit:    Class edu.rit.pj2.test.Test05
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

package edu.rit.pj2.test;

import edu.rit.pj2.LongLoop;
import edu.rit.pj2.Task;
import edu.rit.pj2.Vbl;
import edu.rit.pj2.vbl.LongVbl;

/**
 * Class Test05 is a unit test main program for the automatic reduction
 * capability of a parallel for loop.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.test.Test05 <I>lb</I> <I>ub</I>
 * <I>reps</I></TT>
 * <P>
 * A thread team executes a parallel for loop with the given lower and upper
 * bounds. The program computes and prints the sum of the loop indexes. This is
 * repeated <TT><I>reps</I></TT> times.
 *
 * @author  Alan Kaminsky
 * @version 24-Mar-2014
 */
public class Test05
	extends Task
	{

	// Global shared long integer variable with sum reduction.
	LongVbl sum = new LongVbl()
		{
		public void reduce (Vbl vbl)
			{
			this.item += ((LongVbl)vbl).item;
			}
		};

	/**
	 * Perform this task's computation.
	 *
	 * @param  args  Array of zero or more command line argument strings.
	 *
	 * @exception  Exception
	 *     The <TT>main()</TT> method can throw any exception.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		if (args.length != 3) usage();
		long lb = Long.parseLong (args[0]);
		long ub = Long.parseLong (args[1]);
		int reps = Integer.parseInt (args[2]);

		if (threads() == THREADS_EQUALS_CORES)
			System.out.printf ("threads = THREADS_EQUALS_CORES%n");
		else
			System.out.printf ("threads = %d%n", threads());
		System.out.printf ("actual threads = %d%n", actualThreads());
		System.out.printf ("schedule = %s%n", schedule());
		if (chunk() == STANDARD_CHUNK)
			System.out.printf ("chunk = STANDARD_CHUNK%n");
		else
			System.out.printf ("chunk = %d%n", chunk());

		for (int i = 1; i <= reps; ++ i)
			{
			System.out.printf ("******** Repetition %d ********%n", i);
			sum.item = 0L;
			parallelFor (lb, ub) .exec (new LongLoop()
				{
				LongVbl thrSum;
				public void start()
					{
					thrSum = threadLocal (sum);
					}
				public void run (long i)
					{
					thrSum.item += i;
					}
				});
			System.out.printf ("Sum = %d%n", sum.item);
			}
		}

// Hidden operations.

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.test.Test05 <lb> <ub> <reps>");
		System.exit (1);
		}

	}
