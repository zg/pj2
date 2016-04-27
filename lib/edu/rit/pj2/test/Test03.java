//******************************************************************************
//
// File:    Test03.java
// Package: edu.rit.pj2.test
// Unit:    Class edu.rit.pj2.test.Test03
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

import edu.rit.pj2.Loop;
import edu.rit.pj2.Task;
import java.util.ArrayList;

/**
 * Class Test03 is a unit test main program for the {@link
 * edu.rit.pj2.Task#parallelFor(int,int) parallelFor(int,int)} method of class
 * {@linkplain edu.rit.pj2.Task Task}.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.test.Test03 <I>lb</I> <I>ub</I>
 * <I>reps</I></TT>
 * <P>
 * A thread team executes a parallel for loop with the given lower and upper
 * bounds. The program records and prints which loop iterations were executed by
 * which threads. This is repeated <TT><I>reps</I></TT> times.
 *
 * @author  Alan Kaminsky
 * @version 24-Mar-2014
 */
public class Test03
	extends Task
	{

// Exported operations.

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
		int lb = Integer.parseInt (args[0]);
		int ub = Integer.parseInt (args[1]);
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
			parallelFor (lb, ub) .exec (new Loop()
				{
				ArrayList<Integer> iterations;

				public void start()
					{
					iterations = new ArrayList<Integer>();
					}

				public void run (int i)
					{
					iterations.add (i);
					try { Thread.sleep (10); }
					catch (InterruptedException exc) {}
					}

				public void finish()
					{
					synchronized (System.out)
						{
						System.out.printf ("Thread %d:", rank());
						for (int i : iterations)
							System.out.printf (" %d", i);
						System.out.println();
						}
					}
				});
			}
		}

// Hidden operations.

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.test.Test03 <lb> <ub> <reps>");
		System.exit (1);
		}

	}
