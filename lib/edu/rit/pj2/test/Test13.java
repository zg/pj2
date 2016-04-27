//******************************************************************************
//
// File:    Test13.java
// Package: edu.rit.pj2.test
// Unit:    Class edu.rit.pj2.test.Test13
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

import edu.rit.pj2.Job;
import edu.rit.pj2.Loop;
import edu.rit.pj2.Task;

/**
 * Class Test13 is a unit test main program for classes {@linkplain
 * edu.rit.pj2.Job Job} and {@linkplain edu.rit.pj2.Task Task}. The job runs a
 * number of tasks <I>N</I> specified on the command line. Each task runs
 * <I>NT</I> threads (default: one thread on each core of its node). Each thread
 * prints a unique "Hello, world" message.
 * <P>
 * Usage: <TT>java pj2 [threads=<I>NT</I>] edu.rit.pj2.test.Test13 <I>N</I></TT>
 *
 * @author  Alan Kaminsky
 * @version 30-May-2014
 */
public class Test13
	extends Job
	{

// Exported operations.

	/**
	 * Perform this job's computation.
	 *
	 * @param  args  Array of zero or more command line argument strings.
	 *
	 * @exception  Exception
	 *     The <TT>main()</TT> method can throw any exception.
	 */
	public void main
		(String[] args)
		{
		if (args.length != 1) usage();
		int N = Integer.parseInt (args[0]);
		for (int task = 0; task < N; ++ task)
			rule() .task (HelloTask.class) .args (""+task);
		}

// Hidden operations.

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 [threads=<NT>] edu.rit.pj2.test.Test13 <N>");
		System.exit (1);
		}

// Hidden helper classes.

	/**
	 * Class HelloTask prints "Hello, world" messages as part of a {@linkplain
	 * Test13 Test13} job.
	 *
	 * @author  Alan Kaminsky
	 * @version 30-May-2014
	 */
	private static class HelloTask
		extends Task
		{
		public void main
			(String[] args)
			{
			final int task = Integer.parseInt (args[0]);
			parallelFor (0, actualThreads() - 1) .exec (new Loop()
				{
				public void run (int rank) throws Exception
					{
					System.out.printf
						("Hello, world from thread %d of task %d%n",
						 rank, task);
					}
				});
			}
		}

	}
