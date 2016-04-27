//******************************************************************************
//
// File:    Test16.java
// Package: edu.rit.pj2.test
// Unit:    Class edu.rit.pj2.test.Test16
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
import edu.rit.pj2.LongLoop;
import edu.rit.pj2.Task;
import edu.rit.util.LongAction;
import edu.rit.util.LongList;

/**
 * Class Test16 is a unit test main program for the master-worker cluster
 * parallel for loop capability in methods {@link Job#masterFor(long,long,Class)
 * Job.masterFor(long,long,Class)} and {@link Task#workerFor()
 * Task.workerFor()}.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.test.Test16 <I>lb</I> <I>ub</I></TT>
 * <P>
 * A group of worker tasks executes a parallel for loop with the given lower and
 * upper bounds. The program records and prints which loop iterations were
 * executed by which threads in which worker tasks.
 *
 * @author  Alan Kaminsky
 * @version 30-May-2014
 */
public class Test16
	extends Job
	{

// Exported operations.

	/**
	 * Job main program.
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
		if (args.length != 2) usage();
		long lb = Long.parseLong (args[0]);
		long ub = Long.parseLong (args[1]);

		System.out.printf ("Job properties:%n");
		System.out.printf ("\tworkers = %d%n", workers());
		System.out.printf ("\tmasterSchedule = %s%n", masterSchedule());
		if (masterChunk() == STANDARD_CHUNK)
			System.out.printf ("\tmasterChunk = STANDARD_CHUNK%n");
		else
			System.out.printf ("\tmasterChunk = %d%n", masterChunk());
		if (threads() == THREADS_EQUALS_CORES)
			System.out.printf ("\tthreads = THREADS_EQUALS_CORES%n");
		else
			System.out.printf ("\tthreads = %d%n", threads());
		System.out.printf ("\tschedule = %s%n", schedule());
		if (chunk() == STANDARD_CHUNK)
			System.out.printf ("\tchunk = STANDARD_CHUNK%n");
		else
			System.out.printf ("\tchunk = %d%n", chunk());
		if (nodeName() == ANY_NODE_NAME)
			System.out.printf ("\tnodeName = ANY_NODE_NAME%n");
		else
			System.out.printf ("\tnodeName = \"%s\"%n", nodeName());
		if (cores() == ALL_CORES)
			System.out.printf ("\tcores = ALL_CORES%n");
		else
			System.out.printf ("\tcores = %d%n", cores());
		if (gpus() == ALL_GPUS)
			System.out.printf ("\tgpus = ALL_GPUS%n");
		else
			System.out.printf ("\tgpus = %d%n", gpus());

		masterFor (lb, ub, WorkerTask.class);
		}

// Worker task class.

	private static class WorkerTask
		extends Task
		{
		public void main
			(String[] args)
			{
			workerFor() .exec (new LongLoop()
				{
				LongList iterations;

				public void start()
					{
					iterations = new LongList();
					}

				public void run (long i)
					{
					iterations.addLast (i);
					try { Thread.sleep (10); }
					catch (InterruptedException exc) {}
					}

				public void finish() throws Exception
					{
					System.out.printf ("Task %d thread %d:",
						taskRank(), rank());
					iterations.forEachItemDo (new LongAction()
						{
						public void run (long i)
							{
							System.out.printf (" %d", i);
							}
						});
					System.out.println();
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
		System.err.println ("Usage: java pj2 edu.rit.pj2.test.Test16 <lb> <ub>");
		System.exit (1);
		}

	}
