//******************************************************************************
//
// File:    BCAClu.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.BCAClu
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
import edu.rit.pj2.Rule;
import edu.rit.pj2.Task;
import edu.rit.pj2.TaskSpec;
import edu.rit.pj2.Tuple;
import edu.rit.pj2.vbl.IntVbl;
import edu.rit.util.Hex;
import java.io.IOException;

/**
 * Class BCAClu is a cluster parallel program that computes the evolution of a
 * binary cellular automaton (CA). The CA's update rule depends on a 5-cell
 * neighborhood, the cell itself plus its two left neighbors and its two right
 * neighbors. The update rule is specified as a 32-bit hexadecimal number. The
 * CA consists of <I>C</I> cells. The evolution is computed for <I>S</I> steps.
 * The initial state is cell 0's value = 1 and all other cells' values = 0. The
 * program prints the population count (popcount; number of cells with the value
 * 1) for each step.
 * <P>
 * Usage: <TT>java pj2 [workers=<I>K</I>] edu.rit.pj2.example.BCAClu
 * <I>updaterule</I> <I>C</I> <I>S</I></TT>
 * <BR><TT><I>K</I></TT> = Number of worker tasks (default: 1)
 * <BR><TT><I>updaterule</I></TT> = Update rule (8 hex digits)
 * <BR><TT><I>C</I></TT> = Number of cells
 * <BR><TT><I>S</I></TT> = Number of steps
 * <P>
 * The computation is performed in parallel in multiple worker tasks running on
 * separate nodes of the cluster. Each task in turn does its computation in
 * parallel in multiple threads running on separate cores of the node. The cells
 * are partitioned equally among the tasks; the cells in a task are partitioned
 * among the threads using the <TT>schedule</TT> and <TT>chunk</TT> properties.
 *
 * @author  Alan Kaminsky
 * @version 01-Jan-2014
 */
public class BCAClu
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
		if (args.length != 3) usage();
		int updaterule = Hex.toInt (args[0]);
		int C = Integer.parseInt (args[1]);
		int S = Integer.parseInt (args[2]);
		int K = workers();

		// Set up a task group of K worker tasks.
		rule() .task (K, WorkerTask.class)
			.args (Hex.toString (updaterule), ""+C, ""+S);

		// Set up output task.
		rule() .task (OutputTask.class)
			.args (""+S, ""+K) .runInJobProcess();
		}

// Hidden operations.

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 [workers=<K>] edu.rit.pj2.example.BCAClu <rule> <C> <S>");
		System.err.println ("<K> = Number of worker tasks (default: 1)");
		System.err.println ("<rule> = Update rule (8 hex digits)");
		System.err.println ("<C> = Number of cells");
		System.err.println ("<S> = Number of steps");
		throw new IllegalArgumentException();
		}

// Tuple subclasses.

	/**
	 * Class BCAClu.CellTuple contains the value of the cells at <I>index</I>
	 * and <I>index</I>+1 at the given <I>step</I>.
	 *
	 * @author  Alan Kaminsky
	 * @version 05-Dec-2013
	 */
	private static class CellTuple
		extends Tuple
		{
		public int step;
		public int index;
		public byte value_0;
		public byte value_1;

		public CellTuple()
			{
			}

		public CellTuple
			(int step,
			 int index)
			{
			this.step = step;
			this.index = index;
			}

		public CellTuple
			(int step,
			 int index,
			 byte value_0,
			 byte value_1)
			{
			this.step = step;
			this.index = index;
			this.value_0 = value_0;
			this.value_1 = value_1;
			}

		public boolean matchContent
			(Tuple target)
			{
			return ((CellTuple)target).step == this.step &&
				((CellTuple)target).index == this.index;
			}

		public void writeOut
			(OutStream out)
			throws IOException
			{
			out.writeInt (step);
			out.writeInt (index);
			out.writeByte (value_0);
			out.writeByte (value_1);
			}

		public void readIn
			(InStream in)
			throws IOException
			{
			step = in.readInt();
			index = in.readInt();
			value_0 = in.readByte();
			value_1 = in.readByte();
			}
		}

	/**
	 * Class BCAClu.PopCountTuple contains the popcount of a group of cells at a
	 * certain step.
	 *
	 * @author  Alan Kaminsky
	 * @version 05-Dec-2013
	 */
	private static class PopCountTuple
		extends Tuple
		{
		public int step;
		public int popcount;

		public PopCountTuple()
			{
			}

		public PopCountTuple
			(int step)
			{
			this.step = step;
			}

		public PopCountTuple
			(int step,
			 int popcount)
			{
			this.step = step;
			this.popcount = popcount;
			}

		public boolean matchContent
			(Tuple target)
			{
			return ((PopCountTuple)target).step == this.step;
			}

		public void writeOut
			(OutStream out)
			throws IOException
			{
			out.writeInt (step);
			out.writeInt (popcount);
			}

		public void readIn
			(InStream in)
			throws IOException
			{
			step = in.readInt();
			popcount = in.readInt();
			}
		}

// Task subclasses.

	/**
	 * Class BCAClu.WorkerTask provides a task that computes the evolution of a
	 * certain slice of cells.
	 *
	 * @author  Alan Kaminsky
	 * @version 12-Dec-2013
	 */
	private static class WorkerTask
		extends Task
		{
		// Command line arguments.
		int rule;
		int C;
		int S;

		// Slices of current cell array and next cell array.
		int lb, ub, len, first, last;
		byte[] currentCell;
		byte[] nextCell;

		// Templates for receiving cell tuples.
		CellTuple leftTemplate;
		CellTuple rightTemplate;

		// Popcount.
		IntVbl popcount;

		/**
		 * Worker task main program.
		 */
		public void main
			(String[] args)
			throws Exception
			{
			// Parse command line arguments.
			rule = Hex.toInt (args[0]);
			C = Integer.parseInt (args[1]);
			S = Integer.parseInt (args[2]);

			// Set up slices of current cell array and next cell array.
			// - Indexes 0..1 are overlap from the task to the left.
			// - Indexes 2..last are this task's cells.
			// - Indexes last+1..last+2 are overlap from the task to the right.
			Chunk slice = Chunk.partition (0, C-1, groupSize(), taskRank());
			lb = slice.lb();
			ub = slice.ub();
			len = (int) slice.length();
			first = 2;
			last = len + 1;
			currentCell = new byte [len + 4];
			nextCell = new byte [len + 4];

			// Set up templates for receiving cell tuples.
			leftTemplate = new CellTuple (0, (lb - 2 + C) % C);
			rightTemplate = new CellTuple (0, (ub + 1) % C);

			// Initialize cell 0's value = 1.
			popcount = new IntVbl.Sum (0);
			if (slice.contains (0))
				{
				currentCell[2] = 1;
				popcount.item = 1;
				}

			// Do S steps.
			for (int step = 0; step < S; ++ step)
				{
				// Send current step's popcount to output task.
				putTuple (new PopCountTuple (step, popcount.item));

				// Send overlapping cells to left and right tasks.
				putTuple (new CellTuple (step, lb, currentCell[first],
					currentCell[first+1]));
				putTuple (new CellTuple (step, ub - 1, currentCell[last-1],
					currentCell[last]));

				// Receive overlapping cells from left and right tasks.
				leftTemplate.step = rightTemplate.step = step;
				CellTuple tuple = takeTuple (leftTemplate);
				currentCell[0] = tuple.value_0;
				currentCell[1] = tuple.value_1;
				tuple = takeTuple (rightTemplate);
				currentCell[last+1] = tuple.value_0;
				currentCell[last+2] = tuple.value_1;

				// Compute next state of each cell in parallel.
				popcount.item = 0;
				parallelFor (first, last) .exec (new Loop()
					{
					IntVbl thrPopcount;
					public void start()
						{
						thrPopcount = threadLocal (popcount);
						}
					public void run (int i)
						{
						int neighborhood = currentCell[i-2];
						neighborhood <<= 1;
						neighborhood |= currentCell[i-1];
						neighborhood <<= 1;
						neighborhood |= currentCell[i];
						neighborhood <<= 1;
						neighborhood |= currentCell[i+1];
						neighborhood <<= 1;
						neighborhood |= currentCell[i+2];
						nextCell[i] = (byte)((rule >> neighborhood) & 1);
						thrPopcount.item += nextCell[i];
						}
					});

				// Swap cell arrays for next step.
				byte[] tmp = currentCell;
				currentCell = nextCell;
				nextCell = tmp;
				}

			// Send final step's popcount to output task.
			putTuple (new PopCountTuple (S, popcount.item));
			}
		}

	/**
	 * Class BCAClu.OutputTask combines the worker tasks' popcounts and prints
	 * the overall popcount at each step.
	 *
	 * @author  Alan Kaminsky
	 * @version 12-Dec-2013
	 */
	private static class OutputTask
		extends Task
		{
		/**
		 * Output task main program.
		 */
		public void main
			(String[] args)
			throws Exception
			{
			int S = Integer.parseInt (args[0]);
			int K = Integer.parseInt (args[1]);
			PopCountTuple template = new PopCountTuple (0);
			PopCountTuple tuple;
			for (int step = 0; step <= S; ++ step)
				{
				int popcount = 0;
				template.step = step;
				for (int i = 0; i < K; ++ i)
					{
					tuple = takeTuple (template);
					popcount += tuple.popcount;
					}
				System.out.printf ("%d%n", popcount);
				}
			}
		}

	}
