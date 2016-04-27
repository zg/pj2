//******************************************************************************
//
// File:    CCAClu.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.CCAClu
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

import edu.rit.image.GrayArray;
import edu.rit.image.GrayImageQueue;
import edu.rit.image.GrayPngWriter;
import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.pj2.Chunk;
import edu.rit.pj2.Job;
import edu.rit.pj2.Loop;
import edu.rit.pj2.Section;
import edu.rit.pj2.Task;
import edu.rit.pj2.Tuple;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * Class CCAClu is a hybrid SMP cluster parallel program that calculates the
 * evolution of a continuous cellular automaton and stores the result in a
 * grayscale PNG image file.
 * <P>
 * Usage: <TT>java pj2 [workers=<I>K</I>] edu.rit.pj2.example.CCAClu <I>N</I>
 * <I>S</I> <I>A</I> <I>B</I> <I>C</I> <I>D</I> <I>file</I></TT>
 * <BR><TT><I>K</I></TT> = Number of worker tasks (default 1)
 * <BR><TT><I>N</I></TT> = Number of cells (&ge; 1)
 * <BR><TT><I>S</I></TT> = Number of time steps (&ge; 1)
 * <BR><TT><I>A</I></TT> = Left cell multiplicand (rational number)
 * <BR><TT><I>B</I></TT> = Center cell multiplicand (rational number)
 * <BR><TT><I>C</I></TT> = Right cell multiplicand (rational number)
 * <BR><TT><I>D</I></TT> = Addend (rational number)
 * <BR><TT><I>file</I></TT> = Output PNG image file name
 * <P>
 * The cellular automaton (CA) consists of an array of <I>N</I> cells (class
 * {@linkplain CCACell CCACell}). Each cell's value is in the range 0.0 to 1.0.
 * A cell's next value is computed by the formula
 * <P>
 * <CENTER>
 * <I>y</I><SUB>next</SUB> = frac(<I>Ax</I> + <I>By</I> + <I>Cz</I> + <I>D</I>)
 * </CENTER>
 * <P>
 * where <I>A, B, C,</I> and <I>D</I> are parameters from the command line,
 * <I>x</I> is the value of the cell's left neighbor, <I>y</I> is the value of
 * the cell itself, and <I>z</I> is the value of the cell's right neighbor. The
 * cell array has wraparound boundaries. The CA's initial state is all cells
 * 0.0, except the middle cell is 1.0. The program evolves the CA's initial
 * state for <I>S</I> time steps and generates a grayscale image with <I>S</I>+1
 * rows and <I>N</I> columns. The first row of the image corresponds to the CA's
 * initial state, and each subsequent row corresponds to the CA's state after
 * each time step. In the image, each pixel's gray value is proportional to the
 * corresponding cell's value, with 0 being white and 1 being black. The image
 * is stored in a PNG file.
 *
 * @author  Alan Kaminsky
 * @version 04-Jul-2014
 */
public class CCAClu
	extends Job
	{

	/**
	 * Job main program.
	 */
	public void main
		(String[] args)
		{
		// Parse command line arguments.
		if (args.length != 7) usage();
		int N = Integer.parseInt (args[0]);
		int S = Integer.parseInt (args[1]);
		CCACell A = new CCACell() .assign (args[2]);
		CCACell B = new CCACell() .assign (args[3]);
		CCACell C = new CCACell() .assign (args[4]);
		CCACell D = new CCACell() .assign (args[5]);
		File file = new File (args[6]);

		// Set up task group with K worker tasks.
		int K = workers();
		if (K == DEFAULT_WORKERS) K = 1;
		rule() .task (K, WorkerTask.class) .args (args);

		// Set up PNG file writing task.
		rule() .task (OutputTask.class) .args (""+K, ""+N, ""+S, ""+file)
			.runInJobProcess();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 [workers=<K>] edu.rit.pj2.example.CCAClu <N> <S> <A> <B> <C> <D> <file>");
		System.err.println ("<K> = Number of worker tasks (default 1)");
		System.err.println ("<N> = Number of cells (>= 1)");
		System.err.println ("<S> = Number of time steps (>= 1)");
		System.err.println ("<A> = Left cell multiplicand (rational number)");
		System.err.println ("<B> = Center cell multiplicand (rational number)");
		System.err.println ("<C> = Right cell multiplicand (rational number)");
		System.err.println ("<D> = Addend (rational number)");
		System.err.println ("<file> = Output PNG image file name");
		throw new IllegalArgumentException();
		}

	/**
	 * Tuple for sending a cell from one worker task to another.
	 */
	private static class CellTuple
		extends Tuple
		{
		public int step;      // Time step
		public int index;     // Cell index
		public CCACell cell;  // Cell

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
			 CCACell cell)
			{
			this.step = step;
			this.index = index;
			this.cell = cell;
			}

		public boolean matchContent (Tuple target)
			{
			CellTuple t = (CellTuple) target;
			return this.step == t.step && this.index == t.index;
			}

		public void writeOut (OutStream out) throws IOException
			{
			out.writeUnsignedInt (step);
			out.writeUnsignedInt (index);
			out.writeObject (cell);
			}

		public void readIn (InStream in) throws IOException
			{
			step = in.readUnsignedInt();
			index = in.readUnsignedInt();
			cell = (CCACell) in.readObject();
			}
		}

	/**
	 * Tuple for sending results from worker tasks to output task.
	 */
	private static class OutputTuple
		extends Tuple
		{
		public int step;            // Time step
		public int lb;              // Lower bound cell index
		public GrayArray imageRow;  // Portion of image row

		public OutputTuple()
			{
			}

		public OutputTuple
			(int step,
			 int lb,
			 GrayArray imageRow)
			{
			this.step = step;
			this.lb = lb;
			this.imageRow = imageRow;
			}

		public boolean matchContent (Tuple target)
			{
			OutputTuple t = (OutputTuple) target;
			return this.step == t.step;
			}

		public void writeOut (OutStream out) throws IOException
			{
			out.writeUnsignedInt (step);
			out.writeUnsignedInt (lb);
			out.writeObject (imageRow);
			}

		public void readIn (InStream in) throws IOException
			{
			step = in.readUnsignedInt();
			lb = in.readUnsignedInt();
			imageRow = (GrayArray) in.readObject();
			}
		}

	/**
	 * Worker task class.
	 */
	private static class WorkerTask
		extends Task
		{
		// Command line arguments.
		int N;
		int S;
		CCACell A, B, C, D;

		// Index range for this worker's chunk of cells, and boundary indexes.
		int lb;
		int ub;
		int left;
		int right;

		// Old and new cell arrays.
		CCACell[] current;
		CCACell[] next;

		// For writing PNG file.
		GrayArray imageRow;

		// Task main program.
		public void main
			(String[] args)
			throws Exception
			{
			// Parse command line arguments.
			N = Integer.parseInt (args[0]);
			S = Integer.parseInt (args[1]);
			A = new CCACell() .assign (args[2]);
			B = new CCACell() .assign (args[3]);
			C = new CCACell() .assign (args[4]);
			D = new CCACell() .assign (args[5]);

			// Partition the N cells among the worker tasks.
			Chunk chunk = Chunk.partition (0, N - 1, groupSize(), taskRank());
			lb = chunk.lb();
			ub = chunk.ub();
			left = (lb - 1 + N) % N;
			right = (ub + 1) % N;

			// Allocate storage for old and new cell arrays. Initialize all
			// cells to 0, except center cell to 1.
			current = new CCACell [N];
			next = new CCACell [N];
			for (int i = lb; i <= ub; ++ i)
				{
				current[i] = new CCACell();
				next[i] = new CCACell();
				}
			current[left] = new CCACell();
			current[right] = new CCACell();
			next[left] = new CCACell();
			next[right] = new CCACell();
			if (current[N/2] != null)
				current[N/2].assign (1);

			// Set up for writing PNG file.
			imageRow = new GrayArray ((int) chunk.length());

			// Do S time steps.
			CCACell[] tmp;
			for (int s = 0; s < S; ++ s)
				{
				// Send and receive boundary cells in current state.
				putTuple (new CellTuple (s, lb, current[lb]));
				putTuple (new CellTuple (s, ub, current[ub]));
				current[left] =
					((CellTuple)(takeTuple (new CellTuple (s, left))))
						.cell;
				current[right] =
					((CellTuple)(takeTuple (new CellTuple (s, right))))
						.cell;

				// Calculate next state and write current state of each cell.
				parallelFor (lb, ub) .exec (new Loop()
					{
					CCACell T;
					public void start()
						{
						T = new CCACell();
						}
					public void run (int i)
						{
						next[i] .assign
							(T.assign (A) .multiply (current[(i-1+N)%N]));
						next[i] .add
							(T.assign (B) .multiply (current[i]));
						next[i] .add
							(T.assign (C) .multiply (current[(i+1)%N]));
						next[i] .add (D) .fracPart();
						imageRow.gray (i - lb, current[i].floatValue(), false);
						}
					});
				putTuple (new OutputTuple (s, lb, imageRow));

				// Advance one time step -- swap old and new cell arrays.
				tmp = current; current = next; next = tmp;
				}

			// Write final CA state to image file.
			for (int i = lb; i <= ub; ++ i)
				imageRow.gray (i - lb, current[i].floatValue(), false);
			putTuple (new OutputTuple (S, lb, imageRow));
			}
		}

	/**
	 * PNG file writing task class.
	 */
	private static class OutputTask
		extends Task
		{
		// Command line arguments.
		int K;
		int N;
		int S;
		File file;

		// For writing PNG file.
		GrayPngWriter writer;
		GrayImageQueue imageQueue;
		GrayArray imageRow;

		// Task main program.
		public void main
			(String[] args)
			throws Exception
			{
			// Parse command line arguments.
			K = Integer.parseInt (args[0]);
			N = Integer.parseInt (args[1]);
			S = Integer.parseInt (args[2]);
			file = new File (args[3]);

			// Set up for writing PNG file.
			writer = new GrayPngWriter (S + 1, N,
				new BufferedOutputStream (new FileOutputStream (file)));
			file.setReadable (true, false);
			file.setWritable (true, false);
			imageQueue = writer.getImageQueue();
			imageRow = new GrayArray (N);

			// Overlapped result gathering section and file writing section.
			parallelDo (new Section()
				{
				// Result gathering section.
				public void run() throws Exception
					{
					OutputTuple template = new OutputTuple();
					OutputTuple tuple;
					for (int s = 0; s <= S; ++ s)
						{
						template.step = s;
						for (int i = 0; i < K; ++ i)
							{
							tuple = takeTuple (template);
							imageRow.copy (tuple.imageRow, 0, tuple.lb,
								tuple.imageRow.length());
							}
						imageQueue.put (s, imageRow);
						}
					}
				},

			new Section()
				{
				// File writing section.
				public void run() throws Exception
					{
					writer.write();
					}
				});
			}
		}

	}
