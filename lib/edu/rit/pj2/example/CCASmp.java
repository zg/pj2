//******************************************************************************
//
// File:    CCASmp.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.CCASmp
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
import edu.rit.pj2.Loop;
import edu.rit.pj2.Section;
import edu.rit.pj2.Task;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;

/**
 * Class CCASmp is an SMP parallel program that calculates the evolution of a
 * continuous cellular automaton and stores the result in a grayscale PNG image
 * file.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.example.CCASmp <I>N</I> <I>S</I> <I>A</I>
 * <I>B</I> <I>C</I> <I>D</I> <I>file</I></TT>
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
public class CCASmp
	extends Task
	{

// Global variables.

	// Command line arguments.
	int N;
	int S;
	CCACell A, B, C, D;
	File file;

	// Old and new cell arrays.
	CCACell[] current;
	CCACell[] next;

	// For writing PNG file.
	GrayPngWriter writer;
	GrayImageQueue imageQueue;
	GrayArray imageRow;

// Main program.

	/**
	 * Main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Parse command line arguments.
		if (args.length != 7) usage();
		N = Integer.parseInt (args[0]);
		S = Integer.parseInt (args[1]);
		A = new CCACell() .assign (args[2]);
		B = new CCACell() .assign (args[3]);
		C = new CCACell() .assign (args[4]);
		D = new CCACell() .assign (args[5]);
		file = new File (args[6]);

		// Allocate storage for old and new cell arrays. Initialize all cells to
		// 0, except center cell to 1.
		current = new CCACell [N];
		next = new CCACell [N];
		parallelFor (0, N - 1) .exec (new Loop()
			{
			public void run (int i)
				{
				current[i] = new CCACell();
				next[i] = new CCACell();
				}
			});
		current[N/2].assign (1);

		// Set up for writing PNG file.
		writer = new GrayPngWriter (S + 1, N,
			new BufferedOutputStream (new FileOutputStream (file)));
		file.setReadable (true, false);
		file.setWritable (true, false);
		imageQueue = writer.getImageQueue();
		imageRow = new GrayArray (N);

		// Overlapped computation and I/O.
		parallelDo (new Section()
			{
			public void run() throws Exception
				{
				// Computation section. Do S time steps.
				CCACell[] tmp;
				for (int s = 0; s < S; ++ s)
					{
					// Calculate next state and write current state of each
					// cell.
					parallelFor (0, N - 1) .exec (new Loop()
						{
						CCACell T;
						public void start()
							{
							T = new CCACell();
							}
						public void run (int i)
							{
							next[i] .assign (T.assign (A)
								.multiply (current[(i-1+N)%N]));
							next[i] .add (T.assign (B)
								.multiply (current[i]));
							next[i] .add (T.assign (C)
								.multiply (current[(i+1)%N]));
							next[i] .add (D) .fracPart();
							imageRow.gray (i, current[i].floatValue(), false);
							}
						});
					imageQueue.put (s, imageRow);

					// Advance one time step -- swap old and new cell arrays.
					tmp = current; current = next; next = tmp;
					}

				// Write final CA state to image file.
				parallelFor (0, N - 1) .exec (new Loop()
					{
					public void run (int i)
						{
						imageRow.gray (i, current[i].floatValue(), false);
						}
					});
				imageQueue.put (S, imageRow);
				}
			},

		new Section()
			{
			public void run() throws Exception
				{
				// I/O section. Write image file.
				writer.write();
				}
			});
		}

// Hidden operations.

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.CCASmp <N> <S> <A> <B> <C> <D> <file>");
		System.err.println ("<N> = Number of cells (>= 1)");
		System.err.println ("<S> = Number of time steps (>= 1)");
		System.err.println ("<A> = Left cell multiplicand (rational number)");
		System.err.println ("<B> = Center cell multiplicand (rational number)");
		System.err.println ("<C> = Right cell multiplicand (rational number)");
		System.err.println ("<D> = Addend (rational number)");
		System.err.println ("<file> = Output PNG image file name");
		throw new IllegalArgumentException();
		}

	}
