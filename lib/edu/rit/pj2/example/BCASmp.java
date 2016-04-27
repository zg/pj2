//******************************************************************************
//
// File:    BCASmp.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.BCASmp
//
// This Java source file is copyright (C) 2013 by Alan Kaminsky. All rights
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
import edu.rit.pj2.vbl.IntVbl;
import edu.rit.util.Hex;

/**
 * Class BCASmp is an SMP parallel program that computes the evolution of a
 * binary cellular automaton (CA). The CA's update rule depends on a 5-cell
 * neighborhood, the cell itself plus its two left neighbors and its two right
 * neighbors. The update rule is specified as a 32-bit hexadecimal number. The
 * CA consists of <I>C</I> cells. The evolution is computed for <I>S</I> steps.
 * The initial state is cell 0's value = 1 and all other cells' values = 0. The
 * program prints the population count (popcount; number of cells with the value
 * 1) for each step.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.example.BCASmp <I>rule</I> <I>C</I>
 * <I>S</I></TT>
 * <BR><TT><I>rule</I></TT> = Update rule (8 hex digits)
 * <BR><TT><I>C</I></TT> = Number of cells
 * <BR><TT><I>S</I></TT> = Number of steps
 * <P>
 * The computation is performed in parallel in multiple threads running on
 * separate cores. The cells are partitioned among the threads using the
 * <TT>schedule</TT> and <TT>chunk</TT> properties.
 *
 * @author  Alan Kaminsky
 * @version 28-Dec-2013
 */
public class BCASmp
	extends Task
	{
	// Command line arguments.
	int rule;
	int C;
	int S;

	// Current cell array and next cell array.
	byte[] currentCell;
	byte[] nextCell;

	// Popcount.
	IntVbl popcount;

	/**
	 * Task main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Parse command line arguments.
		if (args.length != 3) usage();
		rule = Hex.toInt (args[0]);
		C = Integer.parseInt (args[1]);
		S = Integer.parseInt (args[2]);

		// Set up current cell array and next cell array.
		currentCell = new byte [C];
		nextCell = new byte [C];

		// Initialize cell 0's value = 1.
		popcount = new IntVbl.Sum (1);
		currentCell[0] = 1;

		// Do S steps.
		for (int step = 0; step < S; ++ step)
			{
			// Print current step's popcount.
			System.out.printf ("%d%n", popcount.item);

			// Compute next state of each cell in parallel.
			popcount.item = 0;
			parallelFor (0, C - 1) .exec (new Loop()
				{
				IntVbl thrPopcount;
				public void start()
					{
					thrPopcount = threadLocal (popcount);
					}
				public void run (int i)
					{
					int neighborhood = currentCell[(i-2+C)%C];
					neighborhood <<= 1;
					neighborhood |= currentCell[(i-1+C)%C];
					neighborhood <<= 1;
					neighborhood |= currentCell[i];
					neighborhood <<= 1;
					neighborhood |= currentCell[(i+1)%C];
					neighborhood <<= 1;
					neighborhood |= currentCell[(i+2)%C];
					nextCell[i] = (byte)((rule >> neighborhood) & 1);
					thrPopcount.item += nextCell[i];
					}
				});

			// Swap cell arrays for next step.
			byte[] tmp = currentCell;
			currentCell = nextCell;
			nextCell = tmp;
			}

		// Print final step's popcount.
		System.out.printf ("%d%n", popcount.item);
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.BCASmp <rule> <C> <S>");
		System.err.println ("<rule> = Update rule (8 hex digits)");
		System.err.println ("<C> = Number of cells");
		System.err.println ("<S> = Number of steps");
		throw new IllegalArgumentException();
		}
	}
