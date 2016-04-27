//******************************************************************************
//
// File:    MinVerCovSeq.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.MinVerCovSeq
//
// This Java source file is copyright (C) 2015 by Alan Kaminsky. All rights
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

import edu.rit.pj2.Task;
import edu.rit.util.BitSet64;
import java.io.File;
import java.util.Scanner;

/**
 * Class MinVerCovSeq is a sequential program that finds a minimum vertex cover
 * of a graph via exhaustive search. The program reads the graph from a file.
 * The graph format is as follows: The first line consists of <I>V</I> and
 * <I>E</I>. Each subsequent line consists of two vertex numbers each in the
 * range 0 through <I>V</I>&minus;1, defining an edge between those vertices.
 * The program's running time is proportional to 2<SUP><I>V</I></SUP>.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.example.MinVerCovSeq <I>file</I></TT>
 * <BR><TT><I>file</I></TT> = Graph file
 *
 * @author  Alan Kaminsky
 * @version 12-Jan-2015
 */
public class MinVerCovSeq
	extends Task
	{

	// Number of vertices and edges.
	int V;
	int E;

	// The graph's adjacency matrix. adjacent[i] is the set of vertices adjacent
	// to vertex i.
	BitSet64[] adjacent;

	// Minimum vertex cover.
	BitSet64 minCover;

	/**
	 * Main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Parse command line arguments.
		if (args.length != 1) usage();
		File file = new File (args[0]);

		// Read input file, set up adjacency matrix.
		Scanner s = new Scanner (file);
		V = s.nextInt();
		E = s.nextInt();
		if (V < 1 || V > 63) usage ("V must be >= 1 and <= 63");
		adjacent = new BitSet64 [V];
		for (int i = 0; i < V; ++ i)
			adjacent[i] = new BitSet64();
		for (int i = 0; i < E; ++ i)
			{
			int a = s.nextInt();
			int b = s.nextInt();
			adjacent[a].add (b);
			adjacent[b].add (a);
			}
		s.close();

		// Check all candidate covers (sets of vertices).
		minCover = new BitSet64() .add (0, V);
		long full = minCover.bitmap();
		for (long elems = 0L; elems <= full; ++ elems)
			{
			BitSet64 candidate = new BitSet64 (elems);
			if (candidate.size() < minCover.size() &&
				isCover (candidate))
					minCover.copy (candidate);
			}

		// Print results.
		System.out.printf ("Cover =");
		for (int i = 0; i < V; ++ i)
			if (minCover.contains (i))
				System.out.printf (" %d", i);
		System.out.println();
		System.out.printf ("Size = %d%n", minCover.size());
		}

	/**
	 * Returns true if the given candidate vertex set is a cover.
	 */
	private boolean isCover
		(BitSet64 candidate)
		{
		boolean covered = true;
		for (int i = 0; covered && i < V; ++ i)
			if (! candidate.contains (i))
				covered = adjacent[i].isSubsetOf (candidate);
		return covered;
		}

	/**
	 * Print an error message and exit.
	 */
	private static void usage
		(String msg)
		{
		System.err.printf ("MinVerCovSeq: %s%n", msg);
		usage();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.MinVerCovSeq <file>");
		System.err.println ("<file> = Graph file");
		throw new IllegalArgumentException();
		}

	/**
	 * Specify that this task requires one core.
	 */
	protected static int coresRequired()
		{
		return 1;
		}

	}
