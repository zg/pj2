//******************************************************************************
//
// File:    RandomGraph.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.RandomGraph
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

import edu.rit.util.Random;
import edu.rit.util.RandomSubset;

/**
 * Class RandomGraph is a main program that generates a random graph. Given a
 * number of vertices <I>V</I> and a number of edges <I>E,</I> the program
 * prints a graph on the standard output consisting of a randomly chosen subset
 * of size <I>E</I> of all possible edges among <I>V</I> vertices. The graph
 * format is as follows: The first line consists of <I>V</I> and <I>E</I>. Each
 * subsequent line consists of two vertex numbers each in the range 0 through
 * <I>V</I>&minus;1, defining an edge between those vertices.
 * <P>
 * Usage: <TT>java edu.rit.pj2.example.RandomGraph <I>V</I> <I>E</I>
 * <I>seed</I></TT>
 * <BR><TT><I>V</I></TT> = Number of vertices, <I>V</I> &ge; 1
 * <BR><TT><I>E</I></TT> = Number of edges, 0 &le; <I>E</I> &le;
 * <I>V</I>(<I>V</I>&minus;1)/2
 * <BR><TT><I>seed</I></TT> = Random seed
 *
 * @author  Alan Kaminsky
 * @version 06-Jun-2014
 */
public class RandomGraph
	{

// Prevent construction.

	private RandomGraph()
		{
		}

// Main program.

	/**
	 * Main program.
	 */
	public static void main
		(String[] args)
		{
		// Parse command line arguments.
		if (args.length != 3) usage();
		int V = Integer.parseInt (args[0]);
		int E = Integer.parseInt (args[1]);
		long seed = Long.parseLong (args[2]);

		// Validate command line arguments.
		if (V < 1)
			usage ("V must be >= 1");
		int N = V*(V - 1)/2;
		if (E < 0 || E > N)
			usage (String.format ("E must be >= 0 and <= %d", N));

		// Set up array of all possible edges on V vertices.
		long[] edges = new long [N];
		int i = 0;
		for (int a = 0; a < V - 1; ++ a)
			for (int b = a + 1; b < V; ++ b)
				edges[i++] = ((long)a << 32) | ((long)b);

		// Print out a random subset of the possible edges.
		Random prng = new Random (seed);
		RandomSubset subset = new RandomSubset (prng, N, true);
		System.out.printf ("%d %d%n", V, E);
		for (i = 0; i < E; ++ i)
			{
			long edge = edges[subset.next()];
			int a = (int)(edge >> 32);
			int b = (int)(edge);
			System.out.printf ("%d %d%n", a, b);
			}
		}

// Hidden operations.

	/**
	 * Print an error message and exit.
	 */
	private static void usage
		(String msg)
		{
		System.err.printf ("RandomGraph: %s%n", msg);
		usage();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java edu.rit.pj2.example.RandomGraph <V> <E> <seed>");
		System.err.println ("<V> = Number of vertices, V >= 1");
		System.err.println ("<E> = Number of edges, 0 <= E <= V(V-1)/2");
		System.err.println ("<seed> = Random seed");
		System.exit (1);
		}

	}
