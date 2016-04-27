//******************************************************************************
//
// File:    HamCycSmp.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.HamCycSmp
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

import edu.rit.pj2.ObjectLoop;
import edu.rit.pj2.Task;
import edu.rit.pj2.WorkQueue;
import edu.rit.util.BitSet;
import java.io.File;
import java.util.Formatter;
import java.util.Scanner;

/**
 * Class HamCycSmp is a multicore parallel program that finds a Hamiltonian
 * cycle in a graph via exhaustive search. The program reads the graph from a
 * file. The graph format is as follows: The first line consists of <I>V</I> and
 * <I>E</I>. Each subsequent line consists of two vertex numbers each in the
 * range 0 through <I>V</I>&minus;1, defining an edge between those vertices.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.example.HamCycSmp <I>file</I>
 * <I>threshold</I></TT>
 * <BR><TT><I>file</I></TT> = Graph file
 * <BR><TT><I>threshold</I></TT> = Parallel search threshold level
 * <P>
 * The program traverses the exhaustive search tree down to the given
 * <I>threshold</I> level in a breadth first fashion. The program then searches
 * the subtrees at that level in parallel in a depth first fashion. The
 * <I>threshold</I> should be specified so there are enough subtrees to balance
 * the load among the parallel threads.
 *
 * @author  Alan Kaminsky
 * @version 21-Jul-2015
 */
public class HamCycSmp
	extends Task
	{
	// Command line arguments.
	File file;
	int threshold;

	// Number of vertices and edges.
	int V;
	int E;

	// The graph's adjacency matrix. adjacent[i] is the set of vertices adjacent
	// to vertex i.
	BitSet[] adjacent;

	// For early loop exit.
	volatile boolean found;

	// Parallel work queue.
	WorkQueue<State> queue;

	/**
	 * Class for the search state.
	 */
	private class State
		{
		// Vertices in the path.
		private int[] path;

		// Search level = index of last vertex in the path.
		private int level;

		// Construct a new search state object.
		public State
			(int V)
			{
			path = new int [V];
			for (int i = 0; i < V; ++ i)
				path[i] = i;
			level = 0;
			}

		// Construct a new search state object that is a copy of the given
		// search state object.
		public State
			(State state)
			{
			this.path = (int[]) state.path.clone();
			this.level = state.level;
			}

		// Search the graph from this state.
		public State search()
			{
			if (level < threshold)
				return bfs();
			else
				return dfs();
			}

		// Do a breadth first search of the graph from this state.
		private State bfs()
			{
			// Try extending the path to each vertex adjacent to the current
			// vertex.
			for (int i = level + 1; i < V; ++ i)
				if (adjacent (i))
					{
					++ level;
					swap (level, i);
					queue.add (new State (this));
					-- level;
					}
			return null;
			}

		// Do a depth first search of the graph from this state.
		private State dfs()
			{
			// Base case: Check if there is an edge from the last vertex to the
			// first vertex.
			if (level == V - 1)
				{
				if (adjacent (0))
					return this;
				}

			// Recursive case: Try extending the path to each vertex adjacent to
			// the current vertex.
			else
				{
				for (int i = level + 1; i < V && ! found; ++ i)
					if (adjacent (i))
						{
						++ level;
						swap (level, i);
						if (dfs() != null)
							return this;
						-- level;
						}
				}

			return null;
			}

		// Determine if the given path element is adjacent to the current path
		// element.
		private boolean adjacent
			(int a)
			{
			return adjacent[path[level]].contains (path[a]);
			}

		// Swap the given path elements.
		private void swap
			(int a,
			 int b)
			{
			int t = path[a];
			path[a] = path[b];
			path[b] = t;
			}

		// Returns a string version of this search state object.
		public String toString()
			{
			StringBuilder b = new StringBuilder();
			Formatter f = new Formatter (b);
			for (int i = 0; i <= level; ++ i)
				f.format (" %d", path[i]);
			return b.toString();
			}
		}

	/**
	 * Main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Parse command line arguments.
		if (args.length != 2) usage();
		file = new File (args[0]);
		threshold = Integer.parseInt (args[1]);

		// Read input file, set up adjacency matrix.
		Scanner s = new Scanner (file);
		V = s.nextInt();
		E = s.nextInt();
		if (V < 1) usage ("V must be >= 1");
		adjacent = new BitSet [V];
		for (int i = 0; i < V; ++ i)
			adjacent[i] = new BitSet (V);
		for (int i = 0; i < E; ++ i)
			{
			int a = s.nextInt();
			int b = s.nextInt();
			adjacent[a].add (b);
			adjacent[b].add (a);
			}
		s.close();

		// Set up parallel work queue.
		queue = new WorkQueue<State>();

		// Search the graph in parallel.
		queue.add (new State (V));
		parallelFor (queue) .exec (new ObjectLoop<State>()
			{
			public void run (State state)
				{
				State hamCycle = state.search();
				if (hamCycle != null)
					{
					stop();
					found = true;
					System.out.printf ("Hamiltonian cycle =%s%n", hamCycle);
					}
				}
			});

		// Print negative result.
		if (! found)
			System.out.printf ("No Hamiltonian cycle%n");
		}

	/**
	 * Print an error message and exit.
	 */
	private static void usage
		(String msg)
		{
		System.err.printf ("HamCycSmp: %s%n", msg);
		usage();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.HamCycSmp <file> <threshold>");
		System.err.println ("<file> = Graph file");
		System.err.println ("<threshold> = Parallel search threshold level");
		throw new IllegalArgumentException();
		}

	}
