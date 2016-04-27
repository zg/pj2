//******************************************************************************
//
// File:    HamCycClu.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.HamCycClu
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

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.pj2.Job;
import edu.rit.pj2.Task;
import edu.rit.pj2.Tuple;
import edu.rit.pj2.TupleListener;
import edu.rit.pj2.tuple.EmptyTuple;
import edu.rit.pj2.tuple.ObjectArrayTuple;
import edu.rit.util.BitSet;
import java.io.File;
import java.io.IOException;
import java.util.Formatter;
import java.util.Scanner;

/**
 * Class HamCycClu is a cluster parallel program that finds a Hamiltonian
 * cycle in a graph via exhaustive search. The program reads the graph from a
 * file. The graph format is as follows: The first line consists of <I>V</I> and
 * <I>E</I>. Each subsequent line consists of two vertex numbers each in the
 * range 0 through <I>V</I>&minus;1, defining an edge between those vertices.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.example.HamCycClu <I>file</I>
 * <I>threshold</I></TT>
 * <BR><TT><I>file</I></TT> = Graph file
 * <BR><TT><I>threshold</I></TT> = Parallel search threshold level
 * <P>
 * The program traverses the exhaustive search tree down to the given
 * <I>threshold</I> level in a breadth first fashion. The program then searches
 * the subtrees at that level in parallel in a depth first fashion. The
 * <I>threshold</I> should be specified so there are enough subtrees to balance
 * the load among the cluster nodes and cores.
 *
 * @author  Alan Kaminsky
 * @version 21-Jul-2015
 */
public class HamCycClu
	extends Job
	{

	/**
	 * Job main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Parse command line arguments.
		if (args.length != 2) usage();
		File file = new File (args[0]);
		int threshold = Integer.parseInt (args[1]);

		// Read input file, set up adjacency matrix.
		Scanner s = new Scanner (file);
		int V = s.nextInt();
		int E = s.nextInt();
		if (V < 1) usage ("V must be >= 1");
		BitSet[] adjacent = new BitSet [V];
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
		putTuple (new ObjectArrayTuple<BitSet> (adjacent));

		// Set up first work item.
		putTuple (new StateTuple (V));

		// Perform a search task each time there's a new work item.
		rule() .whenMatch (new StateTuple()) .task (SearchTask.class)
			.args (""+threshold);

		// Task to print negative result.
		rule() .atFinish() .task (ResultTask.class) .runInJobProcess();
		}

	/**
	 * Tuple class for a search state.
	 */
	private static class StateTuple
		extends Tuple
		{
		public int[] path;
		public int level;

		public StateTuple()
			{
			}

		public StateTuple
			(int V)
			{
			this.path = new int [V];
			for (int i = 0; i < V; ++ i)
				this.path[i] = i;
			this.level = 0;
			}

		public StateTuple
			(int[] path,
			 int level)
			{
			this.path = (int[]) path.clone();
			this.level = level;
			}

		public void writeOut
			(OutStream out)
			throws IOException
			{
			out.writeIntArray (path);
			out.writeInt (level);
			}

		public void readIn
			(InStream in)
			throws IOException
			{
			path = in.readIntArray();
			level = in.readInt();
			}
		}

	/**
	 * Search task.
	 */
	private static class SearchTask
		extends Task
		{
		// Command line arguments.
		int threshold;

		// Number of vertices.
		int V;

		// The graph's adjacency matrix. adjacent[i] is the set of vertices
		// adjacent to vertex i.
		BitSet[] adjacent;

		// For early loop exit.
		volatile boolean found;

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
				(int[] path,
				 int level)
				{
				this.path = path;
				this.level = level;
				}

			// Search the graph from this state.
			public State search()
				throws Exception
				{
				if (level < threshold)
					return bfs();
				else
					return dfs();
				}

			// Do a breadth first search of the graph from this state.
			private State bfs()
				throws Exception
				{
				// Try extending the path to each vertex adjacent to the current
				// vertex.
				for (int i = level + 1; i < V; ++ i)
					if (adjacent (i))
						{
						++ level;
						swap (level, i);
						putTuple (new StateTuple (path, level));
						-- level;
						}
				return null;
				}

			// Do a depth first search of the graph from this state.
			private State dfs()
				{
				// Base case: Check if there is an edge from the last vertex to
				// the first vertex.
				if (level == V - 1)
					{
					if (adjacent (0))
						return this;
					}

				// Recursive case: Try extending the path to each vertex
				// adjacent to the current vertex.
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

			// Determine if the given path element is adjacent to the current
			// path element.
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

		// Search task main program.
		public void main
			(String[] args)
			throws Exception
			{
			// Parse command line arguments.
			threshold = Integer.parseInt (args[0]);

			// Get adjacency matrix.
			adjacent = readTuple (new ObjectArrayTuple<BitSet>()) .item;
			V = adjacent.length;

			// Early loop exit when any task finds a Hamiltonian cycle.
			addTupleListener (new TupleListener<EmptyTuple> (new EmptyTuple())
				{
				public void run (EmptyTuple tuple)
					{
					found = true;
					}
				});

			// Search for a Hamiltonian cycle starting from the state that
			// triggered this search task.
			StateTuple tuple = (StateTuple) getMatchingTuple (0);
			State hamCycle = new State (tuple.path, tuple.level) .search();

			// Report positive result.
			if (hamCycle != null)
				{
				putTuple (new EmptyTuple());
				System.out.printf ("Hamiltonian cycle =%s%n", hamCycle);
				}
			}

		// The search task requires one core.
		protected static int coresRequired()
			{
			return 1;
			}
		}

	/**
	 * Result task.
	 */
	private static class ResultTask
		extends Task
		{
		// Task main program.
		public void main
			(String[] args)
			throws Exception
			{
			// Report negative result.
			if (tryToReadTuple (new EmptyTuple()) == null)
				System.out.printf ("No Hamiltonian cycle%n");
			}
		}

	/**
	 * Print an error message and exit.
	 */
	private static void usage
		(String msg)
		{
		System.err.printf ("HamCycClu: %s%n", msg);
		usage();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.HamCycClu <file> <threshold>");
		System.err.println ("<file> = Graph file");
		System.err.println ("<threshold> = Parallel search threshold level");
		throw new IllegalArgumentException();
		}

	}
