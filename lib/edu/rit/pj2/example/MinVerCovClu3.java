//******************************************************************************
//
// File:    MinVerCovClu3.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.MinVerCovClu3
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
import edu.rit.pj2.LongLoop;
import edu.rit.pj2.Task;
import edu.rit.pj2.tuple.ObjectTuple;
import edu.rit.pj2.tuple.ObjectArrayTuple;
import edu.rit.pj2.vbl.BitSetVbl;
import edu.rit.util.BitSet;
import edu.rit.util.Random;
import edu.rit.util.RandomSubset;
import java.io.File;
import java.io.IOException;
import java.util.Scanner;

/**
 * Class MinVerCovClu3 is a cluster parallel program that finds a minimum vertex
 * cover of a graph via heuristic search. The program reads the graph from a
 * file. The graph format is as follows: The first line consists of <I>V</I> and
 * <I>E</I>. Each subsequent line consists of two vertex numbers each in the
 * range 0 through <I>V</I>&minus;1, defining an edge between those vertices.
 * <P>
 * The program performs <I>N</I> trials. For each trial, the program starts with
 * an empty vertex set and adds vertices chosen at random until the vertex set
 * is a cover. The program reports the smallest cover found.
 * <P>
 * Usage: <TT>java pj2 [workers=<I>K</I>] edu.rit.pj2.example.MinVerCovClu3
 * <I>file</I> <I>seed</I> <I>N</I></TT>
 * <BR><TT><I>K</I></TT> = Number of worker tasks (default 1)
 * <BR><TT><I>file</I></TT> = Graph file
 * <BR><TT><I>seed</I></TT> = Random seed
 * <BR><TT><I>N</I></TT> = Number of trials
 *
 * @author  Alan Kaminsky
 * @version 25-Mar-2015
 */
public class MinVerCovClu3
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
		if (args.length != 3) usage();
		File file = new File (args[0]);
		long seed = Long.parseLong (args[1]);
		long N = Long.parseLong (args[2]);

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

		// Put adjacency matrix into tuple space for workers.
		putTuple (new ObjectArrayTuple<BitSet> (adjacent));

		// Set up a task group of K worker tasks.
		masterFor (0, N - 1, WorkerTask.class) .args (""+seed, ""+V);

		// Set up reduction task.
		rule() .atFinish() .task (ReduceTask.class) .args (""+V)
			.runInJobProcess();
		}

	/**
	 * Print an error message and exit.
	 */
	private static void usage
		(String msg)
		{
		System.err.printf ("MinVerCovClu3: %s%n", msg);
		usage();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 [workers=<K>] edu.rit.pj2.example.MinVerCovClu3 <file> <seed> <N>");
		System.err.println ("<K> = Number of worker tasks (default 1)");
		System.err.println ("<file> = Graph file");
		System.err.println ("<seed> = Random seed");
		System.err.println ("<N> = Number of trials");
		throw new IllegalArgumentException();
		}

	/**
	 * Worker task class.
	 */
	private static class WorkerTask
		extends Task
		{
		// Random seed.
		long seed;

		// Number of vertices.
		int V;

		// The graph's adjacency matrix. adjacent[i] is the set of vertices
		// adjacent to vertex i.
		BitSet[] adjacent;

		// Minimum vertex cover.
		BitSetVbl minCover;

		/**
		 * Worker task main program.
		 */
		public void main
			(String[] args)
			throws Exception
			{
			// Parse command line arguments.
			seed = Long.parseLong (args[0]);
			V = Integer.parseInt (args[1]);

			// Read adjacency matrix from tuple space.
			adjacent = readTuple (new ObjectArrayTuple<BitSet>()) .item;

			// Check randomly chosen candidate covers.
			minCover = new BitSetVbl.MinSize (new BitSet (V));
			minCover.bitset.add (0, V);
			workerFor() .exec (new LongLoop()
				{
				BitSetVbl thrMinCover;
				BitSet candidate;
				Random prng;
				RandomSubset rsg;
				public void start()
					{
					thrMinCover = threadLocal (minCover);
					candidate = new BitSet (V);
					prng = new Random (seed + 1000*taskRank() + rank());
					rsg = new RandomSubset (prng, V, true);
					}
				public void run (long i)
					{
					candidate.clear();
					rsg.restart();
					while (! isCover (candidate))
						candidate.add (rsg.next());
					if (candidate.size() < thrMinCover.bitset.size())
						thrMinCover.bitset.copy (candidate);
					}
				});

			// Send best candidate cover to reduction task.
			putTuple (minCover);
			}

		/**
		 * Returns true if the given candidate vertex set is a cover.
		 */
		private boolean isCover
			(BitSet candidate)
			{
			boolean covered = true;
			for (int i = 0; covered && i < V; ++ i)
				if (! candidate.contains (i))
					covered = adjacent[i].isSubsetOf (candidate);
			return covered;
			}
		}

	/**
	 * Reduction task class.
	 */
	private static class ReduceTask
		extends Task
		{
		/**
		 * Reduction task main program.
		 */
		public void main
			(String[] args)
			throws Exception
			{
			// Parse command line arguments.
			int V = Integer.parseInt (args[0]);

			// Reduce all worker task results together.
			BitSetVbl minCover = new BitSetVbl.MinSize (new BitSet (V));
			minCover.bitset.add (0, V);
			BitSetVbl template = new BitSetVbl();
			BitSetVbl taskCover;
			while ((taskCover = tryToTakeTuple (template)) != null)
				minCover.reduce (taskCover);

			// Print final result.
			System.out.printf ("Cover =");
			for (int i = 0; i < V; ++ i)
				if (minCover.bitset.contains (i))
					System.out.printf (" %d", i);
			System.out.println();
			System.out.printf ("Size = %d%n", minCover.bitset.size());
			}
		}

	}
