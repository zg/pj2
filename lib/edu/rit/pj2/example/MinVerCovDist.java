//******************************************************************************
//
// File:    MinVerCovDist.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.MinVerCovDist
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
import edu.rit.numeric.ListSeries;
import edu.rit.numeric.Series;
import edu.rit.pj2.Job;
import edu.rit.pj2.LongLoop;
import edu.rit.pj2.Loop;
import edu.rit.pj2.Task;
import edu.rit.pj2.Tuple;
import edu.rit.pj2.Vbl;
import edu.rit.util.BitSet64;
import edu.rit.util.Random;
import edu.rit.util.RandomSubset;
import java.io.IOException;
import java.util.Date;

/**
 * Class MinVerCovDist is a multicore cluster parallel program that investigates
 * the distribution of minimum vertex covers in random graphs. The program does
 * <I>N</I> trials. For each trial, the program generates a random graph with
 * <I>V</I> vertices and <I>E</I> edges; finds and counts all the vertex covers
 * of the graph via exhaustive search; and determines <I>F</I>, the fraction of
 * the vertex covers that are <I>minimum</I> vertex covers. The program prints
 * the median and mean absolute deviation of <I>F</I> over all the trials.
 * <P>
 * The program's running time is proportional to 2<SUP><I>V</I></SUP>.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.example.MinVerCovDist <I>N</I> <I>V</I>
 * <I>E</I> <I>seed</I></TT>
 * <BR><TT><I>N</I></TT> = Number of graphs
 * <BR><TT><I>V</I></TT> = Number of vertices
 * <BR><TT><I>E</I></TT> = Number of edges
 * <BR><TT><I>seed</I></TT> = Random seed
 *
 * @author  Alan Kaminsky
 * @version 18-Jan-2015
 */
public class MinVerCovDist
	extends Job
	{

	/**
	 * Job main program.
	 */
	public void main
		(String[] args)
		{
		// Parse command line arguments.
		if (args.length != 4) usage();
		int N = Integer.parseInt (args[0]);
		int V = Integer.parseInt (args[1]);
		int E = Integer.parseInt (args[2]);
		long seed = Long.parseLong (args[3]);
		System.out.printf ("java pj2 edu.rit.pj2.example.MinVerCovDist");
		for (String arg : args) System.out.printf (" %s", arg);
		System.out.printf ("%n%s%n", new Date());
		System.out.flush();

		// Validate command line arguments.
		if (N < 2)
			usage ("N must be >= 2");
		if (V < 1 || V > 63)
			usage ("V must be >= 1 and <= 63");
		int Emax = V*(V - 1)/2;
		if (E < 0 || E > Emax)
			usage (String.format ("E must be >= 0 and <= %d", Emax));

		// Set up worker tasks to do trials.
		masterFor (1, N, WorkerTask.class) .args (""+V, ""+E, ""+seed);

		// Set up result task to gather results from worker tasks.
		rule() .task (ResultTask.class) .args (""+N, ""+V) .runInJobProcess();
		}

	/**
	 * Print an error message and exit.
	 */
	private static void usage
		(String msg)
		{
		System.err.printf ("MinVerCovDist: %s%n", msg);
		usage();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 [workers=<K>] edu.rit.pj2.example.MinVerCovDist <N> <V> <E> <seed>");
		System.err.println ("<K> = Number of worker tasks (default 1)");
		System.err.println ("<N> = Number of graphs");
		System.err.println ("<V> = Number of vertices");
		System.err.println ("<E> = Number of edges");
		System.err.println ("<seed> = Random seed");
		throw new IllegalArgumentException();
		}

	/**
	 * Tuple containing worker task's results.
	 */
	private static class ResultTuple
		extends Tuple
		implements Vbl
		{
			// Trial
		public int trial;
			// Number of vertex covers
		public long coverCount;
			// Number of minimum vertex covers
		public long minCoverCount;
			// Size of minimum vertex cover
		public int minCoverSize = Integer.MAX_VALUE;

		// Construct a new default result tuple.
		public ResultTuple()
			{
			}

		// Construct a new result tuple that is a copy of the given result
		// tuple.
		public ResultTuple (ResultTuple tuple)
			{
			this.trial = tuple.trial;
			this.coverCount = tuple.coverCount;
			this.minCoverCount = tuple.minCoverCount;
			this.minCoverSize = tuple.minCoverSize;
			}

		// Clone this result tuple.
		public Object clone()
			{
			return new ResultTuple (this);
			}

		// Assign this result tuple from the given variable.
		public void set (Vbl vbl)
			{
			ResultTuple tuple = (ResultTuple)vbl;
			this.trial = tuple.trial;
			this.coverCount = tuple.coverCount;
			this.minCoverCount = tuple.minCoverCount;
			this.minCoverSize = tuple.minCoverSize;
			}

		// Reduce the given variable into this result tuple.
		public void reduce (Vbl vbl)
			{
			ResultTuple tuple = (ResultTuple)vbl;
			this.coverCount += tuple.coverCount;
			if (tuple.minCoverSize < this.minCoverSize)
				{
				this.minCoverCount = tuple.minCoverCount;
				this.minCoverSize = tuple.minCoverSize;
				}
			else if (tuple.minCoverSize == this.minCoverSize)
				{
				this.minCoverCount += tuple.minCoverCount;
				}
			}

		// Record the given vertex cover size in this result tuple.
		public void recordCover (int size)
			{
			++ this.coverCount;
			if (size < this.minCoverSize)
				{
				this.minCoverCount = 1;
				this.minCoverSize = size;
				}
			else if (size == this.minCoverSize)
				{
				++ this.minCoverCount;
				}
			}

		// Write this result tuple to the given out stream.
		public void writeOut (OutStream out)
			throws IOException
			{
			out.writeInt (trial);
			out.writeLong (coverCount);
			out.writeLong (minCoverCount);
			out.writeInt (minCoverSize);
			}

		// Read this result tuple from the given in stream.
		public void readIn (InStream in)
			throws IOException
			{
			trial = in.readInt();
			coverCount = in.readLong();
			minCoverCount = in.readLong();
			minCoverSize = in.readInt();
			}
		}

	/**
	 * Worker task class.
	 */
	private static class WorkerTask
		extends Task
		{
		// Command line arguments.
		int V;      // Numer of vertices
		int E;      // Number of edges
		long seed;  // Random seed
		long full;  // Full subset of vertices.

		// For generating random graphs.
		long[] edges;
		Random prng;
		RandomSubset rsg;

		// The graph's adjacency matrix. adjacent[i] is the set of vertices
		// adjacent to vertex i.
		BitSet64[] adjacent;

		// Minimum vertex cover size and count.
		ResultTuple result;

		// Worker task main program.
		public void main
			(String[] args)
			throws Exception
			{
			// Parse command line arguments.
			V = Integer.parseInt (args[0]);
			E = Integer.parseInt (args[1]);
			seed = Long.parseLong (args[2]);
			full = (1L << V) - 1L;

			// Set up array of all possible edges on V vertices.
			int Emax = V*(V - 1)/2;
			edges = new long [Emax];
			int j = 0;
			for (int a = 0; a < V - 1; ++ a)
				for (int b = a + 1; b < V; ++ b)
					edges[j++] = ((long)a << 32) | ((long)b);

			// Set up for generating random graphs.
			prng = new Random (seed + taskRank());
			rsg = new RandomSubset (prng, Emax, true);

			// Set up adjacency matrix.
			adjacent = new BitSet64 [V];
			for (int i = 0; i < V; ++ i)
				adjacent[i] = new BitSet64();

			// Perform trials.
			workerFor() .threads (1) .exec (new Loop()
				{
				public void run (int trial) throws Exception
					{
					// Generate a random graph.
					for (int i = 0; i < V; ++ i)
						adjacent[i].clear();
					rsg.restart();
					for (int i = 0; i < E; ++ i)
						{
						long edge = edges[rsg.next()];
						int a = (int)(edge >> 32);
						int b = (int)(edge);
						adjacent[a].add (b);
						adjacent[b].add (a);
						}

					// Check all candidate covers (sets of vertices).
					result = new ResultTuple();
					result.trial = trial;
					parallelFor (0L, full) .exec (new LongLoop()
						{
						ResultTuple thrResult;
						BitSet64 candidate;
						public void start()
							{
							thrResult = threadLocal (result);
							candidate = new BitSet64();
							}
						public void run (long elems)
							{
							// Count actual covers.
							candidate.bitmap (elems);
							if (isCover (candidate))
								thrResult.recordCover (candidate.size());
							}
						});

					// Report results for this trial.
					putTuple (result);
					}
				});
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
		}

	/**
	 * Result task class.
	 */
	private static class ResultTask
		extends Task
		{
		// Task main program.
		public void main
			(String[] args)
			throws Exception
			{
			// Parse command line arguments.
			int N = Integer.parseInt (args[0]);
			int V = Integer.parseInt (args[1]);
			double twoPowV = Math.pow (2.0, V);

			// Take and print results from worker tasks.
			ResultTuple template = new ResultTuple();
			ResultTuple result;
			ListSeries countData = new ListSeries();
			ListSeries minCountData = new ListSeries();
			ListSeries fractionData = new ListSeries();
			ListSeries sizeData = new ListSeries();
			System.out.printf ("Trial\t#Cov\t#Min\tF\tSize%n");
			System.out.flush();
			for (int i = 0; i < N; ++ i)
				{
				result = takeTuple (template);
				countData.add (result.coverCount);
				minCountData.add (result.minCoverCount);
				double frac = (double)result.minCoverCount/
					(double)result.coverCount;
				fractionData.add (frac);
				sizeData.add (result.minCoverSize);
				System.out.printf ("%d\t%d\t%d\t%.4e\t%d%n",
					result.trial, result.coverCount, result.minCoverCount,
					frac, result.minCoverSize);
				System.out.flush();
				}

			// Print statistics.
			Series.RobustStats rstats = countData.robustStats();
			System.out.printf ("#Cov median     = %.0f%n", rstats.median);
			System.out.printf ("#Cov meanAbsDev = %.4e%n", rstats.meanAbsDev);
			rstats = minCountData.robustStats();
			System.out.printf ("#Min median     = %.0f%n", rstats.median);
			System.out.printf ("#Min meanAbsDev = %.4e%n", rstats.meanAbsDev);
			rstats = fractionData.robustStats();
			System.out.printf ("F median        = %.4e%n", rstats.median);
			System.out.printf ("F meanAbsDev    = %.4e%n", rstats.meanAbsDev);
			rstats = sizeData.robustStats();
			System.out.printf ("Size median     = %.0f%n", rstats.median);
			System.out.printf ("Size meanAbsDev = %.4e%n", rstats.meanAbsDev);
			}
		}

	}
