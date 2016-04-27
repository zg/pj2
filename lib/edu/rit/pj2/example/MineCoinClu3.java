//******************************************************************************
//
// File:    MineCoinClu3.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.MineCoinClu3
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

import edu.rit.crypto.SHA256;
import edu.rit.pj2.Job;
import edu.rit.pj2.LongLoop;
import edu.rit.pj2.Task;
import edu.rit.pj2.TupleListener;
import edu.rit.pj2.tuple.EmptyTuple;
import edu.rit.util.Hex;
import edu.rit.util.Packing;

/**
 * Class MineCoinClu3 is a cluster parallel program that does a simplified
 * version of Bitcoin mining. The program is given a coin ID. The program finds
 * a 64-bit nonce such that hash(coin ID||nonce) has a given number of leading
 * zero bits. The hash function is double-SHA-256.
 * <P>
 * Usage: <TT>java pj2 [workers=<I>K</I>] edu.rit.pj2.example.MineCoinClu3
 * <I>coinId</I> <I>N</I></TT>
 * <BR><TT><I>K</I></TT> = Number of worker tasks (default 1)
 * <BR><TT><I>coinId</I></TT> = Coin ID (hexadecimal)
 * <BR><TT><I>N</I></TT> = Number of leading zero bits (1 .. 63)
 * <P>
 * The computation is performed on a cluster parallel computer with <I>K</I>
 * worker tasks running in separate processes on the cluster's backend nodes.
 * The workers use a leapfrog schedule to partition the computation.
 *
 * @author  Alan Kaminsky
 * @version 21-Jul-2015
 */
public class MineCoinClu3
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
		if (args.length != 2) usage();
		String coinId = args[0];
		int N = Integer.parseInt (args[1]);
		if (1 > N || N > 63) usage();

		// Set up master-worker cluster parallel for loop.
		masterSchedule (leapfrog);
		masterFor (0L, 0x7FFFFFFFFFFFFFFFL, WorkerTask.class)
			.args (coinId, ""+N);
		}

// Hidden operations.

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 [workers=<K>] edu.rit.pj2.example.MineCoinClu3 <coinId> <N>");
		System.err.println ("<K> = Number of worker tasks (default 1)");
		System.err.println ("<coinId> = Coin ID (hexadecimal)");
		System.err.println ("<N> = Number of leading zero bits (1 .. 63)");
		throw new IllegalArgumentException();
		}

// Task classes.

	/**
	 * Class MineCoinClu3.WorkerTask provides the worker {@linkplain
	 * edu.rit.pj2.Task Task} in the {@linkplain MineCoinClu3 MineCoinClu3}
	 * program. The worker tasks perform the golden nonce search.
	 *
	 * @author  Alan Kaminsky
	 * @version 21-Jul-2015
	 */
	private static class WorkerTask
		extends Task
		{
		// Command line arguments.
		byte[] coinId;
		int N;

		// Mask for leading zeroes.
		long mask;

		// For early loop exit.
		volatile boolean stop;

		/**
		 * Task main program.
		 */
		public void main
			(String[] args)
			throws Exception
			{
			// Parse command line arguments.
			coinId = Hex.toByteArray (args[0]);
			N = Integer.parseInt (args[1]);

			// Set up mask for leading zeroes.
			mask = ~((1L << (64 - N)) - 1L);

			// Early loop exit when any task finds the golden nonce.
			addTupleListener (new TupleListener<EmptyTuple> (new EmptyTuple())
				{
				public void run (EmptyTuple tuple)
					{
					stop = true;
					}
				});

			// Try all nonces until the digest has N leading zero bits.
			workerFor() .schedule (leapfrog) .exec (new LongLoop()
				{
				byte[] coinIdPlusNonce;
				SHA256 sha256;
				byte[] digest;

				public void start()
					{
					coinIdPlusNonce = new byte [coinId.length + 8];
					System.arraycopy (coinId, 0, coinIdPlusNonce, 0,
						coinId.length);
					sha256 = new SHA256();
					digest = new byte [sha256.digestSize()];
					}

				public void run (long nonce) throws Exception
					{
					if (stop) stop();
					Packing.unpackLongBigEndian (nonce, coinIdPlusNonce,
						coinId.length);
					sha256.hash (coinIdPlusNonce);
					sha256.digest (digest);
					sha256.hash (digest);
					sha256.digest (digest);
					if ((Packing.packLongBigEndian (digest, 0) & mask)
							== 0L)
						{
						putTuple (new EmptyTuple());
						System.out.printf ("Coin ID = %s%n",
							Hex.toString (coinId));
						System.out.printf ("Nonce   = %s%n",
							Hex.toString (nonce));
						System.out.printf ("Digest  = %s%n",
							Hex.toString (digest));
						stop();
						}
					}
				});
			}
		}

	}
