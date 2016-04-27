//******************************************************************************
//
// File:    MineCoinClu.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.MineCoinClu
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

import edu.rit.crypto.SHA256;
import edu.rit.pj2.Job;
import edu.rit.pj2.Task;
import edu.rit.util.Hex;
import edu.rit.util.Packing;

/**
 * Class MineCoinClu is a cluster parallel program that does a simplified
 * version of Bitcoin mining. The program is given a number of coin IDs. For
 * each coin ID, the program finds a 64-bit nonce such that hash(coin ID||nonce)
 * has a given number of leading zero bits. The hash function is double-SHA-256.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.example.MineCoinClu <I>N</I> <I>coinid</I>
 * [<I>coinid</I> ...]</TT>
 * <BR><TT><I>N</I></TT> = Number of leading zero bits (1 .. 63)
 * <BR><TT><I>coinid</I></TT> = Coin ID (hexadecimal)
 * <P>
 * Each coin ID's nonce is computed sequentially as a separate {@linkplain
 * edu.rit.pj2.Task Task}. The tasks are performed in parallel as a {@linkplain
 * edu.rit.pj2.Job Job}. The program measures each coin ID's computation's
 * running time.
 *
 * @author  Alan Kaminsky
 * @version 30-May-2014
 */
public class MineCoinClu
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
		if (args.length < 2) usage();
		int N = Integer.parseInt (args[0]);
		if (1 > N || N > 63) usage();

		// Set up one task for each coin ID.
		for (int i = 1; i < args.length; ++ i)
			rule() .task (MineCoinTask.class) .args (args[i], args[0]);
		}

// Hidden operations.

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.MineCoinClu <N> <coinid> [<coinid> ...]");
		System.err.println ("<N> = Number of leading zero bits (1 .. 63)");
		System.err.println ("<coinid> = Coin ID (hexadecimal)");
		throw new IllegalArgumentException();
		}

// Task subclass.

	/**
	 * Class MineCoinClu.MineCoinTask provides the {@linkplain edu.rit.pj2.Task
	 * Task} that computes one coin ID's nonce in the {@linkplain MineCoinClu
	 * MineCoinClu} program.
	 *
	 * @author  Alan Kaminsky
	 * @version 30-May-2014
	 */
	private static class MineCoinTask
		extends Task
		{
		// Command line arguments.
		byte[] coinId;
		int N;

		// Mask for leading zeroes.
		long mask;

		// For computing hash digests.
		byte[] coinIdPlusNonce;
		SHA256 sha256;
		byte[] digest;

		// Timestamps.
		long t1, t2;

		/**
		 * Task main program.
		 */
		public void main
			(String[] args)
			throws Exception
			{
			// Start timing.
			t1 = System.currentTimeMillis();

			// Parse command line arguments.
			coinId = Hex.toByteArray (args[0]);
			N = Integer.parseInt (args[1]);

			// Set up mask for leading zeroes.
			mask = ~((1L << (64 - N)) - 1L);

			// Set up for computing hash digests.
			coinIdPlusNonce = new byte [coinId.length + 8];
			System.arraycopy (coinId, 0, coinIdPlusNonce, 0, coinId.length);
			sha256 = new SHA256();
			digest = new byte [sha256.digestSize()];

			// Try all nonces until the digest has N leading zero bits.
			for (long nonce = 0L; nonce <= 0x7FFFFFFFFFFFFFFFL; ++ nonce)
				{
				// Test nonce.
				Packing.unpackLongBigEndian
					(nonce, coinIdPlusNonce, coinId.length);
				sha256.hash (coinIdPlusNonce);
				sha256.digest (digest);
				sha256.hash (digest);
				sha256.digest (digest);
				if ((Packing.packLongBigEndian (digest, 0) & mask) == 0L)
					{
					// Stop timing and print result.
					t2 = System.currentTimeMillis();
					System.out.printf ("Coin ID = %s%n",
						Hex.toString (coinId));
					System.out.printf ("Nonce   = %s%n",
						Hex.toString (nonce));
					System.out.printf ("Digest  = %s%n",
						Hex.toString (digest));
					System.out.printf ("%d msec%n", t2 - t1);
					break;
					}
				}
			}

		/**
		 * The task requires one core.
		 */
		protected static int coresRequired()
			{
			return 1;
			}
		}

	}
