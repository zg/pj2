//******************************************************************************
//
// File:    MineCoinSmp.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.MineCoinSmp
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

import edu.rit.crypto.SHA256;
import edu.rit.pj2.LongLoop;
import edu.rit.pj2.Schedule;
import edu.rit.pj2.Task;
import edu.rit.util.Hex;
import edu.rit.util.Packing;

/**
 * Class MineCoinSmp is an SMP parallel program that does a simplified version
 * of Bitcoin mining. The program is given a coin ID. The program finds a 64-bit
 * nonce such that hash(coin ID||nonce) has a given number of leading zero bits.
 * The hash function is double-SHA-256.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.example.MineCoinSmp <I>coinid</I>
 * <I>N</I></TT>
 * <BR><TT><I>coinid</I></TT> = Coin ID (hexadecimal)
 * <BR><TT><I>N</I></TT> = Number of leading zero bits (1 .. 63)
 *
 * @author  Alan Kaminsky
 * @version 28-Dec-2013
 */
public class MineCoinSmp
	extends Task
	{

// Program shared variables.

	// Command line arguments.
	byte[] coinId;
	int N;

	// Mask for leading zeroes.
	long mask;

// Main program.

	/**
	 * Main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Validate command line arguments.
		if (args.length != 2) usage();
		coinId = Hex.toByteArray (args[0]);
		N = Integer.parseInt (args[1]);
		if (1 > N || N > 63) usage();

		// Set up mask for leading zeroes.
		mask = ~((1L << (64 - N)) - 1L);

		// Try all nonces until the digest has N leading zero bits.
		parallelFor (0L, 0x7FFFFFFFFFFFFFFFL)
			.schedule (leapfrog) .exec (new LongLoop()
			{
			// For computing hash digests.
			byte[] coinIdPlusNonce;
			SHA256 sha256;
			byte[] digest;

			public void start() throws Exception
				{
				// Set up for computing hash digests.
				coinIdPlusNonce = new byte [coinId.length + 8];
				System.arraycopy (coinId, 0, coinIdPlusNonce, 0, coinId.length);
				sha256 = new SHA256();
				digest = new byte [sha256.digestSize()];
				}

			public void run (long nonce)
				{
				// Test nonce.
				Packing.unpackLongBigEndian (nonce, coinIdPlusNonce,
					coinId.length);
				sha256.hash (coinIdPlusNonce);
				sha256.digest (digest);
				sha256.hash (digest);
				sha256.digest (digest);
				if ((Packing.packLongBigEndian (digest, 0) & mask) == 0L)
					{
					// Print results.
					System.out.printf ("Coin ID = %s%n", Hex.toString (coinId));
					System.out.printf ("Nonce   = %s%n", Hex.toString (nonce));
					System.out.printf ("Digest  = %s%n", Hex.toString (digest));
					stop();
					}
				}
			});
		}

// Hidden operations.

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.MineCoinSmp <coinid> <N>");
		System.err.println ("<coinid> = Coin ID (hexadecimal)");
		System.err.println ("<N> = Number of leading zero bits (1 .. 63)");
		throw new IllegalArgumentException();
		}

	}
