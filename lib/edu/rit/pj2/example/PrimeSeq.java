//******************************************************************************
//
// File:    PrimeSeq.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.PrimeSeq
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

import edu.rit.pj2.Task;

/**
 * Class PrimeSeq is a sequential program that tests numbers for primality. It
 * prints the numbers on the command line that are prime.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.example.PrimeSeq <I>number</I> ...</TT>
 *
 * @author  Alan Kaminsky
 * @version 30-May-2014
 */
public class PrimeSeq
	extends Task
	{

// Main program.

	/**
	 * Main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Validate command line arguments.
		if (args.length < 1) usage();

		// Test numbers for primality.
		for (int i = 0; i < args.length; ++ i)
			if (isPrime (Long.parseLong (args[i])))
				System.out.printf ("%s%n", args[i]);
		}

// Hidden operations.

	/**
	 * Test the given number for primality.
	 *
	 * @param  x  Number &ge; 3.
	 *
	 * @return  True if x is prime, false otherwise.
	 */
	private static boolean isPrime
		(long x)
		{
		if (x % 2 == 0) return false;
		long p = 3;
		long psqr = p*p;
		while (psqr <= x)
			{
			if (x % p == 0) return false;
			p += 2;
			psqr = p*p;
			}
		return true;
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pj2.example.PrimeSeq <number> ...");
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
