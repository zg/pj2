//******************************************************************************
//
// File:    Test01.java
// Package: edu.rit.gpu.example
// Unit:    Class edu.rit.gpu.example.Test01
//
// This Java source file is copyright (C) 2016 by Alan Kaminsky. All rights
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

package edu.rit.gpu.example;

import edu.rit.util.Instance;

/**
 * Class Test01 is a unit test main program for a class implementing interface
 * {@linkplain KnapsackProblem KnapsackProblem}.
 * <P>
 * Usage: <TT>java edu.rit.gpu.example.Test01 "<I>constructor</I>"</TT>
 */
class Test01
	{
	public static void main
		(String[] args)
		throws Exception
		{
		if (args.length != 1) usage();
		KnapsackProblem kp = (KnapsackProblem) Instance.newInstance (args[0]);
		System.out.printf ("Capacity = %d%n", kp.capacity());
		int N = kp.itemCount();
		for (int i = 0; i < N; ++ i)
			System.out.printf ("%d\t%s%n", i, kp.next());
		}

	private static void usage()
		{
		System.err.println ("Usage: java edu.rit.gpu.example.Test01 \"<constructor>\"");
		System.exit (1);
		}
	}
