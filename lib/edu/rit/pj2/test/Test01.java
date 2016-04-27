//******************************************************************************
//
// File:    Test01.java
// Package: edu.rit.pj2.test
// Unit:    Class edu.rit.pj2.test.Test01
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

package edu.rit.pj2.test;

import edu.rit.pj2.Task;

/**
 * Class Test01 is a unit test main program for class {@linkplain
 * edu.rit.pj2.Task Task}. It prints the task's <TT>threads</TT>,
 * <TT>schedule</TT>, <TT>chunk</TT>, <TT>nodeName</TT>, <TT>cores</TT>, and
 * <TT>gpus</TT> properties.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.test.Test01
 *
 * @author  Alan Kaminsky
 * @version 30-May-2014
 */
public class Test01
	extends Task
	{

// Exported operations.

	/**
	 * Perform this task's computation.
	 *
	 * @param  args  Array of zero or more command line argument strings.
	 *
	 * @exception  Exception
	 *     The <TT>main()</TT> method can throw any exception.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		if (threads() == THREADS_EQUALS_CORES)
			System.out.printf ("threads = THREADS_EQUALS_CORES%n");
		else
			System.out.printf ("threads = %d%n", threads());
		System.out.printf ("actual threads = %d%n", actualThreads());
		System.out.printf ("schedule = %s%n", schedule());
		if (chunk() == STANDARD_CHUNK)
			System.out.printf ("chunk = STANDARD_CHUNK%n");
		else
			System.out.printf ("chunk = %d%n", chunk());
		if (nodeName() == ANY_NODE_NAME)
			System.out.printf ("nodeName = ANY_NODE_NAME%n");
		else
			System.out.printf ("nodeName = \"%s\"%n", nodeName());
		if (cores() == ALL_CORES)
			System.out.printf ("cores = ALL_CORES%n");
		else
			System.out.printf ("cores = %d%n", cores());
		if (gpus() == ALL_GPUS)
			System.out.printf ("gpus = ALL_GPUS%n");
		else
			System.out.printf ("gpus = %d%n", gpus());
		}

	}
