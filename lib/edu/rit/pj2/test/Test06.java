//******************************************************************************
//
// File:    Test06.java
// Package: edu.rit.pj2.test
// Unit:    Class edu.rit.pj2.test.Test06
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

package edu.rit.pj2.test;

import edu.rit.pj2.ObjectLoop;
import edu.rit.pj2.Task;
import edu.rit.pj2.WorkQueue;

/**
 * Class Test06 is a unit test main program for the {@link
 * edu.rit.pj2.Task#parallelFor(edu.rit.pj2.WorkQueue) parallelFor(WorkQueue)}
 * method of class {@linkplain edu.rit.pj2.Task Task}.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.test.Test06 <I>string</I> ...</TT>
 * <P>
 * The strings on the command line are printed by a work sharing parallel for
 * loop.
 *
 * @author  Alan Kaminsky
 * @version 18-May-2015
 */
public class Test06
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
		WorkQueue<String> strings = new WorkQueue<String>();
		for (int i = 0; i < args.length; ++ i)
			strings.add (args[i]);
		parallelFor (strings) .exec (new ObjectLoop<String>()
			{
			public void run (String s) throws Exception
				{
				System.out.printf ("%d: %s%n", rank(), s);
				System.out.flush();
				Thread.sleep (10L);
				}
			});
		}

	}
