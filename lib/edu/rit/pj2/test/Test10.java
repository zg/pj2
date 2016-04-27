//******************************************************************************
//
// File:    Test10.java
// Package: edu.rit.pj2.test
// Unit:    Class edu.rit.pj2.test.Test10
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

import edu.rit.pj2.Job;
import edu.rit.pj2.Rule;
import edu.rit.pj2.Task;
import edu.rit.pj2.TaskSpec;

/**
 * Class Test10 is a unit test main program for classes {@linkplain
 * edu.rit.pj2.Job Job} and {@linkplain edu.rit.pj2.Task Task}. The job runs a
 * task that prints the value of the JVM's <TT>pj2.test</TT> property. This
 * property is set to the string specified on the command line.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.test.Test10 <I>string</I></TT>
 *
 * @author  Alan Kaminsky
 * @version 30-May-2014
 */
public class Test10
	extends Job
	{

// Exported operations.

	/**
	 * Perform this job's computation.
	 *
	 * @param  args  Array of zero or more command line argument strings.
	 *
	 * @exception  Exception
	 *     The <TT>main()</TT> method can throw any exception.
	 */
	public void main
		(String[] args)
		{
		rule() .task (OutputTask.class) .jvmFlags ("-Dpj2.test=" + args[0]);
		}

// Hidden helper classes.

	/**
	 * Class OutputTask prints the value of the JVM's <TT>pj2.test</TT> property
	 * as part of a {@linkplain Test10 Test10} job.
	 *
	 * @author  Alan Kaminsky
	 * @version 12-Dec-2013
	 */
	private static class OutputTask
		extends Task
		{
		public void main
			(String[] args)
			throws Exception
			{
			String value = System.getProperty ("pj2.test");
			if (value == null)
				System.out.printf ("pj2.test = <undefined>%n");
			else
				System.out.printf ("pj2.test = \"%s\"%n", value);
			}

		protected static int coresRequired()
			{
			return 1;
			}
		}

	}
