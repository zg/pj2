//******************************************************************************
//
// File:    Test02.java
// Package: edu.rit.pj2.test
// Unit:    Class edu.rit.pj2.test.Test02
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

package edu.rit.pj2.test;

import edu.rit.pj2.Section;
import edu.rit.pj2.Task;

/**
 * Class Test02 is a unit test main program for the {@link
 * edu.rit.pj2.Task#parallelDo(Section[]) parallelDo()} method of class
 * {@linkplain edu.rit.pj2.Task Task}.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pj2.test.Test02 <I>string</I> ...</TT>
 * <P>
 * Each of the strings on the command line is printed by a separate thread.
 *
 * @author  Alan Kaminsky
 * @version 17-May-2013
 */
public class Test02
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
		int NT = args.length;
		Section[] sections = new Section [NT];
		for (int i = 0; i < NT; ++ i)
			sections[i] = new Test02Section (args[i]);
		parallelDo (sections);
		}

// Hidden helper classes.

	private static class Test02Section
		extends Section
		{
		String arg;

		public Test02Section
			(String arg)
			{
			this.arg = arg;
			}

		public void run()
			{
			System.out.println (arg);
			}
		}

	}
