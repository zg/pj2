//******************************************************************************
//
// File:    TestSet.java
// Package: edu.rit.util.test
// Unit:    Class edu.rit.util.test.TestSet
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

package edu.rit.util.test;

import edu.rit.util.Action;
import edu.rit.util.Set;
import edu.rit.util.IntList;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Class TestSet is a unit test main program for class {@linkplain
 * edu.rit.util.Set Set}. The program creates two sets of strings named
 * <TT>a</TT> and <TT>b</TT>. The program reads commands from the standard
 * input. Each command is of the form <TT>"a.<I>method</I>(<I>args</I>)"</TT> or
 * <TT>"b.<I>method</I>(<I>args</I>)"</TT>. There is no whitespace in the
 * command. The arguments must be strings of digits 0 through 9. The program
 * calls the given method with the given arguments on the given set, then prints
 * the state of both sets. The command <TT>"q"</TT> quits the program.
 * <P>
 * Usage: <TT>java edu.rit.util.test.TestSet</TT>
 *
 * @author  Alan Kaminsky
 * @version 07-Jan-2015
 */
public class TestSet
	{

// Prevent construction.

	private TestSet()
		{
		}

// Unit test main program.

	/**
	 * Unit test main program.
	 */
	public static void main
		(String[] args)
		{
		Set<String> a = new Set<String>();
		Set<String> b = new Set<String>();
		Scanner s = new Scanner (System.in);

		for (;;)
			{
			print ("a", a);
			print ("b", b);

			System.out.print ("? ");
			System.out.flush();
			String cmd = s.nextLine();

			if (cmd.equals ("q"))
				break;

			Set<String> set = null;
			if (cmd.charAt (0) == 'a')
				set = a;
			else if (cmd.charAt (0) == 'b')
				set = b;
			else
				{
				huh();
				continue;
				}

			if (cmd.substring(2).startsWith ("isEmpty"))
				System.out.printf ("%b%n", set.isEmpty());
			else if (cmd.substring(2).startsWith ("clear"))
				set.clear();
			else if (cmd.equals ("a.copy(b)"))
				a.copy (a);
			else if (cmd.equals ("b.copy(a)"))
				b.copy (a);
			else if (cmd.substring(2).startsWith ("size"))
				System.out.printf ("%d%n", set.size());
			else if (cmd.substring(2).startsWith ("contains"))
				{
				Matcher m = INTEGER.matcher (cmd);
				if (m.find())
					System.out.printf ("%b%n", set.contains (m.group()));
				else
					huh();
				}
			else if (cmd.substring(2).startsWith ("add"))
				{
				Matcher m = INTEGER.matcher (cmd);
				if (m.find())
					System.out.printf ("%b%n", set.add (m.group()));
				else
					huh();
				}
			else if (cmd.substring(2).startsWith ("remove"))
				{
				Matcher m = INTEGER.matcher (cmd);
				if (m.find())
					System.out.printf ("%b%n", set.remove (m.group()));
				else
					huh();
				}
			else if (cmd.equals ("a.isSubsetOf(b)"))
				System.out.printf ("%b%n", a.isSubsetOf (b));
			else if (cmd.equals ("b.isSubsetOf(a)"))
				System.out.printf ("%b%n", b.isSubsetOf (a));
			else if (cmd.equals ("a.union(b)"))
				a.union (b);
			else if (cmd.equals ("b.union(a)"))
				b.union (a);
			else if (cmd.equals ("a.intersection(b)"))
				a.intersection (b);
			else if (cmd.equals ("b.intersection(a)"))
				b.intersection (a);
			else if (cmd.equals ("a.difference(b)"))
				a.difference (b);
			else if (cmd.equals ("b.difference(a)"))
				b.difference (a);
			else if (cmd.equals ("a.symmetricDifference(b)"))
				a.symmetricDifference (b);
			else if (cmd.equals ("b.symmetricDifference(a)"))
				b.symmetricDifference (a);
			else
				huh();
			}
		}

	private static final Pattern INTEGER = Pattern.compile ("\\d+");

	private static void huh()
		{
		System.out.printf ("Huh?%n");
		}

	private static void print
		(String label,
		 Set<String> set)
		{
		System.out.printf ("%s = {", label);
		set.forEachItemDo (new Action<String>()
			{
			boolean first = true;
			public void run (String elem)
				{
				System.out.printf ("%s%s", first ? "" : ",", elem);
				first = false;
				}
			});
		System.out.printf ("}%n");
		}

	}
