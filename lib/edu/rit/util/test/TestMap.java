//******************************************************************************
//
// File:    TestMap.java
// Package: edu.rit.util.test
// Unit:    Class edu.rit.util.test.TestMap
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
import edu.rit.util.Map;
import edu.rit.util.Pair;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Class TestMap is a unit test main program for class {@linkplain
 * edu.rit.util.Map Map}. The program creates two maps named <TT>a</TT> and
 * <TT>b</TT>. The key is a string, the value is a string. The program reads
 * commands from the standard input. Each command is of the form
 * <TT>"a.<I>method</I>(<I>args</I>)"</TT> or
 * <TT>"b.<I>method</I>(<I>args</I>)"</TT>. There is no whitespace in the
 * command. The arguments must be strings of digits 0 through 9. The program
 * calls the given method with the given arguments on the given map, then prints
 * the state of both maps. The command <TT>"q"</TT> quits the program.
 * <P>
 * Usage: <TT>java edu.rit.util.test.TestMap</TT>
 *
 * @author  Alan Kaminsky
 * @version 09-Jan-2015
 */
public class TestMap
	{

// Prevent construction.

	private TestMap()
		{
		}

// Unit test main program.

	/**
	 * Unit test main program.
	 */
	public static void main
		(String[] args)
		{
		Map<String,String> a = new Map<String,String>();
		Map<String,String> b = new Map<String,String>();
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

			Map<String,String> map = null;
			if (cmd.charAt (0) == 'a')
				map = a;
			else if (cmd.charAt (0) == 'b')
				map = b;
			else
				{
				huh();
				continue;
				}

			if (cmd.substring(2).startsWith ("isEmpty"))
				System.out.printf ("%b%n", map.isEmpty());
			else if (cmd.substring(2).startsWith ("clear"))
				map.clear();
			else if (cmd.equals ("a.copy(b)"))
				a.copy (a);
			else if (cmd.equals ("b.copy(a)"))
				b.copy (a);
			else if (cmd.substring(2).startsWith ("size"))
				System.out.printf ("%d%n", map.size());
			else if (cmd.substring(2).startsWith ("contains"))
				{
				Matcher m = INTEGER.matcher (cmd);
				if (m.find())
					System.out.printf ("%b%n", map.contains (m.group()));
				else
					huh();
				}
			else if (cmd.substring(2).startsWith ("get"))
				{
				Matcher m = INTEGER.matcher (cmd);
				if (m.find())
					System.out.printf ("%s%n", map.get (m.group()));
				else
					huh();
				}
			else if (cmd.substring(2).startsWith ("put"))
				{
				Matcher m = INTEGER.matcher (cmd);
				String key = null;
				if (m.find())
					key = m.group();
				String value = null;
				if (m.find())
					value = m.group();
				if (key != null && value != null)
					map.put (key, value);
				else
					huh();
				}
			else if (cmd.substring(2).startsWith ("remove"))
				{
				Matcher m = INTEGER.matcher (cmd);
				if (m.find())
					map.remove (m.group());
				else
					huh();
				}
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
		 Map<String,String> map)
		{
		System.out.printf ("%s =", label);
		map.forEachItemDo (new Action<Pair<String,String>>()
			{
			public void run (Pair<String,String> pair)
				{
				System.out.printf (" (%s,%s)", pair.key(), pair.value());
				}
			});
		System.out.println();
		}

	}
