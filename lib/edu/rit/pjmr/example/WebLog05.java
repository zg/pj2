//******************************************************************************
//
// File:    WebLog05.java
// Package: edu.rit.pjmr.example
// Unit:    Class edu.rit.pjmr.example.WebLog05
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

package edu.rit.pjmr.example;

import edu.rit.pj2.vbl.LongVbl;
import edu.rit.pjmr.Combiner;
import edu.rit.pjmr.Customizer;
import edu.rit.pjmr.Mapper;
import edu.rit.pjmr.PjmrJob;
import edu.rit.pjmr.Reducer;
import edu.rit.pjmr.TextFileSource;
import edu.rit.pjmr.TextId;
import java.util.Date;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Class WebLog05 is the main program for a PJMR map-reduce job that analyzes
 * web server log files.
 * <P>
 * Usage: <TT>java pj2 [threads=<I>NT</I>] edu.rit.pjmr.example.WebLog05
 * <I>nodes</I> <I>file</I> <I>ipaddr</I> [ <I>pattern</I> ]</TT>
 * <P>
 * The <I>nodes</I> argument is a comma-separated list of backend node names.
 * The program runs a separate mapper task on each of the given nodes. Each
 * mapper task has one source and <I>NT</I> mappers (default: one mapper). The
 * source reads the given web log <I>file</I> on the node where the mapper task
 * is running.
 * <P>
 * From the web requests made by the given IP address <I>ipaddr</I>, the program
 * lists the unique URLs that were requested, along with the number of requests
 * made to each URL. If the URL refers to a document in a subdirectory, only the
 * top-level directory (up to the first <TT>'/'</TT> character) is listed. The
 * URLs are printed in descending order of number of requests. For an equal
 * number of requests, the URLs are printed in ascending order.
 * <P>
 * If the optional <I>pattern</I> argument is specified, only web log entries
 * that match the pattern are analyzed. For example, the pattern could be a
 * date, to limit the analysis to that date. The pattern must obey the syntax
 * described in class {@linkplain java.util.regex.Pattern
 * java.util.regex.Pattern}. If the <I>pattern</I> argument is omitted, all web
 * log entries are analyzed.
 *
 * @author  Alan Kaminsky
 * @version 03-Aug-2015
 */
public class WebLog05
	extends PjmrJob<TextId,String,String,LongVbl>
	{

	/**
	 * PJMR job main program.
	 *
	 * @param  args  Command line arguments.
	 */
	public void main
		(String[] args)
		{
		// Parse command line arguments.
		if (args.length < 3 || args.length > 4) usage();
		String[] nodes = args[0].split (",");
		String file = args[1];
		String ipaddr = args[2];
		String pattern = null;
		if (args.length >= 4)
			{
			pattern = args[3];
			Pattern.compile (pattern); // Verify that pattern compiles
			}

		// Determine number of mapper threads.
		int NT = Math.max (threads(), 1);

		// Print provenance.
		System.out.printf
			("$ java pj2 threads=%d edu.rit.pjmr.example.WebLog05", NT);
		for (String arg : args)
			System.out.printf (" %s", arg);
		System.out.println();
		System.out.printf ("%s%n", new Date());
		System.out.flush();

		// Configure mapper tasks.
		for (String node : nodes)
			mapperTask (node)
				.source (new TextFileSource (file))
				.mapper (NT, MyMapper.class, ipaddr, pattern);

		// Configure reducer task.
		reducerTask()
			.customizer (MyCustomizer.class)
			.reducer (MyReducer.class);

		startJob();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 [threads=<NT>] edu.rit.pjmr.example.WebLog05 <nodes> <file> <ipaddr> [<pattern>]");
		throw new IllegalArgumentException();
		}

	/**
	 * Mapper class.
	 */
	private static class MyMapper
		extends Mapper<TextId,String,String,LongVbl>
		{
		private static final LongVbl ONE = new LongVbl.Sum (1L);
		private static final Pattern getPattern =
			Pattern.compile ("GET /([^ \\t\\n\\x0B\\f\\r/]+)");
		private Pattern ipaddrPattern;
		private Pattern pattern;

		public void start
			(String[] args,
			 Combiner<String,LongVbl> combiner)
			{
			ipaddrPattern = Pattern.compile ("^" + args[0] + "\\s");
			if (args[1] != null)
				pattern = Pattern.compile (args[1]);
			}

		public void map
			(TextId inKey,   // File name and line number
			 String inValue, // Line from file
			 Combiner<String,LongVbl> combiner)
			{
			if ((pattern == null || pattern.matcher (inValue) .find()) &&
					ipaddrPattern.matcher (inValue) .find())
				{
				Matcher m = getPattern.matcher (inValue);
				if (m.find())
					combiner.add (m.group (1), ONE);
				}
			}
		}

	/**
	 * Reducer task customizer class.
	 */
	private static class MyCustomizer
		extends Customizer<String,LongVbl>
		{
		public boolean comesBefore
			(String key_1, LongVbl value_1, // URL -> # of requests
			 String key_2, LongVbl value_2)
			{
			if (value_1.item > value_2.item)
				return true;
			else if (value_1.item < value_2.item)
				return false;
			else
				return key_1.compareTo (key_2) < 0;
			}
		}

	/**
	 * Reducer class.
	 */
	private static class MyReducer
		extends Reducer<String,LongVbl>
		{
		public void reduce
			(String key,    // URL
			 LongVbl value) // Number of requests
			{
			System.out.printf ("%s\t%s%n", value, key);
			System.out.flush();
			}
		}

	}
