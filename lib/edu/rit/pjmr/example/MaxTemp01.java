//******************************************************************************
//
// File:    MaxTemp01.java
// Package: edu.rit.pjmr.example
// Unit:    Class edu.rit.pjmr.example.MaxTemp01
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

import edu.rit.numeric.ListXYSeries;
import edu.rit.numeric.plot.Plot;
import edu.rit.pj2.vbl.DoubleVbl;
import edu.rit.pjmr.Combiner;
import edu.rit.pjmr.Customizer;
import edu.rit.pjmr.Mapper;
import edu.rit.pjmr.PjmrJob;
import edu.rit.pjmr.Reducer;
import edu.rit.pjmr.TextDirectorySource;
import edu.rit.pjmr.TextId;
import java.io.File;
import java.io.IOException;
import java.util.Date;

/**
 * Class MaxTemp01 is the main program for a PJMR map-reduce job that analyzes
 * climate data files from the U.S. National Climatic Data Center's (NCDC's)
 * Global Historical Climatology Network (GHCN) data set.
 * <P>
 * Usage: <TT>java pj2 [threads=<I>NT</I>] edu.rit.pjmr.example.MaxTemp01
 * <I>nodes</I> <I>directory</I> <I>yearlb</I> <I>yearub</I>
 * <I>plotfile</I></TT>
 * <P>
 * The <I>nodes</I> argument is a comma-separated list of backend node names.
 * The program runs a separate mapper task on each of the given nodes. Each
 * mapper task has one source and <I>NT</I> mappers (default: one mapper). The
 * source reads all files in the given <I>directory</I> on the node where the
 * mapper task is running.
 * <P>
 * For each year in the data set from <I>yearlb</I> to <I>yearub</I> inclusive,
 * the program finds the mean of all the maximum temperatures (in TMAX records)
 * for that year. The program prints a listing of the year and the average
 * maximum temperature (Celsius). The program also creates a plot of average
 * maximum temperature versus year and stores the plot in the given
 * <I>plotfile</I>. Use the {@linkplain View View} program to view the plot.
 *
 * @author  Alan Kaminsky
 * @version 10-Aug-2015
 */
public class MaxTemp01
	extends PjmrJob<TextId,String,Integer,DoubleVbl>
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
		if (args.length != 5) usage();
		String[] nodes = args[0].split (",");
		String directory = args[1];
		int yearlb = Integer.parseInt (args[2]);
		int yearub = Integer.parseInt (args[3]);
		String plotfile = args[4];

		// Determine number of mapper threads.
		int NT = Math.max (threads(), 1);

		// Print provenance.
		System.out.printf
			("$ java pj2 threads=%d edu.rit.pjmr.example.MaxTemp01", NT);
		for (String arg : args)
			System.out.printf (" %s", arg);
		System.out.println();
		System.out.printf ("%s%n", new Date());
		System.out.flush();

		// Configure mapper tasks.
		for (String node : nodes)
			mapperTask (node)
				.source (new TextDirectorySource (directory))
				.mapper (NT, MyMapper.class, ""+yearlb, ""+yearub);

		// Configure reducer task.
		reducerTask()
			.runInJobProcess()
			.customizer (MyCustomizer.class)
			.reducer (MyReducer.class, plotfile);

		startJob();
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 [threads=<NT>] edu.rit.pjmr.example.MaxTemp01 <nodes> <directory> <yearlb> <yearub> <plotfile>");
		throw new IllegalArgumentException();
		}

	/**
	 * Mapper class.
	 */
	private static class MyMapper
		extends Mapper<TextId,String,Integer,DoubleVbl>
		{
		private int yearlb;
		private int yearub;

		// Record year range.
		public void start
			(String[] args,
			 Combiner<Integer,DoubleVbl> combiner)
			{
			yearlb = Integer.parseInt (args[0]);
			yearub = Integer.parseInt (args[1]);
			}

		// Process one data record.
		public void map
			(TextId id,
			 String data,
			 Combiner<Integer,DoubleVbl> combiner)
			{
			// If record is not 269 characters, ignore it.
			if (data.length() < 269) return;

			// Look only at TMAX records.
			if (! data.substring (17, 21) .equals ("TMAX")) return;

			// Get year.
			int year = 0;
			try
				{
				year = parseInt (data, 11, 15);
				}
			catch (NumberFormatException exc)
				{
				return;
				}

			// Look only at years in the specified range.
			if (yearlb > year || year > yearub) return;

			// Look at each day's maximum temperature.
			for (int i = 21; i < 269; i += 8)
				{
				int tmax = 0;
				try
					{
					tmax = parseInt (data, i, i + 5);
					}
				catch (NumberFormatException exc)
					{
					continue;
					}
				char qflag = data.charAt (i + 6);
				if (tmax != -9999 && qflag == ' ')
					combiner.add (year, new DoubleVbl.Mean (tmax));
				}
			}

		// Parse an integer, ignoring leading and trailing whitespace.
		private static int parseInt (String s, int from, int to)
			{
			return Integer.parseInt (s.substring (from, to) .trim());
			}
		}

	/**
	 * Reducer task customizer class.
	 */
	private static class MyCustomizer
		extends Customizer<Integer,DoubleVbl>
		{
		// Sort into ascending order of year (key).
		public boolean comesBefore
			(Integer key_1, DoubleVbl value_1,
			 Integer key_2, DoubleVbl value_2)
			{
			return key_1 < key_2;
			}
		}

	/**
	 * Reducer class.
	 */
	private static class MyReducer
		extends Reducer<Integer,DoubleVbl>
		{
		File plotfile;
		ListXYSeries data;

		// Initialize data series.
		public void start (String[] args)
			{
			plotfile = new File (args[0]);
			data = new ListXYSeries();
			}

		// Print the year (key) and the average maximum temperature (value).
		public void reduce
			(Integer key,
			 DoubleVbl value)
			{
			int year = key.intValue();
			double mean = value.doubleValue()/10.0;
			data.add (year, mean);
			System.out.printf ("%d\t%.1f%n", year, mean);
			System.out.flush();
			}

		// Generate plot.
		public void finish()
			{
			Plot plot = new Plot()
				.xAxisTitle ("Year")
				.yAxisTitle ("Average Maximum Temperature (C)")
				.seriesDots (null)
				.xySeries (data);
			try
				{
				Plot.write (plot, plotfile);
				}
			catch (IOException exc)
				{
				exc.printStackTrace (System.err);
				}
			}
		}

	}
