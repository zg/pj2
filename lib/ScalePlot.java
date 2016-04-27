//******************************************************************************
//
// File:    ScalePlot.java
// Package: ---
// Unit:    Class ScalePlot
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

import edu.rit.numeric.ListXYSeries;
import edu.rit.numeric.NonNegativeLeastSquares;
import edu.rit.numeric.plot.Dots;
import edu.rit.numeric.plot.Plot;
import edu.rit.numeric.plot.Strokes;
import edu.rit.util.AList;
import edu.rit.util.Action;
import edu.rit.util.DoubleList;
import edu.rit.util.Map;
import edu.rit.util.Set;
import java.awt.Color;
import java.io.File;
import java.net.URLDecoder;
import java.text.DecimalFormat;
import java.util.Scanner;

/**
 * Class ScalePlot is a program that creates plots of a parallel program's
 * performance under scaling, as calculated from measured running time data. The
 * ScalePlot program also prints the plotted data on the standard output.
 * <P>
 * Usage: <TT>java ScalePlot <I>file</I></TT>
 * <P>
 * Each line of the ScalePlot program's input file contains the following data.
 * Each data item is separated from the next by whitespace. Blank lines are
 * ignored.
 * <UL>
 * <P><LI>
 * Problem size label. This is a string denoting the problem size. The string
 * uses URL encoding, as explained in class {@linkplain java.net.URLEncoder
 * URLEncoder}. It is used to label the printout and plots.
 * <P><LI>
 * Problem size <I>N</I>. This is the actual problem size (number of
 * computations).
 * <P><LI>
 * Number of cores <I>K</I>. This is either 0 denoting a sequential program run,
 * or a number &ge; 1 denoting a parallel program run on <I>K</I> cores.
 * <P><LI>
 * One or more running times <I>T</I> in milliseconds. These give the measured
 * running time of the program on <I>K</I> cores on a problem of size <I>N</I>.
 * The ScalePlot program determines the running time <I>T</I>(<I>N,K</I>)
 * to be the minimum <I>T</I> value on the input line for <I>N</I> and <I>K</I>.
 * </UL>
 * <P>
 * The ScalePlot program fits the following model to the (<I>N</I>, <I>K</I>,
 * <I>T</I>) data and prints the model parameters <I>a</I> through <I>f</I>:
 * <P><CENTER>
 * <I>T</I> = (<I>a</I> + <I>bN</I>) + (<I>c</I> + <I>dN</I>)<I>K</I> + (<I>e</I> + <I>fN</I>)/<I>K</I>
 * </CENTER>
 * <P>
 * The ScalePlot program fits the model to the data using the non-negative least
 * squares procedure in class {@linkplain NonNegativeLeastSquares}.
 * <P>
 * The ScalePlot program generates the following plots. Each plot includes
 * one data series for each problem size label.
 * <UL>
 * <P><LI>
 * Parallel program's running time <I>T</I> versus <I>K</I> for each problem
 * size label. A log-log plot.
 * <P><LI>
 * Parallel program's speedup versus <I>K</I> for each problem size label.
 * Speedup is the sequential program's running time on one core divided by the
 * parallel program's running time on <I>K</I> cores, times the parallel
 * program's problem size on <I>K</I> cores divided by the sequential program
 * problem size on one core.
 * <P><LI>
 * Parallel program's efficiency versus <I>K</I> for each problem size label.
 * Efficiency is the speedup divided by <I>K</I>.
 * <P><LI>
 * Parallel program's predicted running time <I>T</I> versus <I>K</I> for each
 * problem size label, based on the fitted running time model. A log-log plot.
 * The model is shown as a red line and the measured data is shown as black
 * dots; this lets you eyeball how well the model matches the data.
 * </UL>
 * <P>
 * The ScalePlot program also prints the running time, speedup, efficiency, and
 * sequential fraction metrics. Referring to the above running time model, the
 * sequential fraction is
 * <I>F</I> = (<I>a</I> + <I>bN</I>)/<I>T</I>(<I>N</I>,1).
 *
 * @author  Alan Kaminsky
 * @version 08-Jan-2015
 */
public class ScalePlot
	{

// Prevent construction.

	private ScalePlot()
		{
		}

// Hidden helper classes.

	private static class Key
		{
		public String label;
		public int K;
		public Key (String label, int K)
			{
			this.label = label;
			this.K = K;
			}
		public boolean equals (Object obj)
			{
			return (obj instanceof Key) &&
				((Key) obj).label.equals (this.label) &&
				((Key) obj).K == this.K;
			}
		public int hashCode()
			{
			return label.hashCode()*31 + K;
			}
		}

// Global variables.

	private static Map<Key,Long> timeMap;
	private static Map<Key,Double> sizeMap;
	private static int Kmax;
	private static double a, b, c, d, e, f;
	private static Plot runningTimePlot;
	private static Plot modelPlot;
	private static Plot sizeupPlot;
	private static Plot efficiencyPlot;

// Main program.

	/**
	 * Main program.
	 */
	public static void main
		(String[] args)
		throws Exception
		{
		// Parse command line arguments.
		if (args.length != 1) usage();
		File file = new File (args[0]);

		// Mapping from (label,K) to T.
		timeMap = new Map<Key,Long>();

		// Mapping from (label,K) to N.
		sizeMap = new Map<Key,Double>();

		// Set and list of problem size labels.
		Set<String> problemSizeSet = new Set<String>();
		AList<String> problemSizeList = new AList<String>();

		// Largest K value encountered.
		Kmax = -1;

		// For calculating running time model parameters.
		AList<double[]> A = new AList<double[]>();
		DoubleList B = new DoubleList();

		// Read input file.
		Scanner scanner = new Scanner (file);
		while (scanner.hasNextLine())
			{
			Scanner linescanner = new Scanner (scanner.nextLine());
			if (! linescanner.hasNext()) continue;
			String label = URLDecoder.decode (linescanner.next(), "UTF-8");
			if (! linescanner.hasNextDouble())
				throw new IllegalArgumentException (String.format
					("N missing for label=\"%s\"", label));
			double N = linescanner.nextDouble();
			if (! linescanner.hasNextInt())
				throw new IllegalArgumentException (String.format
					("K missing for label=\"%s\"", label));
			int K = linescanner.nextInt();
			Kmax = Math.max (Kmax, K);
			long Tmin = Long.MAX_VALUE;
			while (linescanner.hasNextLong())
				Tmin = Math.min (Tmin, linescanner.nextLong());
			if (Tmin == Long.MAX_VALUE)
				throw new IllegalArgumentException (String.format
					("No T data for label=\"%s\", K=%d", label, K));
			timeMap.put (new Key (label, K), Tmin);
			sizeMap.put (new Key (label, K), N);
			if (problemSizeSet.add (label))
				problemSizeList.addLast (label);

			// Accumulate data for running time model.
			if (K >= 1)
				{
				double[] Arow = new double [6];
				Arow[0] = 1.0;
				Arow[1] = N;
				Arow[2] = K;
				Arow[3] = N*K;
				Arow[4] = 1.0/K;
				Arow[5] = N/K;
				A.addLast (Arow);
				B.addLast (Tmin/1000.0);
				}
			}

		// Make sure sequential program data is there.
		problemSizeList.forEachItemDo (new Action<String>()
			{
			public void run (String label)
				{
				Key key = new Key (label, 0);
				if (! timeMap.contains (key))
					throw new IllegalArgumentException (String.format
						("No data for label=\"%s\", K=0", label));
				key = new Key (label, 1);
				if (! timeMap.contains (key))
					throw new IllegalArgumentException (String.format
						("No data for label=\"%s\", K=1", label));
				}
			});

		// Calculate running time model.
		int M = A.size();
		int N = 6;
		NonNegativeLeastSquares model = new NonNegativeLeastSquares (M, N);
		for (int i = 0; i < M; ++ i)
			{
			model.a[i] = A.get(i);
			model.b[i] = B.get(i);
			}
		model.solve();
		a = model.x[0];
		b = model.x[1];
		c = model.x[2];
		d = model.x[3];
		e = model.x[4];
		f = model.x[5];
		System.out.printf ("Running time model%n");
		System.out.printf ("T (sec) = (a + bN) + (c + dN)K + (e + fN)/K%n");
		System.out.printf ("a = %.6g%n", a);
		System.out.printf ("b = %.6g%n", b);
		System.out.printf ("c = %.6g%n", c);
		System.out.printf ("d = %.6g%n", d);
		System.out.printf ("e = %.6g%n", e);
		System.out.printf ("f = %.6g%n", f);
		System.out.printf ("normsqr = %.6g%n", model.normsqr);

		// Print data.
		problemSizeList.forEachItemDo (new Action<String>()
			{
			public void run (String label)
				{
				System.out.println();
				System.out.printf ("Label\tN       \tK\tT\tSpdup\tEffic\tSeqFr%n");
				Key key = new Key (label, 0);
				double N_0 = sizeMap.get (key);
				long T_0 = timeMap.get (key);
				System.out.printf ("%s\t%.2e\tseq\t%d%n", label, N_0, T_0);
				key = new Key (label, 1);
				double N_1 = sizeMap.get (key);
				long T_1 = timeMap.get (key);
				System.out.printf ("\t%.2e\t%d\t%d\t%.3f\t%.3f\t%.3f%n",
					N_1, 1, T_1,
					speedup (N_0, T_0, N_1, T_1),
					effic (N_0, T_0, N_1, T_1, 1),
					seqfr (a, b, N_1, T_1));
				for (int K = 2; K <= Kmax; ++ K)
					{
					key = new Key (label, K);
					if (timeMap.contains (key))
						{
						double N_K = sizeMap.get (key);
						long T_K = timeMap.get (key);
						System.out.printf ("\t%.2e\t%d\t%d\t%.3f\t%.3f%n",
							N_K, K, T_K,
							speedup (N_0, T_0, N_K, T_K),
							effic (N_0, T_0, N_K, T_K, K));
						}
					}
				}
			});

		// Generate running time plot.
		runningTimePlot = new Plot()
			.plotTitle ("Running Time vs. Cores")
			.rightMargin (54)
			.majorGridLines (true)
			.minorGridLines (true)
			.xAxisKind (Plot.LOGARITHMIC)
			.xAxisMinorDivisions (10)
			.xAxisTitle ("Cores")
			.yAxisKind (Plot.LOGARITHMIC)
			.yAxisMinorDivisions (10)
			.yAxisTitle ("Running time (sec)")
			.yAxisTickFormat (new DecimalFormat ("0E0"))
			.labelPosition (Plot.RIGHT)
			.labelOffset (6);
		problemSizeList.forEachItemDo (new Action<String>()
			{
			public void run (String label)
				{
				ListXYSeries series = new ListXYSeries();
				for (int K = 1; K <= Kmax; ++ K)
					{
					Key key = new Key (label, K);
					if (timeMap.contains (key))
						series.add (K, timeMap.get(key)/1000.0);
					}
				runningTimePlot
					.xySeries (series)
					.label (label, series.x (series.length() - 1),
						series.y (series.length() - 1));
				}
			});

		// Generate running time model plot.
		modelPlot = new Plot()
			.plotTitle ("Running Time Model")
			.rightMargin (54)
			.majorGridLines (true)
			.minorGridLines (true)
			.xAxisKind (Plot.LOGARITHMIC)
			.xAxisMinorDivisions (10)
			.xAxisTitle ("Cores")
			.yAxisKind (Plot.LOGARITHMIC)
			.yAxisMinorDivisions (10)
			.yAxisTitle ("Running time (sec)")
			.yAxisTickFormat (new DecimalFormat ("0E0"))
			.labelPosition (Plot.RIGHT)
			.labelOffset (6)
			.seriesDots (null)
			.seriesStroke (Strokes.solid (2))
			.seriesColor (Color.RED);
		problemSizeList.forEachItemDo (new Action<String>()
			{
			public void run (String label)
				{
				ListXYSeries series = new ListXYSeries();
				for (int K = 1; K <= Kmax; ++ K)
					{
					Key key = new Key (label, K);
					if (sizeMap.contains (key))
						{
						double S = sizeMap.get (key);
						series.add (K, (a + b*S) + (c + d*S)*K + (e + f*S)/K);
						}
					}
				modelPlot
					.xySeries (series)
					.label (label, series.x (series.length() - 1),
						series.y (series.length() - 1));
				}
			});
		modelPlot
			.seriesDots (Dots.circle())
			.seriesStroke (null);
		problemSizeList.forEachItemDo (new Action<String>()
			{
			public void run (String label)
				{
				ListXYSeries series = new ListXYSeries();
				for (int K = 1; K <= Kmax; ++ K)
					{
					Key key = new Key (label, K);
					if (timeMap.contains (key))
						series.add (K, timeMap.get(key)/1000.0);
					}
				modelPlot .xySeries (series);
				}
			});

		// Generate speedup plot.
		sizeupPlot = new Plot()
			.plotTitle ("Speedup vs. Cores")
			.rightMargin (54)
			.majorGridLines (true)
			.xAxisTitle ("Cores")
			.yAxisTitle ("Speedup")
			.labelPosition (Plot.RIGHT)
			.labelOffset (6)
			.seriesDots (null)
			.seriesColor (Color.RED)
			.xySeries (0, 0, Kmax, Kmax)
			.seriesDots (Dots.circle())
			.seriesColor (Color.BLACK);
		problemSizeList.forEachItemDo (new Action<String>()
			{
			public void run (String label)
				{
				ListXYSeries series = new ListXYSeries();
				Key key = new Key (label, 0);
				double N_0 = sizeMap.get (key);
				long T_0 = timeMap.get (key);
				for (int K = 1; K <= Kmax; ++ K)
					{
					key = new Key (label, K);
					if (timeMap.contains (key))
						series.add (K, speedup (N_0, T_0, sizeMap.get(key),
							timeMap.get(key)));
					}
				sizeupPlot
					.xySeries (series)
					.label (label, series.x (series.length() - 1),
						series.y (series.length() - 1));
				}
			});

		// Generate efficiency plot.
		efficiencyPlot = new Plot()
			.plotTitle ("Efficiency vs. Cores")
			.rightMargin (54)
			.majorGridLines (true)
			.xAxisTitle ("Cores")
			.yAxisTitle ("Efficiency")
			.yAxisTickFormat (new DecimalFormat ("0.0"))
			.labelPosition (Plot.RIGHT)
			.labelOffset (6)
			.seriesDots (null)
			.seriesColor (Color.RED)
			.xySeries (0, 1, Kmax, 1)
			.seriesDots (Dots.circle())
			.seriesColor (Color.BLACK);
		problemSizeList.forEachItemDo (new Action<String>()
			{
			public void run (String label)
				{
				ListXYSeries series = new ListXYSeries();
				Key key = new Key (label, 0);
				double N_0 = sizeMap.get (key);
				long T_0 = timeMap.get (key);
				for (int K = 1; K <= Kmax; ++ K)
					{
					key = new Key (label, K);
					if (timeMap.contains (key))
						series.add (K, effic (N_0, T_0, sizeMap.get(key),
							timeMap.get(key), K));
					}
				efficiencyPlot
					.xySeries (series)
					.label (label, series.x (series.length() - 1),
						series.y (series.length() - 1));
				}
			});

		// Display plots.
		runningTimePlot.getFrame().setVisible (true);
		sizeupPlot.getFrame().setVisible (true);
		efficiencyPlot.getFrame().setVisible (true);
		modelPlot.getFrame().setVisible (true);
		}

// Hidden operations.

	/**
	 * Compute speedup.
	 */
	private static double speedup
		(double N_0,
		 double T_0,
		 double N_K,
		 double T_K)
		{
		return (N_K/N_0)*(T_0/T_K);
		}

	/**
	 * Compute efficiency.
	 */
	private static double effic
		(double N_0,
		 double T_0,
		 double N_K,
		 double T_K,
		 int K)
		{
		return speedup (N_0, T_0, N_K, T_K) / K;
		}

	/**
	 * Compute sequential fraction.
	 */
	private static double seqfr
		(double a,
		 double b,
		 double N_K,
		 double T_1)
		{
		return (a + b*N_K)/T_1*1000.0;
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java ScalePlot <file>");
		System.exit (1);
		}

	}
