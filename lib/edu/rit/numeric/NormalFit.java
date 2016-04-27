//******************************************************************************
//
// File:    NormalFit.java
// Package: edu.rit.numeric
// Unit:    Class edu.rit.numeric.NormalFit
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

package edu.rit.numeric;

import edu.rit.numeric.plot.*;
import edu.rit.util.Random;
import edu.rit.util.Sorting;
import java.awt.Color;
import java.text.DecimalFormat;

/**
 * Class NormalFit fits a series of data points to a normal distribution. The
 * null hypothesis is that the data points are drawn from a normal distribution
 * <I>N</I>(<I>&mu;,&sigma;</I>) with unknown mean <I>&mu;</I> and unknown
 * standard deviation <I>&sigma;</I>. The values of <I>&mu;</I> and
 * <I>&sigma;</I> are fitted so as to minimize the maximum absolute difference
 * between the cumulative distribution of the data points and the cumulative
 * distribution of <I>N</I>(<I>&mu;,&sigma;</I>). A Kolmogorov-Smirnov (K-S)
 * test is performed to determine the <I>p</I>-value of the null hypothesis with
 * the fitted <I>&mu;</I> and <I>&sigma;</I> parameters.
 * <P>
 * To do the fitting procedure, call the static {@link #fit(Series)
 * NormalFit.fit(Series)} or the static {@link #fit(double[])
 * NormalFit.fit(double[])} method, passing in the data points. The method
 * returns an instance of class NormalFit whose fields contain the fitting
 * procedure's results.
 *
 * @author  Alan Kaminsky
 * @version 20-Jun-2014
 */
public class NormalFit
	{

// Exported data members.

	/**
	 * The number of original data points.
	 */
	public int N;

	/**
	 * The original data points in ascending order.
	 */
	public double[] data;

	/**
	 * The cumulative distribution of the data points.
	 */
	public double[] cumulDist;

	/**
	 * The mean of the fitted normal distribution.
	 */
	public double mu;

	/**
	 * The standard deviation of the fitted normal distribution.
	 */
	public double sigma;

	/**
	 * The K-S statistic.
	 */
	public double D;

	/**
	 * The <I>p</I>-value of the K-S statistic.
	 */
	public double pvalue;

// Hidden constructors.

	private NormalFit()
		{
		}

// Exported operations.

	/**
	 * Fit the data in the given series to a normal distribution.
	 *
	 * @param  data  Data series.
	 *
	 * @return  A NormalFit object containing the fitting procedure's results.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>data</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>data</TT> is empty.
	 */
	public static NormalFit fit
		(Series data)
		{
		if (data == null)
			throw new NullPointerException
				("NormalFit.fit(): data is null");
		if (data.isEmpty())
			throw new NullPointerException
				("NormalFit.fit(): data is empty");
		NormalFit result = new NormalFit();
		result.N = data.length();
		result.data = new double [result.N];
		for (int i = 0; i < result.N; ++ i)
			result.data[i] = data.x(i);
		return result.performFit();
		}

	/**
	 * Fit the data in the given array to a normal distribution.
	 *
	 * @param  data  Data array.
	 *
	 * @return  A NormalFit object containing the fitting procedure's results.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>data</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>data</TT> is zero length.
	 */
	public static NormalFit fit
		(double[] data)
		{
		if (data == null)
			throw new NullPointerException
				("NormalFit.fit(): data is null");
		if (data.length == 0)
			throw new NullPointerException
				("NormalFit.fit(): data is zero length");
		NormalFit result = new NormalFit();
		result.N = data.length;
		result.data = (double[]) data.clone();
		return result.performFit();
		}

// Hidden operations.

	/**
	 * Fit the data in this NormalFit object to a normal distribution.
	 */
	private NormalFit performFit()
		{
		// Sort the data into ascending order.
		Sorting.sort (data);

		// Set up cumulative distribution of the data.
		cumulDist = new double [N];
		for (int i = 0; i < N; ++ i)
			cumulDist[i] = ((double)i + 1.0)/(double)N;

		// Set up multidimensional function to be minimized.
		MDFunction func = new MDFunction()
			{
			// Function that returns the normal c.d.f. at x with mean mu and
			// standard deviation sigma.
			private double mu, sigma;
			private Function cdf = new Function()
				{
				public double f (double x)
					{
					return normalCDF (x, mu, sigma);
					}
				};

			// MD function has two arguments, mu and sigma.
			public int argumentLength()
				{
				return 2;
				}

			// Returns the MD function to be minimized, namely the K-S statistic
			// for the null hypothesis that the data was drawn from a normal
			// distribution with mean mu and standard deviation sigma.
			public double f (double[] x)
				{
				mu = x[0];
				sigma = x[1];
				return Statistics.ksTest (data, cdf);
				}
			};

		// Set up downhill simplex minimization object.
		MDMinimizationDownhillSimplex mdmin =
			new MDMinimizationDownhillSimplex (func)
				{
				protected void subclassDebug (int iter, int eval)
					{
					System.out.printf ("Iteration %d%n", iter);
					for (int i = 0; i < x.length; ++ i)
						System.out.printf ("  %.4g  %.4g  %.4g%n",
							x[i][0], x[i][1], f[i]);
					}
				};
		mdmin.debug = false;

		// Compute initial guesses for mu and sigma = sample mean and sample
		// standard deviation of the data.
		Series.Stats stats = new ArraySeries (data) .stats();

		// Set initial simplex.
		mdmin.x[0][0] = stats.mean;
		mdmin.x[0][1] = stats.stddev;
		mdmin.x[1][0] = perturb (stats.mean);
		mdmin.x[1][1] = stats.stddev;
		mdmin.x[2][0] = stats.mean;
		mdmin.x[2][1] = perturb (stats.stddev);

		// Adjust mu and sigma to minimize the K-S statistic.
		mdmin.minimize();

		// Record results.
		mu = mdmin.x[0][0];
		sigma = mdmin.x[0][1];
		D = mdmin.f[0];
		pvalue = Statistics.ksPvalue (N, D);

		return this;
		}

	/**
	 * Returns the normal c.d.f. at x with mean mu and standard deviation sigma.
	 */
	private static double normalCDF
		(double x,
		 double mu,
		 double sigma)
		{
		return 0.5*Mathe.erfc (-(x - mu)/sigma/SQRT2);
		}

	private static double SQRT2 = Math.sqrt (2.0);

	/**
	 * Returns x perturbed.
	 */
	private static double perturb
		(double x)
		{
		if (Math.abs(x) > 1.0e-6)
			return x*1.01;
		else if (x > 0.0)
			return x + 1.0e-6;
		else
			return x - 1.0e-6;
		}

// Unit test main program.

	/**
	 * Unit test main program. Generates <I>N</I> random data points from a
	 * normal distribution with mean <I>&mu;</I> and standard deviation
	 * <I>&sigma;</I>, then fits the data to a normal distribution.
	 * <P>
	 * Usage: <TT>java edu.rit.numeric.NormalFit <I>seed</I> <I>N</I> <I>mu</I>
	 * <I>sigma</I></TT>
	 */
	public static void main
		(String[] args)
		{
		if (args.length != 4) usage();
		long seed = Long.parseLong (args[0]);
		int N = Integer.parseInt (args[1]);
		double mu = Double.parseDouble (args[2]);
		double sigma = Double.parseDouble (args[3]);

		NormalPrng prng = new NormalPrng (new Random (seed), mu, sigma);
		double[] data = new double [N];
		for (int i = 0; i < N; ++ i)
			data[i] = prng.next();

		NormalFit fit = NormalFit.fit (data);
		System.out.printf ("mu     = %.4g%n", fit.mu);
		System.out.printf ("sigma  = %.4g%n", fit.sigma);
		System.out.printf ("D      = %.4g%n", fit.D);
		System.out.printf ("pvalue = %.4g%n", fit.pvalue);

		double[] fittedDist = new double [N];
		for (int i = 0; i < N; ++ i)
			fittedDist[i] = normalCDF (fit.data[i], fit.mu, fit.sigma);

		new Plot()
			.yAxisTickFormat (new DecimalFormat ("0.0"))
			.seriesStroke (null)
			.xySeries (fit.data, fit.cumulDist)
			.seriesDots (null)
			.seriesStroke (Strokes.solid (1))
			.seriesColor (Color.RED)
			.xySeries (fit.data, fittedDist)
			.getFrame()
			.setVisible (true);
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java edu.rit.numeric.NormalFit <seed> <N> <mu> <sigma>");
		System.exit (1);
		}

	}
