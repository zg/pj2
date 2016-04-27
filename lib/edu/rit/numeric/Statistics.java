//******************************************************************************
//
// File:    Statistics.java
// Package: edu.rit.numeric
// Unit:    Class edu.rit.numeric.Statistics
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

package edu.rit.numeric;

import static edu.rit.numeric.Mathe.*;

// For unit test main program.
import edu.rit.util.Random;
import edu.rit.util.Sorting;

/**
 * Class Statistics provides static methods for doing statistical tests.
 * <P>
 * For each statistical test, there is a method that returns the "p-value" of
 * the test statistic. This is the probability that the test statistic would
 * have a value greater than or equal to the observed value if the null
 * hypothesis is true.
 *
 * @author  Alan Kaminsky
 * @version 02-Dec-2015
 */
public class Statistics
	{

// Prevent construction.

	private Statistics()
		{
		}

// Exported operations.

	/**
	 * Do a chi-square test on the given data. The null hypothesis is that the
	 * data was drawn from the distribution given by <TT>expected</TT>. The
	 * <TT>measured</TT> and <TT>expected</TT> arrays must be the same length.
	 *
	 * @param  measured  Measured count in each bin.
	 * @param  expected  Expected count in each bin.
	 *
	 * @return  Chi-square statistic.
	 */
	public static double chiSquareTest
		(double[] measured,
		 double[] expected)
		{
		double chisqr = 0.0;
		for (int i = 0; i < measured.length; ++ i)
			{
			double d = measured[i] - expected[i];
			chisqr += d*d/expected[i];
			}
		return chisqr;
		}

	/**
	 * Returns the p-value of a chi-square statistic.
	 *
	 * @param  N       Degrees of freedom.
	 * @param  chisqr  Chi-square statistic.
	 *
	 * @return  P-value.
	 */
	public static double chiSquarePvalue
		(double N,
		 double chisqr)
		{
		return gammq (0.5*N, 0.5*chisqr);
		}

	/**
	 * Do a Bernoulli chi-square test on the given data. The null hypothesis is
	 * that the data was drawn from a Bernoulli distribution with both outcomes
	 * equally likely (e.g., a fair coin). <TT>total</TT> is the total number of
	 * trials. <TT>measured</TT> is the number of trials yielding one of the
	 * outcomes. (<TT>total</TT>&nbsp;&minus;&nbsp;<TT>measured</TT>) is the
	 * number of trials yielding the other outcome.
	 *
	 * @param  total     Total number of trials.
	 * @param  measured  Number of trials yielding one of the outcomes.
	 *
	 * @return  Chi-square statistic.
	 */
	public static double bernoulliChiSquareTest
		(long total,
		 long measured)
		{
		double expected = 0.5*total;
		double d = measured - expected;
		return 2.0*d*d/expected;
		}

	/**
	 * Returns the p-value of a Bernoulli chi-square statistic.
	 *
	 * @param  chisqr  Chi-square statistic.
	 *
	 * @return  P-value.
	 */
	public static double bernoulliChiSquarePvalue
		(double chisqr)
		{
		return gammq (0.5, 0.5*chisqr);
		}

	/**
	 * Do a Y-square test on the given data. The null hypothesis is that the
	 * data was drawn from the distribution given by <TT>expected</TT>. The
	 * <TT>measured</TT> and <TT>expected</TT> arrays must be the same length.
	 * <P>
	 * The Y-square test is similar to the chi-square test, except the Y-square
	 * statistic is valid even if the expected counts in some of the bins are
	 * small, which is not true of the chi-square statistic. For further
	 * information, see:
	 * <P>
	 * L. Lucy. Hypothesis testing for meagre data sets. <I>Monthly Notices of
	 * the Royal Astronomical Society,</I> 318(1):92-100, October 2000.
	 *
	 * @param  N         Degrees of freedom.
	 * @param  measured  Measured count in each bin.
	 * @param  expected  Expected count in each bin.
	 *
	 * @return  Y-square statistic.
	 */
	public static double ySquareTest
		(int N,
		 double[] measured,
		 double[] expected)
		{
		double twoN = 2.0*N;
		double sum = 0.0;
		for (int i = 0; i < expected.length; ++ i)
			{
			sum += 1.0/expected[i];
			}
		return N + Math.sqrt(twoN/(twoN + sum))*
					(chiSquareTest (measured, expected) - N);
		}

	/**
	 * Returns the p-value of a Y-square statistic.
	 *
	 * @param  N     Degrees of freedom.
	 * @param  ysqr  Y-square statistic.
	 *
	 * @return  P-value.
	 */
	public static double ySquarePvalue
		(double N,
		 double ysqr)
		{
		return gammq (0.5*N, 0.5*ysqr);
		}

	/**
	 * Do a Kolmogorov-Smirnov (K-S) test on the given data. The null hypothesis
	 * is that the data was drawn from a uniform distribution between 0.0 and
	 * 1.0.
	 * <P>
	 * The values in the <TT>data</TT> array must all be in the range 0.0
	 * through 1.0 and must be in ascending numerical order. The
	 * <TT>ksTest()</TT> method does not sort the data itself because the
	 * process that produced the data might already have sorted the data. If
	 * necessary, call {@link Sorting#sort(double[])
	 * Sorting.sort}<TT>(data)</TT> before calling <TT>ksTest(data)</TT>.
	 *
	 * @param  data  Data array.
	 *
	 * @return  K-S statistic.
	 */
	public static double ksTest
		(double[] data)
		{
		int M = data.length;
		double N = M;
		double D = 0.0;
		double F_lower = 0.0;
		double F_upper;
		for (int i = 0; i < M; ++ i)
			{
			F_upper = (i+1) / N;
			D = Math.max (D, Math.abs (data[i] - F_lower));
			D = Math.max (D, Math.abs (data[i] - F_upper));
			F_lower = F_upper;
			}
		return D;
		}

	/**
	 * Do a Kolmogorov-Smirnov (K-S) test on the given data. The null hypothesis
	 * is that the data was drawn from the distribution specified by the given
	 * {@linkplain Function}. <TT>cdf.f(x)</TT> must return the value of the
	 * cumulative distribution function at <I>x</I>, in the range 0.0 through
	 * 1.0.
	 * <P>
	 * The values in the <TT>data</TT> array must all be in the domain of
	 * <TT>cdf</TT> and must be in ascending numerical order. The
	 * <TT>ksTest()</TT> method does not sort the data itself because the
	 * process that produced the data might already have sorted the data. If
	 * necessary, call {@link Sorting#sort(double[])
	 * Sorting.sort}<TT>(data)</TT> before calling <TT>ksTest(data)</TT>.
	 *
	 * @param  data  Data array.
	 * @param  cdf   Cumulative distribution function.
	 *
	 * @return  K-S statistic.
	 */
	public static double ksTest
		(double[] data,
		 Function cdf)
		{
		int M = data.length;
		double N = M;
		double D = 0.0;
		double F_lower = 0.0;
		double F_upper;
		double cdf_i;
		for (int i = 0; i < M; ++ i)
			{
			F_upper = (i+1) / N;
			cdf_i = cdf.f (data[i]);
			D = Math.max (D, Math.abs (cdf_i - F_lower));
			D = Math.max (D, Math.abs (cdf_i - F_upper));
			F_lower = F_upper;
			}
		return D;
		}

	/**
	 * Returns the p-value of a K-S statistic.
	 *
	 * @param  N  Number of data points.
	 * @param  D  K-S statistic.
	 *
	 * @return  P-value.
	 */
	public static double ksPvalue
		(double N,
		 double D)
		{
		double sqrt_N = Math.sqrt(N);
		double x = (sqrt_N + 0.12 + 0.11/sqrt_N) * D;
		x = -2.0*x*x;
		double a = 2.0;
		double sum = 0.0;
		double term;
		double absterm;
		double prevterm = 0.0;
		for (int j = 1; j <= 100; ++ j)
			{
			term = a * Math.exp (x*j*j);
			sum += term;
			absterm = Math.abs(term);
			if (absterm <= 1.0e-6*prevterm || absterm <= 1.0e-12*sum)
				{
				return sum;
				}
			a = -a;
			prevterm = absterm;
			}
		return 1.0; // Failed to converge
		}

	/**
	 * Returns the p-value of a statistic drawn from a normal distribution.
	 *
	 * @param  x       Statistic.
	 * @param  mean    Mean of the normal distribution.
	 * @param  stddev  Standard deviation of the normal distribution.
	 *
	 * @return  P-value.
	 */
	public static double normalPvalue
		(double x,
		 double mean,
		 double stddev)
		{
		return 1.0 - 0.5*erfc(-INV_SQRT_2*(x - mean)/stddev);
		}

	private static final double INV_SQRT_2 = 1.0/Math.sqrt(2.0);

	/**
	 * Do an unequal-variance <I>t</I>-test on the two given data series. The
	 * null hypothesis is that the two data series have the same mean; however,
	 * the two series are assumed to have different variances. An array of two
	 * doubles is returned; element 0 gives the <I>t</I> statistic; element 1
	 * gives the <I>p</I>-value (significance) of the <I>t</I> statistic.
	 * Roughly speaking, the <I>p</I>-value is the probability that the
	 * hypothesis is true. If the <I>p</I>-value falls below a significance
	 * threshold, the hypothesis is not true, and the two data series have
	 * different means.
	 *
	 * @param  data1  First data series.
	 * @param  data2  Second data series.
	 *
	 * @return  <I>t</I> statistic and its <I>p</I>-value.
	 */
	public static double[] tTestUnequalVariance
		(Series data1,
		 Series data2)
		{
		int n1 = data1.length();
		Series.Stats stats1 = data1.stats();
		double mean1 = stats1.mean;
		double var1 = stats1.var;
		int n2 = data2.length();
		Series.Stats stats2 = data2.stats();
		double mean2 = stats2.mean;
		double var2 = stats2.var;
		double t = (mean1 - mean2)/Math.sqrt(var1/n1 + var2/n2);
		double df = sqr(var1/n1 + var2/n2)/
			(sqr(var1/n1)/(n1 - 1) + sqr(var2/n2)/(n2 - 1));
		double p = betai (0.5*df, 0.5, df/(df + sqr(t)));
		return new double[] { t, p };
		}

// Unit test main program.

//	/**
//	 * Unit test main program. Does a K-S test on N random doubles, prints the
//	 * K-S statistic, and prints the p-value.
//	 * <P>
//	 * Usage: java edu.rit.numeric.Statistics <I>seed</I> <I>N</I>
//	 * <BR><I>seed</I> = Random seed
//	 * <BR><I>N</I> = Number of data points
//	 */
//	public static void main
//		(String[] args)
//		throws Exception
//		{
//		if (args.length != 2) usage();
//		long seed = Long.parseLong (args[0]);
//		int N = Integer.parseInt (args[1]);
//		Random prng = new Random (seed);
//		double[] data = new double [N];
//		for (int i = 0; i < N; ++ i)
//			{
//			data[i] = prng.nextDouble();
//			}
//		Sorting.sort (data);
//		double D = ksTest (data);
//		System.out.println ("D = " + D);
//		System.out.println ("p = " + ksPvalue (N, D));
//		}
//
//	private static void usage()
//		{
//		System.err.println ("Usage: java edu.rit.numeric.Statistics <seed> <N>");
//		System.err.println ("<seed> = Random seed");
//		System.err.println ("<N> = Number of data points");
//		System.exit (1);
//		}

	}
