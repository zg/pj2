//******************************************************************************
//
// File:    Mathe.java
// Package: ---
// Unit:    Class Mathe
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

import edu.rit.util.Map;
import static java.lang.Math.*;

/**
 * Class Mathe provides static methods for various mathematical functions. The
 * class name is "Mathe" to distinguish it from class java.lang.Math.
 *
 * @author  Alan Kaminsky
 * @version 11-Apr-2015
 */
public class Mathe
	{

// Prevent construction.

	private Mathe()
		{
		}

// Exported operations.

	/**
	 * Returns <I>x</I><SUP>2</SUP>.
	 */
	public static double sqr
		(double x)
		{
		return x*x;
		}

	/**
	 * Returns the base-2 logarithm of <I>x</I>, log<SUB>2</SUB>&nbsp;<I>x</I>.
	 */
	public static double log2
		(double x)
		{
		return log(x)/LOG_2;
		}

	// Constant for log2().
	private static final double LOG_2 = log (2.0);

	/**
	 * Returns the natural logarithm of the gamma function,
	 * log&nbsp;&Gamma;(<I>x</I>). If <I>x</I> is an integer, then
	 * &Gamma;(<I>x</I>+1) = <I>x</I>!.
	 */
	public static double logGamma
		(double x)
		{
		double y, tmp, ser;

		if (x <= 0.0)
			throw new IllegalArgumentException (String.format
				("Gamma.logGamma(): x = %g illegal", x));

		y = x;
		tmp = x + 5.2421875;
		tmp = (x + 0.5)*log(tmp) - tmp;
		ser = 0.999999999999997092;
		for (int i = 0; i < LGAMMA_COF.length; ++ i)
			{
			y += 1.0;
			ser += LGAMMA_COF[i]/y;
			}
		return tmp + log(2.5066282746310005*ser/x);
		}

	// Coefficients for logGamma().
	private static final double[] LGAMMA_COF = new double[]
		{ 57.1562356658629235,
		 -59.5979603554754912,
		  14.1360979747417471,
		  -0.491913816097620199,
		   0.339946499848118887e-4,
		   0.465236289270485756e-4,
		  -0.983744753048795646e-4,
		   0.158088703224912494e-3,
		  -0.210264441724104883e-3,
		   0.217439618115212643e-3,
		  -0.164318106536763890e-3,
		   0.844182239838527433e-4,
		  -0.261908384015814087e-4,
		   0.368991826595316234e-5};

	/**
	 * Returns the natural logarithm of the factorial function,
	 * log&nbsp;<I>n</I>!.
	 */
	public static double logFactorial
		(long n)
		{
		synchronized (logFactorialTable)
			{
			Double logf = logFactorialTable.get (n);
			if (logf == null)
				{
				logf = logGamma (n + 1);
				logFactorialTable.put (n, logf);
				}
			return logf;
			}
		}

	// Memoization for logFactorial().
	private static Map<Long,Double> logFactorialTable =
		new Map<Long,Double>();

	/**
	 * Returns the natural logarithm of the binomial coefficient,
	 * log&nbsp;choose(<I>n,k</I>).
	 * choose(<I>n,k</I>)&nbsp;=&nbsp;<I>n</I>!/<I>k</I>!/(<I>n</I>&minus;<I>k</I>)!.
	 */
	public static double logChoose
		(long n,
		 long k)
		{
		return logFactorial(n) - logFactorial(n - k) - logFactorial(k);
		}

	/**
	 * Returns the logarithm of the binomial probability.
	 * <TT>logBinomProb(n,k,p)</TT> = log
	 * choose(<I>n,k</I>)&nbsp;+&nbsp;<I>k</I>&nbsp;log&nbsp;<I>p</I>&nbsp;+&nbsp;(<I>n</I>&minus;<I>k</I>)&nbsp;log&nbsp;(1&minus;<I>p</I>).
	 */
	public static double logBinomProb
		(long n,
		 long k,
		 double p)
		{
		return logBinomProb (n, k, log(p), log1p(-p));
		}

	/**
	 * Returns the logarithm of the binomial probability given n, k, log(p), and
	 * log(1-p).
	 */
	private static double logBinomProb
		(long n,
		 long k,
		 double logp,
		 double log1mp)
		{
		return logChoose(n,k) + k*logp + (n - k)*log1mp;
		}

	/**
	 * Returns the binomial probability.
	 * <TT>binomProb(n,k,p)</TT> =
	 * choose(<I>n,k</I>)&nbsp;<I>p</I><SUP><I>k</I></SUP>&nbsp;(1&minus;<I>p</I>)<SUP><I>n</I>&minus;<I>k</I></SUP>.
	 */
	public static double binomProb
		(long n,
		 long k,
		 double p)
		{
		// Special cases.
		if (k < 0L || k > n) return 0.0;
		return exp (logBinomProb (n, k, p));
		}

	/**
	 * Returns the cumulative binomial probability. This is the sum from
	 * <I>i</I> = 0 to <I>k</I> of {@link #binomProb(long,long,double)
	 * binomProb(n,i,p)}.
	 */
	public static double cumulBinomProb
		(long n,
		 long k,
		 double p)
		{
		// Special cases.
		if (k < 0L) return 0.0;
		if (k >= n) return 1.0;

		// Do the log-sum-exp trick.
		double logp = log(p);
		double log1mp = log1p(-p);
		double logmax = logBinomProb (n, k, logp, log1mp);
		double cumul = 1.0;
		for (long i = 0; i < k; ++ i)
			cumul += exp (logBinomProb (n, i, logp, log1mp) - logmax);
		return exp (logmax) * cumul;
		}

	/**
	 * Returns the incomplete gamma function, <I>P</I>(<I>a,x</I>).
	 */
	public static double gammp
		(double a,
		 double x)
		{
		if (a <= 0.0)
			{
			throw new IllegalArgumentException ("gammp(): a = "+a+" illegal");
			}
		if (x < 0.0)
			{
			throw new IllegalArgumentException ("gammp(): x = "+x+" illegal");
			}
		return x == 0.0 ? 0.0 : x < a + 1.0 ? gser(a,x) : 1.0 - gcf(a,x);
		}

	/**
	 * Returns the complementary incomplete gamma function
	 * <I>Q</I>(<I>a,x</I>) = 1&nbsp;&minus;&nbsp;<I>P</I>(<I>a,x</I>).
	 */
	public static double gammq
		(double a,
		 double x)
		{
		if (a <= 0.0)
			{
			throw new IllegalArgumentException ("gammq(): a = "+a+" illegal");
			}
		if (x < 0.0)
			{
			throw new IllegalArgumentException ("gammq(): x = "+x+" illegal");
			}
		return x == 0.0 ? 1.0 : x < a + 1.0 ? 1.0 - gser(a,x) : gcf(a,x);
		}

	/**
	 * Returns the error function erf(<I>x</I>).
	 */
	public static double erf
		(double x)
		{
		return x >= 0.0 ? 1.0 - erfccheb (x) : erfccheb (-x) - 1.0;
		}

	/**
	 * Returns the complementary error function erfc(<I>x</I>).
	 */
	public static double erfc
		(double x)
		{
		return x >= 0.0 ? erfccheb (x) : 2.0 - erfccheb (-x);
		}

	/**
	 * Returns the cumulative normal distribution at <TT>x</TT> for the given
	 * mean <TT>mu</TT> and standard deviation <TT>sigma</TT>.
	 */
	public static double normalCDF
		(double x,
		 double mu,
		 double sigma)
		{
		return 0.5*erfc (-(x - mu)/sigma/SQRT2);
		}

	private static final double SQRT2 = Math.sqrt(2.0);

	/**
	 * Returns the incomplete beta function
	 * <I>I</I><SUB><I>x</I></SUB>(<I>a,b</I>).
	 */
	public static double betai
		(double a,
		 double b,
		 double x)
		{
		if (a <= 0.0)
			{
			throw new IllegalArgumentException ("betai(): a = "+a+" illegal");
			}
		if (b <= 0.0)
			{
			throw new IllegalArgumentException ("betai(): b = "+b+" illegal");
			}
		if (x < 0.0 || x > 1.0)
			{
			throw new IllegalArgumentException ("betai(): x = "+x+" illegal");
			}
		if (x == 0.0 || x == 1.0) return x;
		double bt = Math.exp (logGamma(a+b) - logGamma(a) - logGamma(b) +
			a*Math.log(x) + b*Math.log(1.0 - x));
		if (x < (a + 1.0)/(a + b + 2.0))
			return bt*betacf(a,b,x)/a;
		else
			return 1.0 - bt*betacf(b,a,1.0-x)/b;
		}

// Hidden operations.

	private static final int GAMMA_ITMAX = 200;
	private static final double GAMMA_EPS = 2.22e-16;
	private static final double GAMMA_FPMIN = (2.23e-308/GAMMA_EPS);

	/**
	 * Returns the incomplete gamma function <I>P</I>(<I>a,x</I>), evaluated by
	 * its series representation. Assumes <TT>a</TT> &gt; 0 and <TT>x</TT> &ge;
	 * 0.
	 */
	private static double gser
		(double a,
		 double x)
		{
		double ap, del, sum;
		int i;

		ap = a;
		del = 1.0/a;
		sum = del;
		for (i = 1; i <= GAMMA_ITMAX; ++ i)
			{
			ap += 1.0;
			del *= x/ap;
			sum += del;
			if (Math.abs(del) < Math.abs(sum)*GAMMA_EPS)
				{
				return sum*Math.exp(-x + a*Math.log(x) - logGamma(a));
				}
			}
		return 1.0; // Too many iterations
		}

	/**
	 * Returns the complementary incomplete gamma function <I>Q</I>(<I>a,x</I>),
	 * evaluated by its continued fraction representation. Assumes <TT>a</TT>
	 * &gt; 0 and <TT>x</TT> &ge; 0.
	 */
	private static double gcf
		(double a,
		 double x)
		{
		double b, c, d, h, an, del;
		int i;

		b = x + 1.0 - a;
		c = 1.0/GAMMA_FPMIN;
		d = 1.0/b;
		h = d;
		for (i = 1; i <= GAMMA_ITMAX; ++ i)
			{
			an = -i*(i - a);
			b += 2.0;
			d = an*d + b;
			if (Math.abs(d) < GAMMA_FPMIN) d = GAMMA_FPMIN;
			c = b + an/c;
			if (Math.abs(c) < GAMMA_FPMIN) c = GAMMA_FPMIN;
			d = 1.0/d;
			del = d*c;
			h *= del;
			if (Math.abs(del - 1.0) < GAMMA_EPS)
				{
				return Math.exp(-x + a*Math.log(x) - logGamma(a))*h;
				}
			}
		return 0.0; // Too many iterations
		}

	/**
	 * Returns the complementary error function erfc(<I>x</I>) for <I>x</I> &ge;
	 * 0 using a Chebyshev polynomial approximation.
	 */
	private static double erfccheb
		(double z)
		{
		double t = 2.0/(2.0 + z);
		double ty = 4.0*t - 2.0;
		double d = 0.0;
		double dd = 0.0;
		double tmp;
		for (int j = 27; j > 0; -- j)
			{
			tmp = d;
			d = ty*d - dd + erfcCoeff[j];
			dd = tmp;
			}
		return t*Math.exp(-z*z + 0.5*(erfcCoeff[0] + ty*d) - dd);
		}

	private static final double[] erfcCoeff = new double[]
		{
		-1.3026537197817094,
		 6.4196979235649026e-1,
		 1.9476473204185836e-2,
		-9.561514786808631e-3,
		-9.46595344482036e-4,
		 3.66839497852761e-4,
		 4.2523324806907e-5,
		-2.0278578112534e-5,
		-1.624290004647e-6,
		 1.303655835580e-6,
		 1.5626441722e-8,
		-8.5238095915e-8,
		 6.529054439e-9,
		 5.059343495e-9,
		-9.91364156e-10,
		-2.27365122e-10,
		 9.6467911e-11,
		 2.394038e-12,
		-6.886027e-12,
		 8.94487e-13,
		 3.13092e-13,
		-1.12708e-13,
		 3.81e-16,
		 7.106e-15,
		-1.523e-15,
		-9.4e-17,
		 1.21e-16,
		-2.8e-17,
		};

	private static final int BETA_ITMAX = 10000;
	private static final double BETA_EPS = 2.22e-16;
	private static final double BETA_FPMIN = (2.23e-308/BETA_EPS);

	/**
	 * Returns the incomplete beta function
	 * <I>I</I><SUB><I>x</I></SUB>(<I>a,b</I>) evaluated by its continued
	 * fraction representation.
	 */
	private static double betacf
		(double a,
		 double b,
		 double x)
		{
		int m, m2;
		double aa, c, d, del, h, qab, qam, qap;
		qab = a + b;
		qap = a + 1.0;
		qam = a - 1.0;
		c = 1.0;
		d = 1.0 - qab*x/qap;
		if (Math.abs(d) < BETA_FPMIN) d = BETA_FPMIN;
		d = 1.0/d;
		h = d;
		for (m = 1; m < BETA_ITMAX; ++ m)
			{
			m2 = 2*m;
			aa = m*(b - m)*x/((qam + m2)*(a + m2));
			d = 1.0 + aa*d;
			if (Math.abs(d) < BETA_FPMIN) d = BETA_FPMIN;
			c = 1.0 + aa/c;
			if (Math.abs(c) < BETA_FPMIN) c = BETA_FPMIN;
			d = 1.0/d;
			h *= d*c;
			aa = -(a + m)*(qab + m)*x/((a + m2)*(qap + m2));
			d = 1.0 + aa*d;
			if (Math.abs(d) < BETA_FPMIN) d = BETA_FPMIN;
			c = 1.0 + aa/c;
			if (Math.abs(c) < BETA_FPMIN) c = BETA_FPMIN;
			d = 1.0/d;
			del = d*c;
			h *= del;
			if (Math.abs(del - 1.0) <= BETA_EPS) break;
			}
		return h;
		}

// Unit test main program.

//	/**
//	 * Unit test main program.
//	 */
//	public static void main
//		(String[] args)
//		{
//		// Parse command line arguments.
//		if (args.length != 3) usage();
//		long n = Long.parseLong (args[0]);
//		long k = Long.parseLong (args[1]);
//		double p = Double.parseDouble (args[2]);
//
//		System.out.printf ("n      = %.16e%n", (double)n);
//		System.out.printf ("log n! = %.16e%n", logFactorial(n));
//		System.out.printf ("n!     = %.16e%n", exp(logFactorial(n)));
//		System.out.printf ("k      = %.16e%n", (double)k);
//		System.out.printf ("log k! = %.16e%n", logFactorial(k));
//		System.out.printf ("k!     = %.16e%n", exp(logFactorial(k)));
//		System.out.printf ("log choose(n,k) = %.16e%n", logChoose(n,k));
//		System.out.printf ("choose(n,k)     = %.16e%n", exp(logChoose(n,k)));
//		System.out.printf ("log binomProb(n,k,p)  = %.16e%n", logBinomProb(n,k,p));
//		System.out.printf ("binomProb(n,k,p)      = %.16e%n", binomProb(n,k,p));
//		System.out.printf ("cumulBinomProb(n,k,p) = %.16e%n", cumulBinomProb(n,k,p));
//		}
//
//	/**
//	 * Print a usage message and exit.
//	 */
//	private static void usage()
//		{
//		System.err.println ("Usage: java edu.rit.numeric.Mathe <n> <k> <p>");
//		System.exit (1);
//		}

	}
