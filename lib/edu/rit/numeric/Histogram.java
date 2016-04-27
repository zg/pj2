//******************************************************************************
//
// File:    Histogram.java
// Package: edu.rit.numeric
// Unit:    Class edu.rit.numeric.Histogram
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

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import java.io.IOException;

/**
 * Class Histogram is the base class for a histogram that categorizes values
 * into bins and counts the number of occurrences in each bin. Subclasses of
 * class Histogram categorize items of various types.
 * <P>
 * Class Histogram supports doing a chi-square test on the bin counts. See the
 * {@link #chisqr() chisqr()}, {@link #pvalue(double) pvalue()}, and {@link
 * #expectedCount(int) expectedCount()} methods.
 *
 * @author  Alan Kaminsky
 * @version 26-Mar-2015
 */
public class Histogram
	implements Cloneable, Streamable
	{

// Hidden data members.

	private int B;        // Number of bins
	private long[] count; // Count in each bin
	private long total;   // Total count in all bins

// Exported constructors.

	/**
	 * Construct a new uninitialized histogram. This constructor is for use only
	 * by object streaming.
	 */
	public Histogram()
		{
		}

	/**
	 * Construct a new histogram with the given number of bins.
	 *
	 * @param  B  Number of bins (&ge; 2).
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>B</TT> &lt; 2.
	 */
	public Histogram
		(int B)
		{
		if (B < 2)
			throw new IllegalArgumentException (String.format
				("Histogram(): B = %d illegal", B));
		this.B = B;
		this.count = new long [B];
		this.total = 0;
		}

	/**
	 * Construct a new histogram that is a deep copy of the given histogram.
	 *
	 * @param  hist  Histogram to copy.
	 */
	public Histogram
		(Histogram hist)
		{
		copy (hist);
		}

// Exported operations.

	/**
	 * Clear this histogram. All the bin counts are set to zero.
	 */
	public void clear()
		{
		for (int i = 0; i < B; ++ i)
			count[i] = 0L;
		total = 0L;
		}

	/**
	 * Make this histogram be a deep copy of the given histogram.
	 *
	 * @param  hist  Histogram to copy.
	 *
	 * @return  This histogram.
	 */
	public Histogram copy
		(Histogram hist)
		{
		this.B = hist.B;
		this.count = hist.count == null ? null : (long[]) hist.count.clone();
		this.total = hist.total;
		return this;
		}

	/**
	 * Create a clone of this histogram.
	 *
	 * @return  Clone.
	 */
	public Object clone()
		{
		try
			{
			Histogram hist = (Histogram) super.clone();
			hist.copy (this);
			return hist;
			}
		catch (CloneNotSupportedException exc)
			{
			throw new RuntimeException ("Shouldn't happen", exc);
			}
		}

	/**
	 * Returns the number of bins in this histogram.
	 *
	 * @return  Number of bins.
	 */
	public int size()
		{
		return B;
		}

	/**
	 * Accumulate the given value of type <TT>int</TT> into this histogram.
	 * <P>
	 * The base class implementation of the <TT>accumulate()</TT> method
	 * increments bin number <TT>x</TT>. A subclass may override this method to
	 * increment some other bin number.
	 *
	 * @param  x  Value.
	 *
	 * @see  #increment(int)
	 */
	public void accumulate
		(int x)
		{
		increment (x);
		}

	/**
	 * Accumulate the given value of type <TT>long</TT> into this histogram.
	 * <P>
	 * The base class implementation of the <TT>accumulate()</TT> method throws
	 * an UnsupportedOperationException. If long values are to be accumulated, a
	 * subclass must override this method to increment the proper bin number.
	 *
	 * @param  x  Value.
	 *
	 * @see  #increment(int)
	 */
	public void accumulate
		(long x)
		{
		throw new UnsupportedOperationException();
		}

	/**
	 * Accumulate the given value of type <TT>float</TT> into this histogram.
	 * <P>
	 * The base class implementation of the <TT>accumulate()</TT> method throws
	 * an UnsupportedOperationException. If float values are to be accumulated,
	 * a subclass must override this method to increment the proper bin number.
	 *
	 * @param  x  Value.
	 *
	 * @see  #increment(int)
	 */
	public void accumulate
		(float x)
		{
		throw new UnsupportedOperationException();
		}

	/**
	 * Accumulate the given value of type <TT>double</TT> into this histogram.
	 * <P>
	 * The base class implementation of the <TT>accumulate()</TT> method throws
	 * an UnsupportedOperationException. If double values are to be accumulated,
	 * a subclass must override this method to increment the proper bin number.
	 *
	 * @param  x  Value.
	 *
	 * @see  #increment(int)
	 */
	public void accumulate
		(double x)
		{
		throw new UnsupportedOperationException();
		}

	/**
	 * Accumulate the given object of type <TT>T</TT> into this histogram.
	 * <P>
	 * The base class implementation of the <TT>accumulate()</TT> method throws
	 * an UnsupportedOperationException. If objects are to be accumulated, a
	 * subclass must override this method to increment the proper bin number.
	 *
	 * @param  <T>  Object data type.
	 * @param  x    Value.
	 *
	 * @see  #increment(int)
	 */
	public <T> void accumulate
		(T x)
		{
		throw new UnsupportedOperationException();
		}

	/**
	 * Returns the count in the given bin of this histogram.
	 *
	 * @param  i  Bin number.
	 *
	 * @return  Count in bin <TT>i</TT>.
	 */
	public long count
		(int i)
		{
		return count[i];
		}

	/**
	 * Returns the observed probability of incrementing the count in the given
	 * bin of this histogram. This method returns {@link #count(int)
	 * count(i)}/{@link #total() total()}.
	 *
	 * @param  i  Bin number.
	 *
	 * @return  Observed probability for bin <TT>i</TT>.
	 */
	public double prob
		(int i)
		{
		return (double)count[i]/total;
		}

	/**
	 * Returns the total count in all bins of this histogram.
	 *
	 * @return  Total count.
	 */
	public long total()
		{
		return total;
		}

	/**
	 * Determine the expected count in the given bin for a chi-square test. This
	 * method returns {@link #expectedProb(int) expectedProb(i)}*{@link #total()
	 * total()}.
	 *
	 * @param  i  Bin number.
	 *
	 * @return  Expected count in bin <TT>i</TT>.
	 */
	public double expectedCount
		(int i)
		{
		return expectedProb(i)*total;
		}

	/**
	 * Determine the expected probability of incrementing the given bin for a
	 * chi-square test.
	 * <P>
	 * The base class implementation of this method assumes that the bin counts
	 * are supposed to be all the same; thus, the expected probability for each
	 * bin is the reciprocal of the number of bins. A subclass can override this
	 * method to return different expected probabilities.
	 *
	 * @param  i  Bin number.
	 *
	 * @return  Expected probability for bin <TT>i</TT>.
	 */
	public double expectedProb
		(int i)
		{
		return 1.0/B;
		}

	/**
	 * Returns the chi-square statistic for this histogram. The expected count
	 * in bin <I>i</I> is determined by calling the {@link #expectedCount(int)
	 * expectedCount(i)} method.
	 *
	 * @return  Chi-square statistic.
	 */
	public double chisqr()
		{
		double chisqr = 0.0;
		for (int i = 0; i < B; ++ i)
			{
			double expected = expectedCount (i);
			double d = expected - count[i];
			chisqr += d*d/expected;
			}
		return chisqr;
		}

	/**
	 * Returns the p-value of the given chi-square statistic for this histogram.
	 * The chi-square statistic is assumed to obey the chi-square distribution
	 * with <I>B</I>&minus;1 degrees of freedom, where <I>B</I> is the number of
	 * bins.
	 *
	 * @param  chisqr  Chi-square statistic.
	 *
	 * @return  P-value of <TT>chisqr</TT>.
	 */
	public double pvalue
		(double chisqr)
		{
		return Statistics.chiSquarePvalue (B - 1, chisqr);
		}

	/**
	 * Add the given histogram to this histogram. The count in each bin of
	 * <TT>hist</TT> is added to the count in the corresponding bin of this
	 * histogram. The histograms must have the same number of bins.
	 *
	 * @param  hist  Histogram to add.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>hist</TT> does not have the same
	 *     number of bins as this histogram.
	 */
	public void add
		(Histogram hist)
		{
		if (hist.B != this.B)
			throw new IllegalArgumentException
				("Histogram.add(): Histograms are different sizes");
		for (int i = 0; i < B; ++ i)
			this.count[i] += hist.count[i];
		this.total += hist.total;
		}

	/**
	 * Write this histogram to the given out stream.
	 *
	 * @param  out  Out stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeOut
		(OutStream out)
		throws IOException
		{
		out.writeInt (B);
		out.writeLongArray (count);
		out.writeLong (total);
		}

	/**
	 * Read this histogram from the given in stream.
	 *
	 * @param  in  In stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readIn
		(InStream in)
		throws IOException
		{
		B = in.readInt();
		count = in.readLongArray();
		total = in.readLong();
		}

// Hidden operations.

	/**
	 * Increment the given bin in this histogram. The subclass of class
	 * Histogram determines the manner in which values are categorized into
	 * bins.
	 *
	 * @param  i  Bin number.
	 */
	protected void increment
		(int i)
		{
		++ count[i];
		++ total;
		}

	}
