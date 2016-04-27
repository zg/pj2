//******************************************************************************
//
// File:    Interpolation.java
// Package: edu.rit.numeric
// Unit:    Class edu.rit.numeric.Interpolation
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
import edu.rit.util.Searching;
import edu.rit.util.Sorting;
import java.io.IOException;

/**
 * Class Interpolation provides an object for interpolating in an {@linkplain
 * XYSeries} of real values (type <TT>double</TT>). Linear interpolation is
 * used. The X-Y series must have at least two elements; the X values must be
 * distinct; but the X values need not be in any particular order. When doing
 * interpolations, the (X,Y) pairs are arranged in ascending order of X values.
 * <P>
 * Class Interpolation implements interface {@linkplain Function}. An instance
 * of class Interpolation can be used as a function object.
 *
 * @author  Alan Kaminsky
 * @version 30-Apr-2015
 */
public class Interpolation
	implements Function, Streamable
	{

// Hidden data members.

	// X and Y values in which to interpolate. X values in ascending order.
	private double[] xData;
	private double[] yData;

// Exported constructors.

	/**
	 * Construct a new uninitialized interpolation object. This constructor is
	 * for use only by object streaming.
	 */
	private Interpolation()
		{
		}

	/**
	 * Construct a new interpolation object that will interpolate between values
	 * in the given X-Y series. The X-Y series must have at least two elements;
	 * the X values must be distinct; but the X values need not be in any
	 * particular order.
	 * <P>
	 * <I>Note:</I> A copy of the given series' elements is made. Changing
	 * <TT>theSeries</TT> will not affect this interpolation object.
	 *
	 * @param  theSeries  X-Y series.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>theSeries</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>theSeries</TT> has fewer than two
	 *     elements. Thrown if the X values in <TT>theSeries</TT> are not
	 *     distinct.
	 */
	public Interpolation
		(XYSeries theSeries)
		{
		int N = theSeries.length();
		if (N < 2)
			throw new IllegalArgumentException
				("Interpolation(): theSeries length < 2");
		xData = new double [N];
		yData = new double [N];
		for (int i = 0; i < N; ++ i)
			{
			xData[i] = theSeries.x(i);
			yData[i] = theSeries.y(i);
			}
		Sorting.sort (xData, new Sorting.Double()
			{
			public void swap (double[] x, int a, int b)
				{
				super.swap (x, a, b);
				double tmp = yData[a];
				yData[a] = yData[b];
				yData[b] = tmp;
				}
			});
		for (int i = 0; i < N - 1; ++ i)
			if (xData[i] == xData[i+1])
				throw new IllegalArgumentException
					("Interpolation(): Duplicate X value: "+xData[i]);
		}

// Exported operations.

	/**
	 * Using linear interpolation, compute the Y value for the given X value.
	 * If <TT>x</TT> is less than the smallest X value in the underlying X-Y
	 * series, the Y value is computed by extrapolating the first interval. If
	 * <TT>x</TT> is greater than the largest X value in the underlying X-Y
	 * series, the Y value is computed by extrapolating the last interval.
	 *
	 * @param  x  X value.
	 *
	 * @return  Interpolated or extrapolated Y value.
	 */
	public double f
		(double x)
		{
		// Find bracketing interval for x.
		int i = Searching.searchInterval (xData, x);
		if (i == 0)
			++ i;
		else if (i == xData.length)
			-- i;
		double x2 = xData[i];
		double x1 = xData[i-1];
		double y2 = yData[i];
		double y1 = yData[i-1];

		// Interpolate on x.
		double t = (x - x1)/(x2 - x1);
		return (1.0 - t)*y1 + t*y2;
		}

	/**
	 * Write this interpolation object to the given out stream.
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
		out.writeDoubleArray (xData);
		out.writeDoubleArray (yData);
		}

	/**
	 * Read this interpolation object from the given in stream.
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
		xData = in.readDoubleArray();
		yData = in.readDoubleArray();
		}

	}
