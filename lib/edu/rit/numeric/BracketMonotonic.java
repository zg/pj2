//******************************************************************************
//
// File:    BracketMonotonic.java
// Package: edu.rit.numeric
// Unit:    Class edu.rit.numeric.BracketMonotonic
//
// This Java source file is copyright (C) 2013 by Alan Kaminsky. All rights
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

/**
 * Class BracketMonotonic provides an object that finds a bracketing interval
 * for a root of a monotonic {@linkplain Function}. Given a real-valued function
 * of a real-valued variable <I>f</I>(<I>x</I>), an instance of class
 * BracketMonotonic finds values <I>x</I><SUB>1</SUB> and <I>x</I><SUB>2</SUB>
 * such that <I>f</I>(<I>x</I><SUB>1</SUB>) and <I>f</I>(<I>x</I><SUB>2</SUB>)
 * have opposite signs, as follows: From the initial bracketing interval, the
 * length of the interval is increased in a multiplicative fashion by increasing
 * <I>x</I><SUB>2</SUB> until the interval brackets a root.
 *
 * @author  Alan Kaminsky
 * @version 06-Oct-2012
 */
public class BracketMonotonic
	extends Bracket
	{

// Exported constructors.

	/**
	 * Construct a bracketing interval object with the initial bracketing
	 * interval 0 to 1.
	 */
	public BracketMonotonic()
		{
		super (0.0, 1.0);
		}

	/**
	 * Construct a bracketing interval object with the initial bracketing
	 * interval 0 to <TT>x2</TT>.
	 *
	 * @param  x2  Initial bracketing interval second endpoint.
	 */
	public BracketMonotonic
		(double x2)
		{
		super (0.0, x2);
		}

	/**
	 * Construct a bracketing interval object with the initial bracketing
	 * interval <TT>x1</TT> to <TT>x2</TT>.
	 *
	 * @param  x1  Initial bracketing interval first endpoint.
	 * @param  x2  Initial bracketing interval second endpoint.
	 */
	public BracketMonotonic
		(double x1,
		 double x2)
		{
		super (x1, x2);
		}

// Exported operations.

	/**
	 * Find a bracketing interval for the given function. When called, fields
	 * <TT>x1</TT> and <TT>x2</TT> are the initial bracketing interval
	 * endpoints. Upon return, fields <TT>x1</TT> and <TT>x2</TT> have been
	 * adjusted to bracket a root; fields <TT>f_x1</TT> and <TT>f_x2</TT>
	 * contain the function values at <TT>x1</TT> and <TT>x2</TT>; and
	 * <TT>f_x1</TT> and <TT>f_x2</TT> have opposite signs.
	 *
	 * @param  func  Function.
	 *
	 * @exception  TooManyIterationsException
	 *     (unchecked exception) Thrown if too many iterations have occurred
	 *     without finding a bracketing interval.
	 */
	public void bracket
		(Function func)
		{
		f_x1 = func.f (x1);
		f_x2 = func.f (x2);
		double len = x2 - x1;
		int ntry = 0;
		while (ntry < 50 && Root.sameSign (f_x1, f_x2))
			{
			len *= 1.6;
			x2 = x1 + len;
			f_x2 = func.f (x2);
			++ ntry;
			}
		if (ntry == 50)
			throw new TooManyIterationsException
				("BracketMonotonic.bracket(): Too many iterations");
		}

	}
