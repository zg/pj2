//******************************************************************************
//
// File:    Bracket.java
// Package: edu.rit.numeric
// Unit:    Class edu.rit.numeric.Bracket
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
 * Class Bracket is the abstract base class for an object that finds a
 * bracketing interval for a root of a {@linkplain Function}. Given a
 * real-valued function of a real-valued variable <I>f</I>(<I>x</I>), an
 * instance of a subclass of class Bracket finds values <I>x</I><SUB>1</SUB> and
 * <I>x</I><SUB>2</SUB> such that <I>f</I>(<I>x</I><SUB>1</SUB>) and
 * <I>f</I>(<I>x</I><SUB>2</SUB>) have opposite signs. Class {@linkplain Root}
 * uses class Bracket.
 *
 * @author  Alan Kaminsky
 * @version 06-Oct-2012
 */
public abstract class Bracket
	{

// Exported data members.

	/**
	 * First endpoint of bracketing interval.
	 */
	public double x1;

	/**
	 * Second endpoint of bracketing interval.
	 */
	public double x2;

	/**
	 * Function evaluated at first endpoint of bracketing interval.
	 */
	public double f_x1;

	/**
	 * Function evaluated at second endpoint of bracketing interval.
	 */
	public double f_x2;

// Exported constructors.

	/**
	 * Construct a bracketing interval object with the initial bracketing
	 * interval <TT>x1</TT> to <TT>x2</TT>.
	 *
	 * @param  x1  Initial bracketing interval first endpoint.
	 * @param  x2  Initial bracketing interval second endpoint.
	 */
	public Bracket
		(double x1,
		 double x2)
		{
		this.x1 = x1;
		this.x2 = x2;
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
	public abstract void bracket
		(Function func);

	}
