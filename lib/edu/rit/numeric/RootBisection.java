//******************************************************************************
//
// File:    RootBisection.java
// Package: edu.rit.numeric
// Unit:    Class edu.rit.numeric.RootBisection
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
 * Class RootBisection provides an object that finds roots of a {@linkplain
 * Function} using bisection. Given a real-valued function of a real-valued
 * variable <I>f</I>(<I>x</I>), an instance of a subclass of class Root finds a
 * value <I>x</I> such that <I>f</I>(<I>x</I>) = 0, as follows: The function is
 * evaluated at the midpoint of the bracketing interval; the midpoint replaces
 * the interval endpoint whose function value has the same sign; this is
 * repeated until the difference between the interval endpoints is less than the
 * given tolerance; the interval midpoint is returned as the root.
 *
 * @author  Alan Kaminsky
 * @version 07-Oct-2012
 */
public class RootBisection
	extends Root
	{

// Exported constructors.

	/**
	 * Construct a new object to find roots of the given function.
	 *
	 * @param  func  Function.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>func</TT> is null.
	 */
	public RootBisection
		(Function func)
		{
		super (func);
		}

// Hidden operations.

	/**
	 * Find a root of <TT>func.f()</TT> in the bracketing interval
	 * <TT>brack.x1</TT> to <TT>brack.x2</TT> with tolerance <TT>tol</TT>.
	 * Function values at the bracketing interval endpoints are
	 * <TT>brack.f_x1</TT> and <TT>brack.f_x2</TT>.
	 *
	 * @param  func   Function.
	 * @param  brack  Bracketing interval object.
	 * @param  tol    Tolerance.
	 *
	 * @return  Root of <TT>func.f</TT>(<I>x</I>).
	 */
	protected double findRoot
		(Function func,
		 Bracket brack,
		 double tol)
		{
		double x1 = brack.x1;
		double x2 = brack.x2;
		double xmid = 0.5*(x1 + x2);
		double f_x1 = brack.f_x1;
		double f_x2 = brack.f_x2;
		double f_xmid = func.f (xmid);
		int ntry = 0;
		while (ntry < 50 && f_xmid != 0.0 && Math.abs (x1 - x2) > tol)
			{
			if (Root.sameSign (f_x1, f_xmid))
				{
				x1 = xmid;
				f_x1 = f_xmid;
				}
			else
				{
				x2 = xmid;
				f_x2 = f_xmid;
				}
			xmid = 0.5*(x1 + x2);
			f_xmid = func.f (xmid);
			++ ntry;
			}
		if (ntry == 40)
			throw new TooManyIterationsException
				("RootBisection.findRoot(): Too many iterations");
		return xmid;
		}

	}
