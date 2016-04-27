//******************************************************************************
//
// File:    Root.java
// Package: edu.rit.numeric
// Unit:    Class edu.rit.numeric.Root
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
 * Class Root is the abstract base class for an object that finds roots of a
 * {@linkplain Function}. Given a real-valued function of a real-valued variable
 * <I>f</I>(<I>x</I>), an instance of a subclass of class Root finds a value
 * <I>x</I> such that <I>f</I>(<I>x</I>) = 0.
 *
 * @author  Alan Kaminsky
 * @version 06-Oct-2012
 */
public abstract class Root
	{

// Hidden data members.

	// The function for which to find roots.
	private Function func;

	// Machine precision.
	private static final double EPS = 2.22e-16;

// Exported constructors.

	/**
	 * Construct a new object to find roots of the given function.
	 *
	 * @param  func  Function.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>func</TT> is null.
	 */
	public Root
		(Function func)
		{
		if (func == null)
			throw new NullPointerException ("Root(): func is null");
		this.func = func;
		}

// Exported operations.

	/**
	 * Find a root of the function in the given bracketing interval. The
	 * <TT>root()</TT> method calls the <TT>brack.bracket()</TT> method to find
	 * the bracketing interval; typically <TT>brack</TT> is a newly constructed
	 * instance of a subclass of class {@linkplain Bracket}. A tolerance of
	 * <I>&epsilon;</I>(|<I>x</I><SUB>1</SUB>|&nbsp;+&nbsp;|<I>x</I><SUB>2</SUB>|)/2
	 * is used, where <I>&epsilon;</I> is the machine precision and the
	 * bracketing interval is <I>x</I><SUB>1</SUB> to <I>x</I><SUB>2</SUB>.
	 *
	 * @param  brack  Bracketing interval object.
	 *
	 * @return  Root of <TT>func.f</TT>(<I>x</I>).
	 */
	public final double root
		(Bracket brack)
		{
		brack.bracket (func);
		return findRoot
			(func, brack,
			 EPS*0.5*(Math.abs(brack.x1) + Math.abs(brack.x2)));
		}

	/**
	 * Find a root of the function in the given bracketing interval to the given
	 * tolerance. The <TT>root()</TT> method calls the <TT>brack.bracket()</TT>
	 * method to find the bracketing interval; typically <TT>brack</TT> is a
	 * newly constructed instance of a subclass of class {@linkplain Bracket}.
	 *
	 * @param  brack  Bracketing interval object.
	 * @param  tol    Tolerance.
	 *
	 * @return  Root of <TT>func.f</TT>(<I>x</I>).
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>tol</TT> &le; 0.
	 */
	public final double root
		(Bracket brack,
		 double tol)
		{
		if (tol <= 0.0)
			throw new IllegalArgumentException (String.format
				("Root.root(): tol = %g illegal", tol));
		brack.bracket (func);
		return findRoot (func, brack, tol);
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
	protected abstract double findRoot
		(Function func,
		 Bracket brack,
		 double tol);

	/**
	 * Returns true if <TT>x1</TT> and <TT>x2</TT> have the same sign.
	 */
	static boolean sameSign
		(double x1,
		 double x2)
		{
		return (x1 < 0.0 && x2 < 0.0) || (x1 >= 0.0 && x2 >= 0.0);
		}

	}
