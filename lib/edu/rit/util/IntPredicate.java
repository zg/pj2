//******************************************************************************
//
// File:    IntPredicate.java
// Package: edu.rit.util
// Unit:    Interface edu.rit.util.IntPredicate
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

package edu.rit.util;

/**
 * Interface IntPredicate specifies the interface for an object that performs
 * some Boolean test on some target integer (type <TT>int</TT>).
 *
 * @author  Alan Kaminsky
 * @version 27-May-2013
 */
public interface IntPredicate
	{

	/**
	 * Determine if the predicate is true for the given target integer.
	 *
	 * @param  target  Target integer.
	 *
	 * @return  Whether the predicate is true or false.
	 */
	public boolean test
		(int target);

	}
