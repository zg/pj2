//******************************************************************************
//
// File:    LongActionResult.java
// Package: edu.rit.util
// Unit:    Interface edu.rit.util.LongActionResult
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
 * Interface LongActionResult specifies the interface for an object that
 * performs some action on some target long integer and that returns some
 * result.
 *
 * @param  <R>  Result data type.
 *
 * @author  Alan Kaminsky
 * @version 15-Jul-2013
 */
public interface LongActionResult<R>
	{

	/**
	 * Perform an action on the given long integer.
	 *
	 * @param  target  Target long integer.
	 */
	public void run
		(long target);

	/**
	 * Return the result.
	 *
	 * @return  Result.
	 */
	public R result();

	}
