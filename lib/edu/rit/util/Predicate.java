//******************************************************************************
//
// File:    Predicate.java
// Package: edu.rit.util
// Unit:    Interface edu.rit.util.Predicate
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
 * Interface Predicate specifies the interface for an object that performs some
 * Boolean test on some target object.
 *
 * @param  <T>  Target object data type.
 *
 * @author  Alan Kaminsky
 * @version 16-May-2013
 */
public interface Predicate<T>
	{

	/**
	 * Determine if the predicate is true for the given target object.
	 *
	 * @param  target  Target object.
	 *
	 * @return  Whether the predicate is true or false.
	 */
	public boolean test
		(T target);

	}
