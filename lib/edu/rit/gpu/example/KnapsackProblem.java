//******************************************************************************
//
// File:    KnapsackProblem.java
// Package: edu.rit.gpu.example
// Unit:    Interface edu.rit.gpu.example.KnapsackProblem
//
// This Java source file is copyright (C) 2016 by Alan Kaminsky. All rights
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

package edu.rit.gpu.example;

import edu.rit.util.Instance;

/**
 * Interface KnapsackProblem is the interface for an object that defines a
 * knapsack problem. The definition consists of the knapsack capacity, the
 * number of items, and the weight and value of each item.
 *
 * @author  Alan Kaminsky
 * @version 15-Feb-2016
 */
public interface KnapsackProblem
	{

// Exported operations.

	/**
	 * Get the knapsack capacity.
	 *
	 * @return  Capacity.
	 */
	public long capacity();

	/**
	 * Get the number of items, <I>N.</I>
	 *
	 * @return  Number of items.
	 */
	public int itemCount();

	/**
	 * Get the weight and value of the next item. This method must be called
	 * <I>N</I> times to get all the items. Each method call returns a new
	 * {@linkplain WV} object.
	 *
	 * @return  Weight/value.
	 *
	 * @exception  NoSuchElementException
	 *     (unchecked exception) Thrown if this method is called more than
	 *     <I>N</I> times.
	 */
	public WV next();

	}
