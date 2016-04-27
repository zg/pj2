//******************************************************************************
//
// File:    CacheConfig.java
// Package: edu.rit.gpu
// Unit:    Enum edu.rit.gpu.CacheConfig
//
// This Java source file is copyright (C) 2014 by Alan Kaminsky. All rights
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

package edu.rit.gpu;

/**
 * Enum CacheConfig enumerates GPU cache configurations.
 *
 * @author  Alan Kaminsky
 * @version 19-Feb-2014
 */
public enum CacheConfig
	{

	/**
	 * No preference for shared memory or L1 cache (default).
	 */
	CU_FUNC_CACHE_PREFER_NONE (0x00),

	/**
	 * Prefer larger shared memory and smaller L1 cache.
	 */
	CU_FUNC_CACHE_PREFER_SHARED (0x01),

	/**
	 * Prefer larger L1 cache and smaller shared memory.
	 */
	CU_FUNC_CACHE_PREFER_L1 (0x02),

	/**
	 * Prefer equal sized L1 cache and shared memory.
	 */
	CU_FUNC_CACHE_PREFER_EQUAL (0x03);

	/**
	 * Enumeral value.
	 */
	public final int value;

	/**
	 * Construct a new CacheConfig enumeral.
	 *
	 * @param  value  Enumeral value.
	 */
	private CacheConfig
		(int value)
		{
		this.value = value;
		}

	/**
	 * Table of enumerals, indexed by enumeral value.
	 */
	private static CacheConfig[] table = new CacheConfig[]
		{
		CU_FUNC_CACHE_PREFER_NONE,
		CU_FUNC_CACHE_PREFER_SHARED,
		CU_FUNC_CACHE_PREFER_L1,
		CU_FUNC_CACHE_PREFER_EQUAL
		};

	/**
	 * Convert the given enumeral value to an enumeral.
	 *
	 * @param  value  Enumeral value.
	 *
	 * @return  Enumeral.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>value</TT> is illegal.
	 */
	public static CacheConfig of
		(int value)
		{
		return table[value];
		}

	}
