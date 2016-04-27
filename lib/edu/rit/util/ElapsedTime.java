//******************************************************************************
//
// File:    ElapsedTime.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.ElapsedTime
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

import java.util.Date;

/**
 * Class ElapsedTime provides an object that records an interval of elapsed wall
 * clock time.
 *
 * @author  Alan Kaminsky
 * @version 06-Jun-2013
 */
public class ElapsedTime
	{

// Hidden data members.

	private long t1;
	private long t2;

// Exported constructors.

	/**
	 * Construct a new elapsed time object. The start time is set to now.
	 */
	public ElapsedTime()
		{
		start();
		}

// Exported operations.

	/**
	 * Set this elapsed time object's start time to now.
	 */
	public void start()
		{
		t2 = t1 = System.currentTimeMillis();
		}

	/**
	 * Set this elapsed time object's stop time to now.
	 */
	public void stop()
		{
		t2 = System.currentTimeMillis();
		}

	/**
	 * Get this elapsed time object's start time.
	 *
	 * @return  Start time in milliseconds since midnight 01-Jan-1970.
	 */
	public long startTime()
		{
		return t1;
		}

	/**
	 * Get this elapsed time object's stop time.
	 *
	 * @return  Stop time in milliseconds since midnight 01-Jan-1970.
	 */
	public long stopTime()
		{
		return t2;
		}

	/**
	 * Get this elapsed time object's start time and date.
	 *
	 * @return  Start time and date.
	 */
	public Date startDate()
		{
		return new Date (t1);
		}

	/**
	 * Get this elapsed time object's stop time and date.
	 *
	 * @return  Stop time and date.
	 */
	public Date stopDate()
		{
		return new Date (t2);
		}

	/**
	 * Get this elapsed time object's elapsed time.
	 *
	 * @return  Elapsed time in milliseconds.
	 */
	public long elapsedTime()
		{
		return t2 - t1;
		}

	/**
	 * Returns a string version of this elapsed time object. The string is
	 * <TT>"<I>t</I> msec"</TT>.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		return String.format ("%d msec", t2 - t1);
		}

	}
