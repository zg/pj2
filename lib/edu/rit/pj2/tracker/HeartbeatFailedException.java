//******************************************************************************
//
// File:    HeartbeatFailedException.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.HeartbeatFailedException
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

package edu.rit.pj2.tracker;

/**
 * Class HeartbeatFailedException is an exception thrown if a heartbeat fails.
 *
 * @author  Alan Kaminsky
 * @version 09-Jun-2013
 */
public class HeartbeatFailedException
	extends Exception
	{

	/**
	 * Construct a new heartbeat failed exception.
	 */
	public HeartbeatFailedException()
		{
		super();
		}

	/**
	 * Construct a new heartbeat failed exception with the given detail message.
	 *
	 * @param  msg  Detail message.
	 */
	public HeartbeatFailedException
		(String msg)
		{
		super (msg);
		}

	/**
	 * Construct a new heartbeat failed exception with the given chained
	 * exception.
	 *
	 * @param  exc  Chained exception.
	 */
	public HeartbeatFailedException
		(Throwable exc)
		{
		super (exc);
		}

	/**
	 * Construct a new heartbeat failed exception with the given detail message
	 * and chained exception.
	 *
	 * @param  msg  Detail message.
	 * @param  exc  Chained exception.
	 */
	public HeartbeatFailedException
		(String msg,
		 Throwable exc)
		{
		super (msg, exc);
		}

	}
