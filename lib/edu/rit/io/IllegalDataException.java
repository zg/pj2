//******************************************************************************
//
// File:    IllegalDataException.java
// Package: edu.rit.io
// Unit:    Class edu.rit.io.IllegalDataException
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

package edu.rit.io;

import java.io.IOException;

/**
 * Class IllegalDataException indicates that an {@linkplain InStream} method
 * read illegal data.
 *
 * @author  Alan Kaminsky
 * @version 05-Nov-2013
 */
public class IllegalDataException
	extends IOException
	{

// Exported constructors.

	/**
	 * Construct a new illegal data exception with no detail message and no
	 * chained exception.
	 */
	public IllegalDataException()
		{
		super();
		}

	/**
	 * Construct a new illegal data exception with the given detail message and
	 * no chained exception.
	 *
	 * @param  msg  Detail message.
	 */
	public IllegalDataException
		(String msg)
		{
		super (msg);
		}

	/**
	 * Construct a new illegal data exception with no detail message and the
	 * given chained exception.
	 *
	 * @param  exc  Chained exception.
	 */
	public IllegalDataException
		(Throwable exc)
		{
		super (exc);
		}

	/**
	 * Construct a new illegal data exception with the given detail message and
	 * the given chained exception.
	 *
	 * @param  msg  Detail message.
	 * @param  exc  Chained exception.
	 */
	public IllegalDataException
		(String msg,
		 Throwable exc)
		{
		super (msg, exc);
		}

	}
