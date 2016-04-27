//******************************************************************************
//
// File:    Streamable.java
// Package: edu.rit.io
// Unit:    Interface edu.rit.io.Streamable
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
 * Interface Streamable specifies the interface for an object that can be
 * written to an output stream by class {@linkplain OutStream} or read from an
 * input stream by class {@linkplain InStream}.
 * <P>
 * A streamable object's class must have a no-argument constructor. When
 * reading a streamable object from an {@linkplain InStream}, an instance is
 * created using the no-argument constructor, then the object's fields are read
 * from the {@linkplain InStream} using the {@link #readIn(InStream) readIn()}
 * method. However, the no-argument constructor need not be public. Also, the
 * class itself need not be public. For further information, see the {@link
 * InStream#readObject() Instream.readObject()} method.
 *
 * @author  Alan Kaminsky
 * @version 05-Dec-2013
 */
public interface Streamable
	{

// Exported operations.

	/**
	 * Write this object's fields to the given out stream.
	 *
	 * @param  out  Out stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeOut
		(OutStream out)
		throws IOException;

	/**
	 * Read this object's fields from the given in stream.
	 *
	 * @param  in  In stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readIn
		(InStream in)
		throws IOException;

	}
