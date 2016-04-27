//******************************************************************************
//
// File:    ContextObjectInputStream.java
// Package: edu.rit.pj2.io
// Unit:    Class edu.rit.pj2.io.ContextObjectInputStream
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
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectStreamClass;

/**
 * Class ContextObjectInputStream is an {@linkplain java.io.ObjectInputStream
 * ObjectInputStream} that uses the calling thread's context class loader to
 * load classes. (The regular ObjectInputStream does not do this.)
 *
 * @author  Alan Kaminsky
 * @version 18-Jun-2013
 */
public class ContextObjectInputStream
	extends ObjectInputStream
	{

// Exported constructors.

	/**
	 * Construct a new context object input stream that will read from the given
	 * stream.
	 *
	 * @param  in  Underlying input stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public ContextObjectInputStream
		(InputStream in)
		throws IOException
		{
		super (in);
		}

// Hidden operations.

	/**
	 * Resolve the given class description using the calling thread's context
	 * class loader.
	 *
	 * @param  desc  Class description.
	 *
	 * @return  Class.
	 *
	 * @exception  ClassNotFoundException
	 *     Thrown if the class could not be found.
	 */
	protected Class<?> resolveClass
		(ObjectStreamClass desc)
		throws ClassNotFoundException
		{
		return Class.forName (desc.getName(), false,
			Thread.currentThread().getContextClassLoader());
		}

	}
