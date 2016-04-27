//******************************************************************************
//
// File:    Struct.java
// Package: edu.rit.gpu
// Unit:    Class edu.rit.gpu.Struct
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

import java.nio.ByteBuffer;

/**
 * Class Struct is the abstract base class for a Java object on the CPU that is
 * converted to or from a C struct on the GPU.
 *
 * @author  Alan Kaminsky
 * @version 03-Jun-2014
 */
public abstract class Struct
	{

// Exported constructors.

	/**
	 * Construct a new struct object.
	 */
	public Struct()
		{
		}

// Exported operations.

	/**
	 * Returns the size in bytes of the C struct. The size must include any
	 * internal padding bytes needed to align the fields of the C struct. The
	 * size must include any padding bytes at the end needed to align a series
	 * of C structs in an array.
	 * <P>
	 * The static <TT>sizeof()</TT> method must be overridden in a subclass.
	 *
	 * @return  Size of C struct (bytes).
	 */
	public static long sizeof()
		{
		throw new UnsupportedOperationException
			("Struct.sizeof(): Must be overridden in a subclass");
		}

	/**
	 * Write this Java object to the given byte buffer in the form of a C
	 * struct. The byte buffer's byte order is little endian. The byte buffer is
	 * positioned at the first byte of the C struct. The <TT>toStruct()</TT>
	 * method must write this object's fields into the byte buffer exactly as
	 * the C struct is laid out in GPU memory.
	 *
	 * @param  buf  Byte buffer to write.
	 */
	public abstract void toStruct
		(ByteBuffer buf);

	/**
	 * Read this Java object from the given byte buffer in the form of a C
	 * struct. The byte buffer's byte order is little endian. The byte buffer is
	 * positioned at the first byte of the C struct. The <TT>fromStruct()</TT>
	 * method must read this object's fields from the byte buffer exactly as the
	 * C struct is laid out in GPU memory.
	 *
	 * @param  buf  Byte buffer to read.
	 */
	public abstract void fromStruct
		(ByteBuffer buf);

	}
