//******************************************************************************
//
// File:    ChunkOutputStream.java
// Package: edu.rit.image.png
// Unit:    Class edu.rit.image.png.ChunkOutputStream
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

package edu.rit.image.png;

import java.io.Closeable;
import java.io.DataOutputStream;
import java.io.Flushable;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Class ChunkOutputStream provides an object for writing PNG image chunks to an
 * {@linkplain OutputStream}.
 *
 * @author  Alan Kaminsky
 * @version 22-May-2013
 */
public class ChunkOutputStream
	implements ChunkOutput, Closeable, Flushable
	{

// Hidden data members.

	private DataOutputStream out;

// Exported constructors.

	/**
	 * Construct a new chunk output stream that will write to the given
	 * underlying output stream.
	 *
	 * @param  out  Output stream.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>out</TT> is null.
	 */
	public ChunkOutputStream
		(OutputStream out)
		{
		this.out = new DataOutputStream (out);
		}

// Exported operations.

	/**
	 * Write the PNG file signature to this chunk output stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeSignature()
		throws IOException
		{
		out.writeByte (137);
		out.writeByte (80);
		out.writeByte (78);
		out.writeByte (71);
		out.writeByte (13);
		out.writeByte (10);
		out.writeByte (26);
		out.writeByte (10);
		}

	/**
	 * Write the given chunk to this chunk output stream.
	 *
	 * @param  chunk  Chunk.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void write
		(Chunk chunk)
		throws IOException
		{
		chunk.write (out);
		}

	/**
	 * Flush this chunk output stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void flush()
		throws IOException
		{
		out.flush();
		}

	/**
	 * Close this chunk output stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void close()
		throws IOException
		{
		out.close();
		}

	}
