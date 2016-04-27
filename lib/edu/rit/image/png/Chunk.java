//******************************************************************************
//
// File:    Chunk.java
// Package: edu.rit.image.png
// Unit:    Class edu.rit.image.png.Chunk
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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.zip.CRC32;

/**
 * Class Chunk is the abstract base class for a chunk stored in a PNG file.
 *
 * @author  Alan Kaminsky
 * @version 22-May-2013
 */
public abstract class Chunk
	implements Externalizable
	{

// Hidden data members.

	private static final long serialVersionUID = -9109990488880582815L;

	private int typeCode;
	private byte[] data;
	private int crc;

// Exported constructors.

	/**
	 * Construct a new uninitialized chunk. This constructor is for use only by
	 * object deserialization.
	 */
	public Chunk()
		{
		}

	/**
	 * Construct a new chunk with data taken from the given array.
	 *
	 * @param  typeCode  Chunk type code.
	 * @param  data      Array of chunk data.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>data</TT> is null.
	 */
	public Chunk
		(int typeCode,
		 byte[] data)
		{
		this (typeCode, data, 0, data.length);
		}

	/**
	 * Construct a new chunk with data taken from a portion of the given array.
	 *
	 * @param  typeCode  Chunk type code.
	 * @param  data      Array of chunk data.
	 * @param  off       Index of first byte of chunk data.
	 * @param  len       Number of bytes of chunk data.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>data</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>data.length</TT>.
	 */
	public Chunk
		(int typeCode,
		 byte[] data,
		 int off,
		 int len)
		{
		this.typeCode = typeCode;
		this.data = new byte [len];
		System.arraycopy (data, off, this.data, 0, len);
		CRC32 checksum = new CRC32();
		checksum.update (typeCode >> 24);
		checksum.update (typeCode >> 16);
		checksum.update (typeCode >> 8);
		checksum.update (typeCode);
		checksum.update (this.data);
		this.crc = (int)(checksum.getValue());
		}

	/**
	 * Write this chunk to the given data output stream.
	 *
	 * @param  out  Data output stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void write
		(DataOutput out)
		throws IOException
		{
		out.writeInt (data.length);
		out.writeInt (typeCode);
		out.write (data);
		out.writeInt (crc);
		}

	/**
	 * Read this chunk from the given data input stream.
	 *
	 * @param  in  Data input stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void read
		(DataInput in)
		throws IOException
		{
		data = new byte [in.readInt()];
		typeCode = in.readInt();
		in.readFully (data);
		crc = in.readInt();
		}

	/**
	 * Write this chunk to the given object output stream.
	 *
	 * @param  out  Object output stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeExternal
		(ObjectOutput out)
		throws IOException
		{
		write (out);
		}

	/**
	 * Read this chunk from the given object input stream.
	 *
	 * @param  in  Object input stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readExternal
		(ObjectInput in)
		throws IOException
		{
		read (in);
		}

	}
