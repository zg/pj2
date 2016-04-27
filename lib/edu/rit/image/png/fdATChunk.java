//******************************************************************************
//
// File:    fdATChunk.java
// Package: edu.rit.image.png
// Unit:    Class edu.rit.image.png.fdATChunk
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

import edu.rit.util.Packing;

/**
 * Class fdATChunk provides a frame data chunk stored in an APNG file.
 *
 * @author  Alan Kaminsky
 * @version 28-Jun-2013
 */
public class fdATChunk
	extends Chunk
	{

// Exported constructors.

	/**
	 * Construct a new uninitialized fdAT chunk. This constructor is for use
	 * only by object deserialization.
	 */
	public fdATChunk()
		{
		super();
		}

	/**
	 * Construct a new fdAT chunk with data taken from the given array.
	 *
	 * @param  seqnum  Chunk sequence number.
	 * @param  data    Array of chunk data.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>data</TT> is null.
	 */
	public fdATChunk
		(int seqnum,
		 byte[] data)
		{
		this (seqnum, data, 0, data.length);
		}

	/**
	 * Construct a new IDAT chunk with data taken from a portion of the given
	 * array.
	 *
	 * @param  seqnum  Chunk sequence number.
	 * @param  data    Array of chunk data.
	 * @param  off     Index of first byte of chunk data.
	 * @param  len     Number of bytes of chunk data.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>data</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>data.length</TT>.
	 */
	public fdATChunk
		(int seqnum,
		 byte[] data,
		 int off,
		 int len)
		{
		super (0x66644154, getData (seqnum, data, off, len));
		}

	private static byte[] getData
		(int seqnum,
		 byte[] data,
		 int off,
		 int len)
		{
		byte[] moredata = new byte [4 + len];
		Packing.unpackIntBigEndian (seqnum, moredata, 0);
		System.arraycopy (data, off, moredata, 4, len);
		return moredata;
		}

	}
