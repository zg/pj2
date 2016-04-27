//******************************************************************************
//
// File:    fdATCompressor.java
// Package: edu.rit.image.png
// Unit:    Class edu.rit.image.png.fdATCompressor
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

/**
 * Class fdATCompressor provides an object that compresses a sequence of bytes
 * into APNG fdAT chunks and writes the chunks to a {@linkplain ChunkOutput}
 * object.
 *
 * @author  Alan Kaminsky
 * @version 01-Jul-2013
 */
public class fdATCompressor
	extends Compressor
	{

// Exported constructors.

	/**
	 * Construct a new fdAT compressor that will write to the given underlying
	 * chunk output object.
	 *
	 * @param  chunkOutput  Chunk output object.
	 * @param  seqnum       Initial chunk sequence number.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>chunkOutput</TT> is null.
	 */
	public fdATCompressor
		(ChunkOutput chunkOutput,
		 int seqnum)
		{
		super (chunkOutput, seqnum);
		}

// Hidden operations.

	/**
	 * Get a PNG chunk of the proper type containing the given compressed data.
	 *
	 * @param  seqnum  Chunk sequence number.
	 * @param  data    Compressed data buffer.
	 * @param  off     Index of first data byte.
	 * @param  len     Number of data bytes.
	 *
	 * @return  Chunk.
	 */
	Chunk getChunk
		(int seqnum,
		 byte[] data,
		 int off,
		 int len)
		{
		return new fdATChunk (seqnum, data, off, len);
		}

	/**
	 * Increment the given sequence number, if required by this compressor's
	 * chunk type.
	 *
	 * @param  seqnum  Chunk sequence number.
	 *
	 * @return  Chunk sequence number after incrementing.
	 */
	int incrementSeqnum
		(int seqnum)
		{
		return seqnum + 1;
		}

	}
