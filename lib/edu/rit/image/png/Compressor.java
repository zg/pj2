//******************************************************************************
//
// File:    Compressor.java
// Package: edu.rit.image.png
// Unit:    Class edu.rit.image.png.Compressor
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

import java.io.IOException;
import java.util.zip.Deflater;

/**
 * Class Compressor is the abstract base class for an object that compresses a
 * sequence of bytes into PNG chunks and writes the chunks to a {@linkplain
 * ChunkOutput} object.
 *
 * @author  Alan Kaminsky
 * @version 01-Jul-2013
 */
public abstract class Compressor
	{

// Hidden data members.

	private ChunkOutput chunkOutput;
	private Deflater deflater = new Deflater();
	private byte[] chunkData = new byte [32768];
	private int chunkLen = 0;
	private int seqnum;

// Exported constructors.

	/**
	 * Construct a new compressor that will write to the given underlying
	 * chunk output object.
	 *
	 * @param  chunkOutput  Chunk output object.
	 * @param  seqnum       Initial chunk sequence number.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>chunkOutput</TT> is null.
	 */
	public Compressor
		(ChunkOutput chunkOutput,
		 int seqnum)
		{
		if (chunkOutput == null)
			throw new NullPointerException
				("Compressor(): chunkOutput is null");
		this.chunkOutput = chunkOutput;
		this.seqnum = seqnum;
		}

// Exported operations.

	/**
	 * Write the given byte to this compressor.
	 *
	 * @param  b       Byte. Only the least significant 8 bits are used.
	 * @param  finish  True if this is the last data to be compressed, false
	 *                 otherwise.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void write
		(int b,
		 boolean finish)
		throws IOException
		{
		write (new byte[] { (byte)b }, finish);
		}

	/**
	 * Write the given byte array to this compressor.
	 *
	 * @param  b       Byte array.
	 * @param  finish  True if this is the last data to be compressed, false
	 *                 otherwise.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void write
		(byte[] b,
		 boolean finish)
		throws IOException
		{
		int len;
		deflater.setInput (b);
		if (finish) deflater.finish();
		while (! deflater.needsInput())
			{
			do
				{
				len = deflater.deflate
					(chunkData, chunkLen, chunkData.length - chunkLen);
				chunkLen += len;
				if (chunkLen == chunkData.length)
					{
					chunkOutput.write
						(getChunk (seqnum, chunkData, 0, chunkLen));
					seqnum = incrementSeqnum (seqnum);
					chunkLen = 0;
					}
				}
			while (len != 0);
			}
		if (deflater.finished())
			{
			chunkOutput.write (getChunk (seqnum, chunkData, 0, chunkLen));
			seqnum = incrementSeqnum (seqnum);
			deflater.end();
			}
		}

	/**
	 * Returns the next chunk sequence number to be used.
	 *
	 * @return  Next chunk sequence number.
	 */
	public int nextSequenceNumber()
		{
		return seqnum;
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
	abstract Chunk getChunk
		(int seqnum,
		 byte[] data,
		 int off,
		 int len);

	/**
	 * Increment the given sequence number, if required by this compressor's
	 * chunk type.
	 *
	 * @param  seqnum  Chunk sequence number.
	 *
	 * @return  Chunk sequence number after incrementing.
	 */
	abstract int incrementSeqnum
		(int seqnum);

	}
