//******************************************************************************
//
// File:    fcTLChunk.java
// Package: edu.rit.image.png
// Unit:    Class edu.rit.image.png.fcTLChunk
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
 * Class fcTLChunk provides a frame control chunk stored in an APNG file.
 *
 * @author  Alan Kaminsky
 * @version 28-Jun-2013
 */
public class fcTLChunk
	extends Chunk
	{

// Exported constants.

	/**
	 * Frame area dispose operation: None.
	 */
	public static final byte APNG_DISPOSE_OP_NONE = (byte) 0;

	/**
	 * Frame area dispose operation: Clear output buffer to fully transparent
	 * black.
	 */
	public static final byte APNG_DISPOSE_OP_BACKGROUND = (byte) 1;

	/**
	 * Frame area dispose operation: Revert output buffer to previous contents.
	 */
	public static final byte APNG_DISPOSE_OP_PREVIOUS = (byte) 2;

	/**
	 * Frame area blend operation: Frame overwrites output buffer.
	 */
	public static final byte APNG_BLEND_OP_SOURCE = (byte) 0;

	/**
	 * Frame area blend operation: Frame composites onto output buffer.
	 */
	public static final byte APNG_BLEND_OP_OVER = (byte) 1;

// Exported constructors.

	/**
	 * Construct a new uninitialized fcTL chunk. This constructor is for use
	 * only by object deserialization.
	 */
	public fcTLChunk()
		{
		super();
		}

	/**
	 * Construct a new fcTL chunk.
	 *
	 * @param  seqnum      Chunk sequence number.
	 * @param  width       Width of the frame.
	 * @param  height      Height of the frame.
	 * @param  xOffset     X position of the frame.
	 * @param  yOffset     Y position of the frame.
	 * @param  delayNumer  Frame delay numerator.
	 * @param  delayDenom  Frame delay denominator.
	 * @param  disposeOp   Frame area dispose operation.
	 * @param  blendOp     Frame area blend operation.
	 */
	public fcTLChunk
		(int seqnum,
		 int width,
		 int height,
		 int xOffset,
		 int yOffset,
		 short delayNumer,
		 short delayDenom,
		 byte disposeOp,
		 byte blendOp)
		{
		super (0x6663544c, getData (seqnum, width, height, xOffset, yOffset,
			delayNumer, delayDenom, disposeOp, blendOp));
		}

	private static byte[] getData
		(int seqnum,
		 int width,
		 int height,
		 int xOffset,
		 int yOffset,
		 short delayNumer,
		 short delayDenom,
		 byte disposeOp,
		 byte blendOp)
		{
		byte[] data = new byte [26];
		Packing.unpackIntBigEndian (seqnum, data, 0);
		Packing.unpackIntBigEndian (width, data, 4);
		Packing.unpackIntBigEndian (height, data, 8);
		Packing.unpackIntBigEndian (xOffset, data, 12);
		Packing.unpackIntBigEndian (yOffset, data, 16);
		Packing.unpackShortBigEndian (delayNumer, data, 20);
		Packing.unpackShortBigEndian (delayDenom, data, 22);
		data[24] = disposeOp;
		data[25] = blendOp;
		return data;
		}

	}
