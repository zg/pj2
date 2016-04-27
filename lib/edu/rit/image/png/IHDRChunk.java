//******************************************************************************
//
// File:    IHDRChunk.java
// Package: edu.rit.image.png
// Unit:    Class edu.rit.image.png.IHDRChunk
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
 * Class IHDRChunk provides an image header chunk stored in a PNG file.
 *
 * @author  Alan Kaminsky
 * @version 22-May-2013
 */
public class IHDRChunk
	extends Chunk
	{

// Exported constructors.

	/**
	 * Construct a new uninitialized IHDR chunk. This constructor is for use
	 * only by object deserialization.
	 */
	public IHDRChunk()
		{
		super();
		}

	/**
	 * Construct a new IHDR chunk. The following fields are fixed: compression
	 * method = 0 (deflate), filter method = 0 (adaptive filtering), interlace
	 * method = 0 (no interlacing).
	 *
	 * @param  width      Image width in pixels.
	 * @param  height     Image height in pixels.
	 * @param  bitDepth   Sample bit depth.
	 * @param  colorType  Color type.
	 */
	public IHDRChunk
		(int width,
		 int height,
		 int bitDepth,
		 int colorType)
		{
		super (0x49484452, getData (width, height, bitDepth, colorType));
		}

	private static byte[] getData
		(int width,
		 int height,
		 int bitDepth,
		 int colorType)
		{
		byte[] data = new byte [13];
		Packing.unpackIntBigEndian (width, data, 0);  // Width
		Packing.unpackIntBigEndian (height, data, 4); // Height
		data[8] = (byte) bitDepth;  // Bit depth
		data[9] = (byte) colorType; // Color type
		data[10] = (byte) 0;        // Compression method
		data[11] = (byte) 0;        // Filter method
		data[12] = (byte) 0;        // Interlace method
		return data;
		}

	}
