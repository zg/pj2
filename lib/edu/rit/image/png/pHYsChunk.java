//******************************************************************************
//
// File:    pHYsChunk.java
// Package: edu.rit.image.png
// Unit:    Class edu.rit.image.png.pHYsChunk
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

import edu.rit.image.PixelUnit;
import edu.rit.util.Packing;

/**
 * Class pHYsChunk provides a physical pixel dimensions chunk stored in a PNG
 * file.
 *
 * @author  Alan Kaminsky
 * @version 19-Jul-2013
 */
public class pHYsChunk
	extends Chunk
	{

// Exported constructors.

	/**
	 * Construct a new uninitialized pHYs chunk. This constructor is for use
	 * only by object deserialization.
	 */
	public pHYsChunk()
		{
		super();
		}

	/**
	 * Construct a new pHYs chunk.
	 *
	 * @param  xPpu  Pixels per unit, X axis.
	 * @param  yPpu  Pixels per unit, Y axis.
	 * @param  unit  Pixel units.
	 */
	public pHYsChunk
		(int xPpu,
		 int yPpu,
		 PixelUnit unit)
		{
		super (0x70485973, getData (xPpu, yPpu, unit));
		}

	private static byte[] getData
		(int xPpu,
		 int yPpu,
		 PixelUnit unit)
		{
		byte[] data = new byte [9];
		if (unit == PixelUnit.NONE)
			{
			Packing.unpackIntBigEndian (xPpu, data, 0);
			Packing.unpackIntBigEndian (yPpu, data, 4);
			data[8] = (byte) 0;
			}
		else
			{
			Packing.unpackIntBigEndian (unit.toPpm (xPpu), data, 0);
			Packing.unpackIntBigEndian (unit.toPpm (yPpu), data, 4);
			data[8] = (byte) 1;
			}
		return data;
		}

	}
