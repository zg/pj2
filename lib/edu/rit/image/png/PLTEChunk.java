//******************************************************************************
//
// File:    PLTEChunk.java
// Package: edu.rit.image.png
// Unit:    Class edu.rit.image.png.PLTEChunk
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

import edu.rit.image.Color;
import edu.rit.util.AList;

/**
 * Class PLTEChunk provides a palette chunk stored in a PNG file.
 *
 * @author  Alan Kaminsky
 * @version 26-Jun-2013
 */
public class PLTEChunk
	extends Chunk
	{

// Exported constructors.

	/**
	 * Construct a new uninitialized PLTE chunk. This constructor is for use
	 * only by object deserialization.
	 */
	public PLTEChunk()
		{
		super();
		}

	/**
	 * Construct a new PLTE chunk. The palette is an {@linkplain
	 * edu.rit.util.AList Alist} of {@linkplain edu.rit.image.Color Color}s; it
	 * must contain at least 1 and at most 256 elements.
	 *
	 * @param  palette  Palette.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>palette</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if the number of elements in
	 *     <TT>palette</TT> is illegal.
	 */
	public PLTEChunk
		(AList<Color> palette)
		{
		super (0x504c5445, getData (palette));
		}

	private static byte[] getData
		(AList<Color> palette)
		{
		if (palette == null)
			throw new NullPointerException
				("PLTEChunk(): palette is null");
		int n = palette.size();
		if (1 > n || n > 256)
			throw new IllegalArgumentException (String.format
				("PLTEChunk(): palette size = %d illegal", n));
		byte[] data = new byte [3*n];
		for (int i = 0; i < n; ++ i)
			{
			Color color = palette.get (i);
			data[3*i  ] = (byte) color.red();
			data[3*i+1] = (byte) color.green();
			data[3*i+2] = (byte) color.blue();
			}
		return data;
		}

	}
