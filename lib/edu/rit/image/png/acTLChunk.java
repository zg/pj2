//******************************************************************************
//
// File:    acTLChunk.java
// Package: edu.rit.image.png
// Unit:    Class edu.rit.image.png.acTLChunk
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
 * Class acTLChunk provides an animation control chunk stored in an APNG file.
 *
 * @author  Alan Kaminsky
 * @version 28-Jun-2013
 */
public class acTLChunk
	extends Chunk
	{

// Exported constructors.

	/**
	 * Construct a new uninitialized acTL chunk. This constructor is for use
	 * only by object deserialization.
	 */
	public acTLChunk()
		{
		super();
		}

	/**
	 * Construct a new acTL chunk.
	 *
	 * @param  numFrames  Number of frames.
	 * @param  numPlays   Number of plays. 0 = infinite looping.
	 */
	public acTLChunk
		(int numFrames,
		 int numPlays)
		{
		super (0x6163544c, getData (numFrames, numPlays));
		}

	private static byte[] getData
		(int numFrames,
		 int numPlays)
		{
		byte[] data = new byte [8];
		Packing.unpackIntBigEndian (numFrames, data, 0);
		Packing.unpackIntBigEndian (numPlays, data, 4);
		return data;
		}

	}
