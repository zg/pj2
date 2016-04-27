//******************************************************************************
//
// File:    GrayArray.java
// Package: edu.rit.image
// Unit:    Class edu.rit.image.GrayArray
//
// This Java source file is copyright (C) 2014 by Alan Kaminsky. All rights
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

package edu.rit.image;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import java.io.IOException;

/**
 * Class GrayArray provides an array of gray shades.
 * <P>
 * <I>Note:</I> An instance of class GrayArray uses less storage than an array
 * of instances of class {@linkplain Gray Gray}.
 *
 * @author  Alan Kaminsky
 * @version 28-Jun-2014
 */
public class GrayArray
	implements Streamable
	{

// Hidden data members.

	byte[] shade;

// Exported constructors.

	/**
	 * Construct a new zero-length gray array.
	 */
	public GrayArray()
		{
		this (0);
		}

	/**
	 * Construct a new gray array of the given length. Each element is a gray
	 * shade of black.
	 *
	 * @param  len  Array length.
	 */
	public GrayArray
		(int len)
		{
		shade = new byte [len];
		}

	/**
	 * Construct a new gray array that is a copy of the given gray array.
	 *
	 * @param  array  Gray array to copy.
	 */
	public GrayArray
		(GrayArray array)
		{
		this.shade = (byte[]) array.shade.clone();
		}

// Exported operations.

	/**
	 * Get this gray array's length.
	 *
	 * @return  Array length.
	 */
	public int length()
		{
		return shade.length;
		}

	/**
	 * Get the gray shade of the given element of this gray array. The gray
	 * shade is an integer in the range 0 (black) through 255 (white).
	 *
	 * @param  i  Array index.
	 *
	 * @return  Gray shade.
	 */
	public int gray
		(int i)
		{
		return shade[i] & 0xFF;
		}

	/**
	 * Set the given element of this gray array to the given gray shade. The
	 * gray shade is an integer in the range 0 (black) through 255 (white). Only
	 * bits 7-0 are used.
	 *
	 * @param  i      Array index.
	 * @param  shade  Gray shade.
	 *
	 * @return  This gray array.
	 */
	public GrayArray gray
		(int i,
		 int shade)
		{
		return gray (i, shade, true);
		}

	/**
	 * Set the given element of this gray array to the given gray shade. The
	 * gray shade is an integer in the range 0 through 255. Only bits 7-0 are
	 * used. If <TT>zeroIsBlack</TT> is true, then 0 = black and 255 = white. If
	 * <TT>zeroIsBlack</TT> is false, then 0 = white and 255 = black.
	 *
	 * @param  i            Array index.
	 * @param  shade        Gray shade.
	 * @param  zeroIsBlack  True if 0 = black, false if 0 = white.
	 *
	 * @return  This gray array.
	 */
	public GrayArray gray
		(int i,
		 int shade,
		 boolean zeroIsBlack)
		{
		this.shade[i] = Gray.intToShade (shade, zeroIsBlack);
		return this;
		}

	/**
	 * Set the given element of this gray array to the given gray shade. The
	 * gray shade is a floating point number in the range 0.0f (black) through
	 * 1.0f (white). Values outside that range are pinned to that range.
	 *
	 * @param  i      Array index.
	 * @param  shade  Gray shade.
	 *
	 * @return  This gray array.
	 */
	public GrayArray gray
		(int i,
		 float shade)
		{
		return gray (i, shade, true);
		}

	/**
	 * Set the given element of this gray array to the given gray shade. The
	 * gray shade is a floating point number in the range 0.0f through 1.0f.
	 * Values outside that range are pinned to that range. If
	 * <TT>zeroIsBlack</TT> is true, then 0.0f = black and 1.0f = white. If
	 * <TT>zeroIsBlack</TT> is false, then 0.0f = white and 1.0f = black.
	 *
	 * @param  i            Array index.
	 * @param  shade        Gray shade.
	 * @param  zeroIsBlack  True if 0.0f = black, false if 0.0f = white.
	 *
	 * @return  This gray array.
	 */
	public GrayArray gray
		(int i,
		 float shade,
		 boolean zeroIsBlack)
		{
		this.shade[i] = Gray.floatToShade (shade, zeroIsBlack);
		return this;
		}

	/**
	 * Copy a portion of the given gray array into this gray array.
	 *
	 * @param  src     Source gray array.
	 * @param  srcoff  First index to read in source gray array.
	 * @param  dstoff  First index to write in this gray array.
	 * @param  len     Number of elements to copy.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if a nonexistent index in the source
	 *     gray array would be read or a nonexistent index in this gray array
	 *     would be written.
	 */
	public void copy
		(GrayArray src,
		 int srcoff,
		 int dstoff,
		 int len)
		{
		System.arraycopy (src.shade, srcoff, this.shade, dstoff, len);
		}

	/**
	 * Write this gray array to the given out stream.
	 *
	 * @param  out  Out stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeOut
		(OutStream out)
		throws IOException
		{
		out.writeByteArray (shade);
		}

	/**
	 * Read this gray array from the given in stream.
	 *
	 * @param  in  In stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readIn
		(InStream in)
		throws IOException
		{
		shade = in.readByteArray();
		}

	}
