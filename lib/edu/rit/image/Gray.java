//******************************************************************************
//
// File:    Gray.java
// Package: edu.rit.image
// Unit:    Class edu.rit.image.Gray
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
 * Class Gray provides a gray shade.
 *
 * @author  Alan Kaminsky
 * @version 27-Jun-2014
 */
public class Gray
	implements Streamable
	{

// Hidden data members.

	byte shade;

// Exported constructors.

	/**
	 * Construct a new gray object with a gray shade of black.
	 */
	public Gray()
		{
		}

	/**
	 * Construct a new gray object that is a copy of the given gray object.
	 *
	 * @param  gray  Gray object to copy.
	 */
	public Gray
		(Gray gray)
		{
		this.shade = gray.shade;
		}

// Exported operations.

	/**
	 * Get the gray shade of this gray object. The gray shade is an integer in
	 * the range 0 (black) through 255 (white).
	 *
	 * @return  Gray shade.
	 */
	public int gray()
		{
		return shade & 0xFF;
		}

	/**
	 * Set this gray object to the given gray shade. The gray shade is an
	 * integer in the range 0 (black) through 255 (white). Only bits 7-0 are
	 * used.
	 *
	 * @param  shade  Gray shade.
	 *
	 * @return  This gray object.
	 */
	public Gray gray
		(int shade)
		{
		return gray (shade, true);
		}

	/**
	 * Set this gray object to the given gray shade. The gray shade is an
	 * integer in the range 0 through 255. Only bits 7-0 are used. If
	 * <TT>zeroIsBlack</TT> is true, then 0 = black and 255 = white. If
	 * <TT>zeroIsBlack</TT> is false, then 0 = white and 255 = black.
	 *
	 * @param  shade        Gray shade.
	 * @param  zeroIsBlack  True if 0 = black, false if 0 = white.
	 *
	 * @return  This gray object.
	 */
	public Gray gray
		(int shade,
		 boolean zeroIsBlack)
		{
		this.shade = intToShade (shade, zeroIsBlack);
		return this;
		}

	/**
	 * Set this gray object to the given gray shade. The gray shade is a
	 * floating point number in the range 0.0f (black) through 1.0f (white).
	 * Values outside that range are pinned to that range.
	 *
	 * @param  shade  Gray shade.
	 *
	 * @return  This gray object.
	 */
	public Gray gray
		(float shade)
		{
		return gray (shade, true);
		}

	/**
	 * Set this gray object to the given gray shade. The gray shade is a
	 * floating point number in the range 0.0f through 1.0f. Values outside that
	 * range are pinned to that range. If <TT>zeroIsBlack</TT> is true, then
	 * 0.0f = black and 1.0f = white. If <TT>zeroIsBlack</TT> is false, then
	 * 0.0f = white and 1.0f = black.
	 *
	 * @param  shade        Gray shade.
	 * @param  zeroIsBlack  True if 0.0f = black, false if 0.0f = white.
	 *
	 * @return  This gray object.
	 */
	public Gray gray
		(float shade,
		 boolean zeroIsBlack)
		{
		this.shade = floatToShade (shade, zeroIsBlack);
		return this;
		}

	/**
	 * Write this gray object to the given out stream.
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
		out.writeByte (shade);
		}

	/**
	 * Read this gray object from the given in stream.
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
		shade = in.readByte();
		}

	/**
	 * Returns a string version of this gray object. The string is in
	 * hexadecimal HTML format; for example, <TT>"#121212"</TT>.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		return String.format ("#%02X%02X%02X", shade, shade, shade);
		}

// Hidden operations.

	/**
	 * Convert the given <TT>int</TT> to a gray shade.
	 */
	static byte intToShade
		(int shade,
		 boolean zeroIsBlack)
		{
		return (byte)(zeroIsBlack ? shade & 255 : 255 - (shade & 255));
		}

	/**
	 * Convert the given <TT>float</TT> to a gray shade.
	 */
	static byte floatToShade
		(float shade,
		 boolean zeroIsBlack)
		{
		shade = Math.max (0.0f, Math.min (shade, 1.0f));
		int g = (int)(256.0f*(zeroIsBlack ? shade : 1.0f - shade));
		return (byte) Math.min (g, 255);
		}

	}
