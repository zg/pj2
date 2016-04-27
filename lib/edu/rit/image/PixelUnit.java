//******************************************************************************
//
// File:    PixelUnit.java
// Package: edu.rit.image
// Unit:    Class edu.rit.image.PixelUnit
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

package edu.rit.image;

/**
 * Enum PixelUnit enumerates the possible pixel dimension units in an image.
 *
 * @author  Alan Kaminsky
 * @version 19-Jul-2013
 */
public enum PixelUnit
	{

	/**
	 * Units of pixels per inch.
	 */
	PIXELS_PER_INCH (1.0/0.0254),

	/**
	 * Units of pixels per centimeter.
	 */
	PIXELS_PER_CENTIMETER (100.0),

	/**
	 * Units of pixels per meter.
	 */
	PIXELS_PER_METER (1.0),

	/**
	 * No pixel dimension units.
	 */
	NONE (1.0);

// Hidden data members.

	private double toPpmFactor;

// Hidden constructors.

	private PixelUnit
		(double toPpmFactor)
		{
		this.toPpmFactor = toPpmFactor;
		}

// Exported operations.

	/**
	 * Convert the given pixel dimension from this pixel units to pixels per
	 * meter.
	 *
	 * @param  dim  Pixel dimension in this pixel units.
	 *
	 * @return  Pixel dimension in pixels per meter.
	 */
	public int toPpm
		(int dim)
		{
		return (int)(dim*toPpmFactor + 0.5);
		}

	}
