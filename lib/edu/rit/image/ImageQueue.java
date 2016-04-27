//******************************************************************************
//
// File:    ImageQueue.java
// Package: edu.rit.image
// Unit:    Class edu.rit.image.ImageQueue
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
 * Class ImageQueue is the abstract base class for a queue of pixel rows for an
 * image. An image queue is constructed by the <TT>getImageQueue()</TT> method
 * of class {@linkplain BilevelPngWriter}, {@linkplain GrayPngWriter},
 * {@linkplain IndexPngWriter}, or {@linkplain ColorPngWriter}.
 *
 * @see  ByteImageQueue
 * @see  ColorImageQueue
 * @see  BilevelPngWriter
 * @see  GrayPngWriter
 * @see  IndexPngWriter
 * @see  ColorPngWriter
 * @see  ColorApngWriter
 *
 * @author  Alan Kaminsky
 * @version 29-Jun-2013
 */
public abstract class ImageQueue
	{

// Hidden data members.

	// Image size.
	int rows;
	int cols;

// Hidden constructors.

	/**
	 * Construct a new image queue for an image with the given dimensions.
	 *
	 * @param  rows  Number of rows. Assumed to be &ge; 1.
	 * @param  cols  Number of columns. Assumed to be &ge; 1.
	 */
	ImageQueue
		(int rows,
		 int cols)
		{
		this.rows = rows;
		this.cols = cols;
		}

// Exported operations.

	/**
	 * Returns the number of rows in this image queue's image.
	 *
	 * @return  Number of rows.
	 */
	public int rows()
		{
		return rows;
		}

	/**
	 * Returns the number of columns in this image queue's image.
	 *
	 * @return  Number of columns.
	 */
	public int cols()
		{
		return cols;
		}

	}
