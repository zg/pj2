//******************************************************************************
//
// File:    ByteImageQueue.java
// Package: edu.rit.image
// Unit:    Class edu.rit.image.ByteImageQueue
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
 * Class ByteImageQueue provides a queue of pixel rows for an image, with the
 * pixel data stored using type <TT>byte</TT>. A byte image queue is constructed
 * by the <TT>getImageQueue()</TT> method of class {@linkplain
 * BilevelPngWriter}, {@linkplain GrayPngWriter}, or {@linkplain
 * IndexPngWriter}.
 * <P>
 * The methods of class ByteImageQueue are multiple thread safe.
 *
 * @see  ImageQueue
 * @see  BilevelPngWriter
 * @see  GrayPngWriter
 * @see  IndexPngWriter
 *
 * @author  Alan Kaminsky
 * @version 29-Jun-2013
 */
public class ByteImageQueue
	extends ImageQueue
	{

// Hidden data members.

	byte[][] pixelData;

// Hidden constructors.

	/**
	 * Construct a new image queue for an image with the given dimensions.
	 *
	 * @param  rows  Number of rows. Assumed to be &ge; 1.
	 * @param  cols  Number of columns. Assumed to be &ge; 1.
	 */
	ByteImageQueue
		(int rows,
		 int cols)
		{
		super (rows, cols);
		pixelData = new byte [rows] [];
		}

// Exported operations.

	/**
	 * Put the given pixel data into the given row of this image queue. If this
	 * row already contains pixel data, the <TT>put()</TT> method blocks until
	 * this row's pixel data has been removed.
	 * <P>
	 * <I>Note:</I> A copy of the <TT>data</TT> array is stored in this image
	 * queue, so the <TT>data</TT> array can be reused for another row.
	 *
	 * @param  row   Row index in the range 0 .. {@link #rows() rows()}&minus;1.
	 * @param  data  Array of pixel data; its length must be equal to
	 *               {@link #cols() cols()}.
	 *
	 * @exception  ArrayIndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>row</TT> is out of bounds.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>data</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>data.length</TT> &ne; {@link
	 *     #cols() cols()}.
	 * @exception  InterruptedException
	 *     Thrown if the calling thread is interrupted while blocked in this
	 *     method.
	 */
	public void put
		(int row,
		 byte[] data)
		throws InterruptedException
		{
		if (data.length != cols)
			throw new IllegalArgumentException
				("ByteImageQueue.put(): data.length illegal");
		byte[] datacopy = (byte[]) data.clone();
		synchronized (this)
			{
			while (pixelData[row] != null) wait();
			pixelData[row] = datacopy;
			notifyAll();
			}
		}

	/**
	 * Remove the pixel data from the given row of this image queue. If this row
	 * does not contain pixel data, the <TT>remove()</TT> method blocks until
	 * pixel data has been put into this row.
	 *
	 * @param  row   Row index in the range 0 .. {@link #rows() rows()}&minus;1.
	 *
	 * @return  Array of pixel data.
	 *
	 * @exception  ArrayIndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>row</TT> is out of bounds.
	 * @exception  InterruptedException
	 *     Thrown if the calling thread is interrupted while blocked in this
	 *     method.
	 */
	public byte[] remove
		(int row)
		throws InterruptedException
		{
		synchronized (this)
			{
			while (pixelData[row] == null) wait();
			byte[] data = pixelData[row];
			pixelData[row] = null;
			notifyAll();
			return data;
			}
		}

	}
