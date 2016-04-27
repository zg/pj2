//******************************************************************************
//
// File:    ColorPngWriter.java
// Package: edu.rit.image
// Unit:    Class edu.rit.image.ColorPngWriter
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

import edu.rit.image.png.ChunkOutputStream;
import edu.rit.image.png.IDATCompressor;
import edu.rit.image.png.IENDChunk;
import edu.rit.image.png.IHDRChunk;
import edu.rit.image.png.pHYsChunk;
import java.io.IOException;
import java.io.OutputStream;

// For unit test main program:
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;

/**
 * Class ColorPngWriter provides an object for writing a 24-bit color PNG image.
 * To write an image:
 * <OL TYPE=1>
 * <P><LI>
 * Create an instance of class ColorPngWriter, specifying the image dimensions
 * and the output stream to which to write the image.
 * <P><LI>
 * Optionally, call the writer's {@link #setPixelDimensions(int,int,PixelUnit)
 * setPixelDimensions()} method to set the pixels' physical size. If not
 * specified, the default is 300&times;300 pixels per inch.
 * <P><LI>
 * Call the writer's {@link #getImageQueue() getImageQueue()} method to get the
 * writer's image queue, which is an instance of class {@linkplain
 * ColorImageQueue}.
 * <P><LI>
 * Generate rows of pixel data and put them into the image queue by calling the
 * image queue's {@link ColorImageQueue#put(int,ColorArray) put()} method. Each
 * row of pixel data must be stored in a {@linkplain ColorArray} whose length is
 * equal to the number of columns in the image. Each element of the array
 * specifies the color for one pixel. You must call the image queue's {@link
 * ColorImageQueue#put(int,ColorArray) put()} method once for every row of
 * pixels in the image. You do not have to put the rows in any particular order.
 * <P><LI>
 * Call the writer's {@link #write() write()} method to write the pixel data to
 * the PNG image.
 * </OL>
 * <P>
 * Steps 4 and 5 can be performed concurrently in separate threads. In Step 4,
 * multiple threads can put the pixel data rows into the image queue
 * concurrently.
 *
 * @author  Alan Kaminsky
 * @version 19-Jul-2013
 */
public class ColorPngWriter
	{

// Hidden data members.

	private ChunkOutputStream chunkOutputStream;
	private IDATCompressor compressor;
	private ColorImageQueue imageQueue;
	private pHYsChunk physChunk =
		new pHYsChunk (300, 300, PixelUnit.PIXELS_PER_INCH);

// Exported constructors.

	/**
	 * Construct a new color PNG writer. The PNG image consists of the given
	 * number of rows and columns. The PNG image will be written to the given
	 * output stream.
	 *
	 * @param  rows  Number of rows. Must be &ge; 1.
	 * @param  cols  Number of columns. Must be &ge; 1.
	 * @param  out   Output stream.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>rows</TT> &lt; 1 or <TT>cols</TT>
	 *     &lt; 1.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>out</TT> is null.
	 */
	public ColorPngWriter
		(int rows,
		 int cols,
		 OutputStream out)
		{
		if (rows < 1)
			throw new IllegalArgumentException (String.format
				("ColorPngWriter(): rows = %d illegal", rows));
		if (cols < 1)
			throw new IllegalArgumentException (String.format
				("ColorPngWriter(): cols = %d illegal", cols));
		chunkOutputStream = new ChunkOutputStream (out);
		compressor = new IDATCompressor (chunkOutputStream, 0);
		imageQueue = new ColorImageQueue (rows, cols);
		}

// Exported operations.

	/**
	 * Set this color PNG writer's physical pixel dimensions to the given
	 * values. If not specified, the default is 300&times;300 pixels per inch.
	 *
	 * @param  xPpu  Pixels per unit, X axis.
	 * @param  yPpu  Pixels per unit, Y axis.
	 * @param  unit  Pixel dimension units.
	 */
	public void setPixelDimensions
		(int xPpu,
		 int yPpu,
		 PixelUnit unit)
		{
		physChunk = new pHYsChunk (xPpu, yPpu, unit);
		}

	/**
	 * Returns this color PNG writer's image queue.
	 *
	 * @return  Image queue.
	 */
	public ColorImageQueue getImageQueue()
		{
		return imageQueue;
		}

	/**
	 * Write this color PNG writer's image to the underlying output stream. The
	 * pixel data is obtained from the image queue. The underlying output stream
	 * is closed once the image is written.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 * @exception  InterruptedException
	 *     Thrown if the calling thread is interrupted while blocked in this
	 *     method.
	 */
	public void write()
		throws IOException, InterruptedException
		{
		int rows = imageQueue.rows();
		int cols = imageQueue.cols();
		int[] pixelData;
		byte[] data = new byte [3*cols];
		byte prev_r, prev_g, prev_b;
		chunkOutputStream.writeSignature();
		chunkOutputStream.write
			(new IHDRChunk
				(/*width    */ cols,
				 /*height   */ rows,
				 /*bitDepth */ 8,
				 /*colorType*/ 2));
		chunkOutputStream.write (physChunk);
		for (int r = 0; r < rows; ++ r)
			{
			pixelData = imageQueue.remove (r) .rgb;
			prev_r = data[0] = (byte)(pixelData[0] >> 16);
			prev_g = data[1] = (byte)(pixelData[0] >>  8);
			prev_b = data[2] = (byte)(pixelData[0]      );
			for (int c = 1; c < cols; ++ c)
				{
				data[3*c  ] = (byte)((pixelData[c] >> 16) - prev_r);
				data[3*c+1] = (byte)((pixelData[c] >>  8) - prev_g);
				data[3*c+2] = (byte)((pixelData[c]      ) - prev_b);
				prev_r = (byte)(data[3*c  ] + prev_r);
				prev_g = (byte)(data[3*c+1] + prev_g);
				prev_b = (byte)(data[3*c+2] + prev_b);
				}
			compressor.write (1, false); // Filter method = 1 = sub
			compressor.write (data, r == rows - 1);
			}
		chunkOutputStream.write (new IENDChunk());
		chunkOutputStream.close();
		}

// Unit test main program.

//	/**
//	 * Unit test main program.
//	 */
//	public static void main
//		(String[] args)
//		throws Exception
//		{
//		if (args.length != 1) usage();
//		File file = new File (args[0]);
//		final ColorPngWriter writer = new ColorPngWriter (100, 200,
//			new BufferedOutputStream (new FileOutputStream (file)));
//		ColorImageQueue imageQueue = writer.getImageQueue();
//		new Thread()
//			{
//			public void run()
//				{
//				try
//					{
//					writer.write();
//					}
//				catch (Throwable exc)
//					{
//					exc.printStackTrace (System.err);
//					System.exit (1);
//					}
//				}
//			}
//			.start();
//		ColorArray data = new ColorArray (200);
//		for (int c = 0; c < 200; ++ c)
//			{
//			data.hsb (c, c/200.0f, 1.0f, 1.0f);
//			System.out.printf ("[%d]\t%s%n", c, data.color (c));
//			}
//		for (int r = 0; r < 100; ++ r)
//			imageQueue.put (r, data);
//		}
//
//	private static void usage()
//		{
//		System.err.println ("Usage: java edu.rit.image.ColorPngWriter <file>");
//		System.exit (1);
//		}

	}
