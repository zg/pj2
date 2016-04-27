//******************************************************************************
//
// File:    IndexPngWriter.java
// Package: edu.rit.image
// Unit:    Class edu.rit.image.IndexPngWriter
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
import edu.rit.image.png.PLTEChunk;
import edu.rit.util.AList;
import java.io.IOException;
import java.io.OutputStream;

// For unit test main program:
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;

/**
 * Class IndexPngWriter provides an object for writing an indexed color PNG
 * image. To write an image:
 * <OL TYPE=1>
 * <P><LI>
 * Create a palette of colors for the image. The palette is an {@linkplain
 * edu.rit.util.AList AList} of {@linkplain Color Color}s; it must contain at
 * least 1 and at most 256 elements.
 * <P><LI>
 * Create an instance of class IndexPngWriter, specifying the image dimensions,
 * the output stream to which to write the image, and the palette.
 * <P><LI>
 * Optionally, call the writer's {@link #setPixelDimensions(int,int,PixelUnit)
 * setPixelDimensions()} method to set the pixels' physical size. If not
 * specified, the default is 300&times;300 pixels per inch.
 * <P><LI>
 * Call the writer's {@link #getImageQueue() getImageQueue()} method to get the
 * writer's image queue, which is an instance of class {@linkplain
 * ByteImageQueue}.
 * <P><LI>
 * Generate rows of pixel data and put them into the image queue by calling the
 * image queue's {@link ByteImageQueue#put(int,byte[]) put()} method. Each row
 * of pixel data must be stored in a <TT>byte[]</TT> array whose length is equal
 * to the number of columns in the image. Each element of the array specifies
 * the palette index of the color for one pixel. You must call the image queue's
 * {@link ByteImageQueue#put(int,byte[]) put()} method once for every row of
 * pixels in the image. You do not have to put the rows in any particular order.
 * <P><LI>
 * Call the writer's {@link #write() write()} method to write the pixel data to
 * the PNG image.
 * </OL>
 * <P>
 * Steps 5 and 6 can be performed concurrently in separate threads. In Step 5,
 * multiple threads can put the pixel data rows into the image queue
 * concurrently.
 *
 * @author  Alan Kaminsky
 * @version 19-Jul-2013
 */
public class IndexPngWriter
	{

// Hidden data members.

	private PLTEChunk plteChunk;
	private ChunkOutputStream chunkOutputStream;
	private IDATCompressor compressor;
	private ByteImageQueue imageQueue;
	private pHYsChunk physChunk =
		new pHYsChunk (300, 300, PixelUnit.PIXELS_PER_INCH);

// Exported constructors.

	/**
	 * Construct a new indexed color PNG writer. The PNG image consists of the
	 * given number of rows and columns. The PNG image will be written to the
	 * given output stream. The PNG image will use the given palette.
	 * <P>
	 * The palette is an {@linkplain edu.rit.util.AList AList} of {@linkplain
	 * Color Color}s; it must contain at least 1 and at most 256 elements.
	 *
	 * @param  rows     Number of rows. Must be &ge; 1.
	 * @param  cols     Number of columns. Must be &ge; 1.
	 * @param  out      Output stream.
	 * @param  palette  Palette.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>out</TT> is null or
	 *     <TT>palette</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>rows</TT> &lt; 1 or <TT>cols</TT>
	 *     &lt; 1. Thrown if the number of elements in <TT>palette</TT> is
	 *     illegal.
	 */
	public IndexPngWriter
		(int rows,
		 int cols,
		 OutputStream out,
		 AList<Color> palette)
		{
		if (rows < 1)
			throw new IllegalArgumentException (String.format
				("BilevelPngWriter(): rows = %d illegal", rows));
		if (cols < 1)
			throw new IllegalArgumentException (String.format
				("BilevelPngWriter(): cols = %d illegal", cols));
		plteChunk = new PLTEChunk (palette);
		chunkOutputStream = new ChunkOutputStream (out);
		compressor = new IDATCompressor (chunkOutputStream, 0);
		imageQueue = new ByteImageQueue (rows, cols);
		}

// Exported operations.

	/**
	 * Set this indexed color PNG writer's physical pixel dimensions to the
	 * given values. If not specified, the default is 300&times;300 pixels per
	 * inch.
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
	 * Returns this indexed color PNG writer's image queue.
	 *
	 * @return  Image queue.
	 */
	public ByteImageQueue getImageQueue()
		{
		return imageQueue;
		}

	/**
	 * Write this indexed color PNG writer's image to the underlying output
	 * stream. The pixel data is obtained from the image queue. The underlying
	 * output stream is closed once the image is written.
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
		chunkOutputStream.writeSignature();
		chunkOutputStream.write
			(new IHDRChunk
				(/*width    */ cols,
				 /*height   */ rows,
				 /*bitDepth */ 8,
				 /*colorType*/ 3));
		chunkOutputStream.write (physChunk);
		chunkOutputStream.write (plteChunk);
		for (int r = 0; r < rows; ++ r)
			{
			compressor.write (0, false); // Filter method = 0 = none
			compressor.write (imageQueue.remove (r), r == rows - 1);
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
//		AList<Color> palette = new AList<Color>();
//		palette.addLast (new Color() .rgb (255,   0,   0));
//		palette.addLast (new Color() .rgb (255, 128,   0));
//		palette.addLast (new Color() .rgb (255, 255,   0));
//		palette.addLast (new Color() .rgb (  0, 255,   0));
//		palette.addLast (new Color() .rgb (  0, 255, 255));
//		palette.addLast (new Color() .rgb (  0,   0, 255));
//		palette.addLast (new Color() .rgb (255,   0, 255));
//		final IndexPngWriter writer = new IndexPngWriter (100, 175,
//			new BufferedOutputStream (new FileOutputStream (file)), palette);
//		ByteImageQueue imageQueue = writer.getImageQueue();
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
//		byte[] data = new byte [175];
//		for (int r = 0; r < 100; ++ r)
//			{
//			for (int c = 0; c < 175; ++ c)
//				data[c] = (byte)(c/25);
//			imageQueue.put (r, data);
//			}
//		}
//
//	private static void usage()
//		{
//		System.err.println ("Usage: java edu.rit.image.IndexPngWriter <file>");
//		System.exit (1);
//		}

	}
