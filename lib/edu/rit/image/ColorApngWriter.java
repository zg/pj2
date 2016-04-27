//******************************************************************************
//
// File:    ColorApngWriter.java
// Package: edu.rit.image
// Unit:    Class edu.rit.image.ColorApngWriter
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

import edu.rit.image.png.acTLChunk;
import edu.rit.image.png.ChunkOutputStream;
import edu.rit.image.png.Compressor;
import edu.rit.image.png.fcTLChunk;
import edu.rit.image.png.fdATCompressor;
import edu.rit.image.png.IDATCompressor;
import edu.rit.image.png.IENDChunk;
import edu.rit.image.png.IHDRChunk;
import java.io.IOException;
import java.io.OutputStream;

// For unit test main program:
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;

/**
 * Class ColorApngWriter provides an object for writing a 24-bit color APNG
 * animated image. To write an animated image:
 * <OL TYPE=1>
 * <P><LI>
 * Create an instance of class ColorApngWriter, specifying the image dimensions,
 * number of frames, number of plays, and the output stream to which to write
 * the image.
 * <P><LI>
 * Call the writer's {@link #getImageQueue() getImageQueue()} method to get the
 * writer's image queue, which is an instance of class {@linkplain
 * ColorImageQueue}.
 * <P><LI>
 * For each frame of the animated image:
 * <OL TYPE=a>
 * <P><LI>
 * Generate rows of pixel data and put them into the image queue by calling the
 * image queue's {@link ColorImageQueue#put(int,ColorArray) put()} method. Each
 * row of pixel data must be stored in a {@linkplain ColorArray} whose length is
 * equal to the number of columns in the image. Each element of the array
 * specifies the color for one pixel. You must call the image queue's {@link
 * ColorImageQueue#put(int,ColorArray) put()} method once for every row of
 * pixels in the frame. You do not have to put the rows in any particular order.
 * <P><LI>
 * Call the writer's {@link #writeFrame() writeFrame()} method to write the
 * frame's pixel data to the APNG animated image.
 * </OL>
 * </OL>
 * <P>
 * Steps 3a and 3b can be performed concurrently in separate threads. In Step
 * 3a, multiple threads can put the pixel data rows into the image queue
 * concurrently.
 *
 * @author  Alan Kaminsky
 * @version 28-Jun-2013
 */
public class ColorApngWriter
	{

// Hidden data members.

	private int frames;
	private int plays;
	private short delayNumer;
	private short delayDenom;

	private ChunkOutputStream chunkOutputStream;
	private ColorImageQueue imageQueue;

	private int framesWritten = 0;
	private int seqnum = 0;

// Exported constructors.

	/**
	 * Construct a new color APNG writer. The APNG animated image consists of
	 * the given number of frames. Each frame consists of the given number of
	 * rows and columns. The animation will be played the given number of times.
	 * Each frame will be displayed for the given number of seconds, expressed
	 * as a fraction <TT>delayNumer/delayDenom</TT>. (Use the {@link
	 * #writeFrame(short,short) writeFrame(short,short)} method to specify a
	 * different delay for a certain frame.) The APNG animated image will be
	 * written to the given output stream.
	 *
	 * @param  frames      Number of frames. Must be &ge; 1.
	 * @param  rows        Number of rows. Must be &ge; 1.
	 * @param  cols        Number of columns. Must be &ge; 1.
	 * @param  plays       Number of plays. Must be &ge; 1, or 0 to loop
	 *                     infinitely.
	 * @param  delayNumer  Frame display delay numerator. Must be &ge; 0.
	 * @param  delayDenom  Frame display delay denominator. Must be &ge; 0.
	 * @param  out         Output stream.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if any argument is illegal.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>out</TT> is null.
	 */
	public ColorApngWriter
		(int frames,
		 int rows,
		 int cols,
		 int plays,
		 short delayNumer,
		 short delayDenom,
		 OutputStream out)
		{
		if (frames < 1)
			throw new IllegalArgumentException (String.format
				("ColorApngWriter(): frames = %d illegal", frames));
		if (rows < 1)
			throw new IllegalArgumentException (String.format
				("ColorApngWriter(): rows = %d illegal", rows));
		if (cols < 1)
			throw new IllegalArgumentException (String.format
				("ColorApngWriter(): cols = %d illegal", cols));
		if (plays < 0)
			throw new IllegalArgumentException (String.format
				("ColorApngWriter(): plays = %d illegal", plays));
		if (delayNumer < 0)
			throw new IllegalArgumentException (String.format
				("ColorApngWriter(): delayNumer = %d illegal", delayNumer));
		if (delayDenom < 0)
			throw new IllegalArgumentException (String.format
				("ColorApngWriter(): delayDenom = %d illegal", delayDenom));

		this.frames = frames;
		this.plays = plays;
		this.delayNumer = delayNumer;
		this.delayDenom = delayDenom;

		chunkOutputStream = new ChunkOutputStream (out);
		imageQueue = new ColorImageQueue (rows, cols);
		}

// Exported operations.

	/**
	 * Returns this color APNG writer's image queue.
	 *
	 * @return  Image queue.
	 */
	public ColorImageQueue getImageQueue()
		{
		return imageQueue;
		}

	/**
	 * Write this color PNG writer's next frame to the underlying output stream.
	 * This frame will be displayed for the number of seconds specified to the
	 * constructor. The pixel data is obtained from the image queue. The
	 * underlying output stream is closed once all frames have been written.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 * @exception  InterruptedException
	 *     Thrown if the calling thread is interrupted while blocked in this
	 *     method.
	 */
	public void writeFrame()
		throws IOException, InterruptedException
		{
		writeFrame (delayNumer, delayDenom);
		}

	/**
	 * Write this color PNG writer's next frame to the underlying output stream.
	 * This frame will be displayed for the given number of seconds, expressed
	 * as a fraction <TT>delayNumer/delayDenom</TT>. The pixel data is obtained
	 * from the image queue. The underlying output stream is closed once all
	 * frames have been written.
	 *
	 * @param  delayNumer  Frame display delay numerator. Must be &ge; 0.
	 * @param  delayDenom  Frame display delay denominator. Must be &ge; 0.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if any argument is illegal.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 * @exception  InterruptedException
	 *     Thrown if the calling thread is interrupted while blocked in this
	 *     method.
	 */
	public void writeFrame
		(short delayNumer,
		 short delayDenom)
		throws IOException, InterruptedException
		{
		if (delayNumer < 0)
			throw new IllegalArgumentException (String.format
				("ColorApngWriter.writeFrame(): delayNumer = %d illegal",
				 delayNumer));
		if (delayDenom < 0)
			throw new IllegalArgumentException (String.format
				("ColorApngWriter.writeFrame(): delayDenom = %d illegal",
				 delayDenom));

		int rows = imageQueue.rows();
		int cols = imageQueue.cols();
		int[] pixelData;
		byte[] data = new byte [3*cols];
		byte prev_r, prev_g, prev_b;
		Compressor compressor;

		if (framesWritten == 0)
			{
			chunkOutputStream.writeSignature();
			chunkOutputStream.write
				(new IHDRChunk
					(/*width    */ cols,
					 /*height   */ rows,
					 /*bitDepth */ 8,
					 /*colorType*/ 2));
			chunkOutputStream.write
				(new acTLChunk
					(/*numFrames*/ frames,
					 /*numPlays */ plays));
			}

		chunkOutputStream.write
			(new fcTLChunk
				(/*seqnum    */ seqnum ++,
				 /*width     */ cols,
				 /*height    */ rows,
				 /*xOffset   */ 0,
				 /*yOffset   */ 0,
				 /*delayNumer*/ delayNumer,
				 /*delayDenom*/ delayDenom,
				 /*disposeOp */ fcTLChunk.APNG_DISPOSE_OP_NONE,
				 /*blendOp   */ fcTLChunk.APNG_BLEND_OP_SOURCE));

		compressor = framesWritten == 0 ?
			new IDATCompressor (chunkOutputStream, seqnum) :
			new fdATCompressor (chunkOutputStream, seqnum);

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

		seqnum = compressor.nextSequenceNumber();
		++ framesWritten;

		if (framesWritten == frames)
			{
			chunkOutputStream.write (new IENDChunk());
			chunkOutputStream.close();
			}
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
//		final ColorApngWriter writer = new ColorApngWriter
//			(/*frames*/ 200,
//			 /*rows  */ 100,
//			 /*cols  */ 200,
//			 /*plays */ 0,
//			 /*delayNumer*/ (short) 1,
//			 /*delayDenom*/ (short) 24,
//			 new BufferedOutputStream (new FileOutputStream (file)));
//		ColorImageQueue imageQueue = writer.getImageQueue();
//		new Thread()
//			{
//			public void run()
//				{
//				try
//					{
//					for (int f = 0; f < 200; ++ f)
//						writer.writeFrame();
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
//		for (int f = 0; f < 200; ++ f)
//			{
//			for (int c = 0; c < 200; ++ c)
//				data.hsb (c, ((c + f) % 200)/200.0f, 1.0f, 1.0f);
//			for (int r = 0; r < 100; ++ r)
//				imageQueue.put (r, data);
//			}
//		}
//
//	private static void usage()
//		{
//		System.err.println ("Usage: java edu.rit.image.ColorApngWriter <file>");
//		System.exit (1);
//		}

	}
