//******************************************************************************
//
// File:    ThreadedOutputStream.java
// Package: edu.rit.io
// Unit:    Class edu.rit.io.ThreadedOutputStream
//
// This Java source file is copyright (C) 2015 by Alan Kaminsky. All rights
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

package edu.rit.io;

import edu.rit.util.AList;
import edu.rit.util.IdentityMap;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Class ThreadedOutputStream provides an object that automatically synchronizes
 * multiple threads writing to an underlying output stream. The bytes written to
 * a ThreadedOutputStream by different threads go into separate internal
 * buffers. When a thread calls the {@link #flush() flush()} method, only that
 * thread's buffer is written to the underlying output stream.
 * <P>
 * To get a multiple safe {@link java.io.PrintStream PrintStream}, layer a new
 * print stream <I>with autoflushing turned off</I> on top of a new threaded
 * output stream on top of an underlying output stream. Multiple threads can
 * then print multiple-line outputs on the new print stream. With autoflushing
 * turned off, nothing will get written to the underlying output stream until a
 * thread calls the new print stream's <TT>flush()</TT> method. This ensures
 * that multiple-line outputs from different threads will not be commingled in
 * the underlying output stream.
 *
 * @author  Alan Kaminsky
 * @version 09-Jan-2015
 */
public class ThreadedOutputStream
	extends OutputStream
	{

// Hidden data members.

	private OutputStream out;
	private IdentityMap<Thread,ByteArrayOutputStream> map =
		new IdentityMap<Thread,ByteArrayOutputStream>();
	private AList<ByteArrayOutputStream> buffers =
		new AList<ByteArrayOutputStream>();

// Exported constructors.

	/**
	 * Construct a new threaded output stream.
	 *
	 * @param  out  Underlying output stream.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>out</TT> is null.
	 */
	public ThreadedOutputStream
		(OutputStream out)
		{
		if (out == null)
			throw new NullPointerException
				("ThreadedOutputStream(): out is null");
		this.out = out;
		}

// Exported operations.

	/**
	 * Write the given byte to this threaded output stream. The byte is stored
	 * in the calling thread's buffer and is not written to the underlying
	 * output stream until the {@link #flush() flush()} method is called.
	 *
	 * @param  b  Byte. Only the least significant 8 bits of <TT>b</TT> are
	 *            written.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void write
		(int b)
		throws IOException
		{
		getThreadBuffer().write (b);
		}

	/**
	 * Write a portion of the given byte array to this threaded output stream.
	 * The bytes are stored in the calling thread's buffer and are not written
	 * to the underlying output stream until the {@link #flush() flush()} method
	 * is called.
	 *
	 * @param  buf  Byte array to write.
	 * @param  off  Index of first byte to write.
	 * @param  len  Number of bytes to write.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>buf</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>buf.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void write
		(byte[] buf,
		 int off,
		 int len)
		throws IOException
		{
		getThreadBuffer().write (buf, off, len);
		}

	/**
	 * Flush the calling thread's buffer to the underlying output stream. The
	 * <TT>flush()</TT> method is synchronized, so that multiple threads calling
	 * <TT>flush()</TT> will not have their buffers' contents commingled.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void flush()
		throws IOException
		{
		ByteArrayOutputStream buffer = getThreadBuffer();
		buffer.writeTo (out);
		buffer.reset();
		out.flush();
		}

	/**
	 * Flush all the threads' buffers to the underlying output stream. The order
	 * in which the threads' buffers are flushed is not specified.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void flushAll()
		throws IOException
		{
		ensureOpen();
		int n = buffers.size();
		for (int i = 0; i < n; ++ i)
			{
			ByteArrayOutputStream buffer = buffers.get (i);
			buffer.writeTo (out);
			buffer.reset();
			out.flush();
			}
		}

	/**
	 * Close this threaded output stream. All the threads' buffers are flushed
	 * to the underlying output stream in an unspecified order; the underlying
	 * output stream is closed; and this threaded output stream is closed.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void close()
		throws IOException
		{
		ensureOpen();
		try
			{
			flushAll();
			out.close();
			}
		finally
			{
			out = null;
			map = null;
			buffers = null;
			}
		}

// Hidden operations.

	/**
	 * Ensure that this threaded output stream is open.
	 *
	 * @exception  IOException
	 *     Thrown if this threaded output stream is closed.
	 */
	private void ensureOpen()
		throws IOException
		{
		if (out == null)
			throw new IOException ("Threaded output stream closed");
		}

	/**
	 * Get the buffer for the calling thread.
	 *
	 * @return  Thread buffer.
	 *
	 * @exception  IOException
	 *     Thrown if this threaded output stream is closed.
	 */
	private synchronized ByteArrayOutputStream getThreadBuffer()
		throws IOException
		{
		ensureOpen();
		Thread thr = Thread.currentThread();
		ByteArrayOutputStream buffer = map.get (thr);
		if (buffer == null)
			{
			buffer = new ByteArrayOutputStream();
			map.put (thr, buffer);
			buffers.addLast (buffer);
			}
		return buffer;
		}

	}
