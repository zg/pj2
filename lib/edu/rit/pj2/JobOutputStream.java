//******************************************************************************
//
// File:    JobOutputStream.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.JobOutputStream
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

package edu.rit.pj2;

import edu.rit.pj2.tracker.JobRef;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Class JobOutputStream provides an object for printing on a {@linkplain
 * edu.rit.pj2.Job Job}'s standard output or standard error stream. When one of
 * the job's {@linkplain edu.rit.pj2.Task Task}s is run in a process other than
 * the job's own process, the task's process's <TT>System.out</TT> and
 * <TT>System.err</TT> are replaced with print streams that write to instances
 * of class JobOutputStream. Printouts on the task's <TT>System.out</TT> and
 * <TT>System.err</TT> will then be sent to the job's process's
 * <TT>System.out</TT> and <TT>System.err</TT>.
 *
 * @author  Alan Kaminsky
 * @version 30-May-2014
 */
class JobOutputStream
	extends OutputStream
	{

// Hidden data members.

	private JobRef job;
	private int stream;

// Exported constructors.

	/**
	 * Construct a new job output stream.
	 *
	 * @param  job     Job.
	 * @param  stream  0 to write to standard output, 1 to write to standard
	 *                 error.
	 */
	public JobOutputStream
		(JobRef job,
		 int stream)
		{
		this.job = job;
		this.stream = stream;
		}

// Exported operations.

	/**
	 * Write the given byte to this job output stream. This operation is not
	 * supported.
	 *
	 * @param  b  Byte.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void write
		(int b)
		throws IOException
		{
		throw new UnsupportedOperationException();
		}

	/**
	 * Write a portion of the given byte array to this job output stream.
	 *
	 * @param  buf  Byte array to write.
	 * @param  off  Index of first byte to write; must be 0.
	 * @param  len  Number of bytes to write.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>buf</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &ne; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>len</TT> &gt; <TT>buf.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void write
		(byte[] buf,
		 int off,
		 int len)
		throws IOException
		{
		if (off != 0 || len < 0 || len > buf.length)
			throw new IndexOutOfBoundsException();
		job.writeStandardStream (stream, len, buf);
		}

	}
