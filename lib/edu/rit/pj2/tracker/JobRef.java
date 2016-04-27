//******************************************************************************
//
// File:    JobRef.java
// Package: edu.rit.pj2.tracker
// Unit:    Interface edu.rit.pj2.tracker.JobRef
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

package edu.rit.pj2.tracker;

import edu.rit.pj2.Tuple;
import edu.rit.util.AList;
import java.io.IOException;

/**
 * Interface JobRef specifies the interface for invoking a {@linkplain
 * edu.rit.pj2.Job Job}.
 *
 * @author  Alan Kaminsky
 * @version 24-Mar-2014
 */
public interface JobRef
	{

// Exported operations.

	/**
	 * Tell this job that the job launched.
	 *
	 * @param  jobID  Job ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void jobLaunched
		(long jobID)
		throws IOException;

	/**
	 * Tell this job that the job started.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void jobStarted()
		throws IOException;

	/**
	 * Tell this job that the given task is launching.
	 *
	 * @param  taskID           Task ID.
	 * @param  devnum           Array of GPU device numbers the task is allowed
	 *                          to use (must be non-null).
	 * @param  runInJobProcess  True to run the task in the job's process, false
	 *                          otherwise.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void taskLaunching
		(long taskID,
		 int[] devnum,
		 boolean runInJobProcess)
		throws IOException;

	/**
	 * Tell this job that the given task was launched on the given backend.
	 *
	 * @param  backend  Backend.
	 * @param  taskID   Task ID.
	 * @param  name     Name of the node where the task is running.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void taskLaunched
		(BackendRef backend,
		 long taskID,
		 String name)
		throws IOException;

	/**
	 * Tell this job to take a tuple that matches the given template out of
	 * tuple space.
	 *
	 * @param  taskID     Task ID.
	 * @param  requestID  Request ID.
	 * @param  template   Template.
	 * @param  blocking   True for a blocking take or read, false for a
	 *                    non-blocking take or read.
	 * @param  taking     True to take a tuple, false to read a tuple.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void takeTuple
		(long taskID,
		 long requestID,
		 Tuple template,
		 boolean blocking,
		 boolean taking)
		throws IOException;

	/**
	 * Tell this job to write the given tuple into tuple space.
	 *
	 * @param  taskID  Task ID.
	 * @param  tuple   Tuple.
	 * @param  copies  Number of copies to write (1 or more).
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeTuple
		(long taskID,
		 Tuple tuple,
		 int copies)
		throws IOException;

	/**
	 * Tell this job that the given task finished.
	 *
	 * @param  taskID  Task ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void taskFinished
		(long taskID)
		throws IOException;

	/**
	 * Tell this job that the given task failed.
	 *
	 * @param  taskID   Task ID.
	 * @param  exc      Exception that was thrown.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void taskFailed
		(long taskID,
		 Throwable exc)
		throws IOException;

	/**
	 * Tell this job that the tracker is still alive.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void heartbeatFromTracker()
		throws IOException;

	/**
	 * Tell this job that the given task is still alive.
	 *
	 * @param  taskID  Task ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void heartbeatFromTask
		(long taskID)
		throws IOException;

	/**
	 * Write the given data to the standard output or standard error stream.
	 *
	 * @param  stream  0 to write to standard output, 1 to write to standard
	 *                 error.
	 * @param  len     Length of data.
	 * @param  data    Data buffer.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeStandardStream
		(int stream,
		 int len,
		 byte[] data)
		throws IOException;

	/**
	 * Returns the host at which this job is listening for connections.
	 * <P>
	 * <I>Note:</I> This method does not do a remote invocation.
	 *
	 * @return  Host name.
	 */
	public String host();

	/**
	 * Returns the port at which this job is listening for connections.
	 * <P>
	 * <I>Note:</I> This method does not do a remote invocation.
	 *
	 * @return  Port number.
	 */
	public int port();

	/**
	 * Tell this job that the whole job failed.
	 * <P>
	 * <I>Note:</I> This method does not do a remote invocation.
	 *
	 * @param  exc  Exception that was thrown.
	 */
	public void jobFailed
		(Throwable exc);

	/**
	 * Gracefully shut down communication with the far end. The
	 * <TT>shutdown()</TT> method blocks until the far end acknowledges that
	 * communication was shut down.
	 */
	public void shutdown();

	/**
	 * Forcibly terminate communication with the far end. The
	 * <TT>terminate()</TT> method returns immediately, without waiting for an
	 * acknowledgment from the far end.
	 */
	public void terminate();

	}
