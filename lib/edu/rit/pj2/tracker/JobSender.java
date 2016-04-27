//******************************************************************************
//
// File:    JobSender.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.JobSender
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
import java.io.IOException;

/**
 * Class JobSender provides an object for sending messages to a Job. A job
 * sender must be attached to a {@linkplain Proxy Proxy} before the job sender
 * can be used.
 *
 * @author  Alan Kaminsky
 * @version 24-Mar-2014
 */
public class JobSender
	extends Sender
	implements JobRef
	{

// Exported constructors.

	/**
	 * Construct a new job sender.
	 */
	public JobSender()
		{
		opcode = Opcode.JOB;
		}

// Exported operations.

	/**
	 * Tell this job that the job launched.
	 *
	 * @param  jobID  Job ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void jobLaunched
		(long jobID)
		throws IOException
		{
		out.writeByte (Opcode.JOBREF_JOB_LAUNCHED);
		out.writeLong (jobID);
		out.flush();
		}

	/**
	 * Tell this job that the job started.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void jobStarted()
		throws IOException
		{
		out.writeByte (Opcode.JOBREF_JOB_STARTED);
		out.flush();
		}

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
	public synchronized void taskLaunching
		(long taskID,
		 int[] devnum,
		 boolean runInJobProcess)
		throws IOException
		{
		out.writeByte (Opcode.JOBREF_TASK_LAUNCHING);
		out.writeLong (taskID);
		out.writeIntArray (devnum);
		out.writeBoolean (runInJobProcess);
		out.flush();
		}

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
	public synchronized void taskLaunched
		(BackendRef backend,
		 long taskID,
		 String name)
		throws IOException
		{
		out.writeByte (Opcode.JOBREF_TASK_LAUNCHED);
		out.writeLong (taskID);
		out.writeString (name);
		out.flush();
		}

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
	public synchronized void takeTuple
		(long taskID,
		 long requestID,
		 Tuple template,
		 boolean blocking,
		 boolean taking)
		throws IOException
		{
		out.writeByte (Opcode.JOBREF_TAKE_TUPLE);
		out.writeLong (taskID);
		out.writeLong (requestID);
		out.writeObject (template);
		out.writeBoolean (blocking);
		out.writeBoolean (taking);
		out.flush();
		}

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
	public synchronized void writeTuple
		(long taskID,
		 Tuple tuple,
		 int copies)
		throws IOException
		{
		out.writeByte (Opcode.JOBREF_WRITE_TUPLE);
		out.writeLong (taskID);
		out.writeObject (tuple);
		out.writeInt (copies);
		out.flush();
		}

	/**
	 * Tell this job that the given task finished.
	 *
	 * @param  taskID  Task ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void taskFinished
		(long taskID)
		throws IOException
		{
		out.writeByte (Opcode.JOBREF_TASK_FINISHED);
		out.writeLong (taskID);
		out.flush();
		}

	/**
	 * Tell this job that the given task failed.
	 *
	 * @param  taskID   Task ID.
	 * @param  exc      Exception that was thrown.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void taskFailed
		(long taskID,
		 Throwable exc)
		throws IOException
		{
		out.writeByte (Opcode.JOBREF_TASK_FAILED);
		out.writeLong (taskID);
		out.writeObject (exc);
		out.flush();
		}

	/**
	 * Tell this job that the tracker is still alive.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void heartbeatFromTracker()
		throws IOException
		{
		out.writeByte (Opcode.JOBREF_HEARTBEAT_FROM_TRACKER);
		out.flush();
		}

	/**
	 * Tell this job that the given task is still alive.
	 *
	 * @param  taskID  Task ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void heartbeatFromTask
		(long taskID)
		throws IOException
		{
		out.writeByte (Opcode.JOBREF_HEARTBEAT_FROM_TASK);
		out.writeLong (taskID);
		out.flush();
		}

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
	public synchronized void writeStandardStream
		(int stream,
		 int len,
		 byte[] data)
		throws IOException
		{
		out.writeByte (Opcode.JOBREF_WRITE_STANDARD_STREAM);
		out.writeInt (stream);
		out.writeInt (len);
		out.writeByteArray (data, 0, len);
		out.flush();
		}

	/**
	 * Tell this job that the whole job failed.
	 * <P>
	 * <I>Note:</I> This method does not do a remote invocation.
	 *
	 * @param  exc  Exception that was thrown.
	 */
	public synchronized void jobFailed
		(Throwable exc)
		{
		}

	}
