//******************************************************************************
//
// File:    BackendSender.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.BackendSender
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

package edu.rit.pj2.tracker;

import edu.rit.pj2.Tuple;
import java.io.IOException;

/**
 * Class BackendSender provides an object for sending messages to a Backend. A
 * backend sender must be attached to a {@linkplain Proxy Proxy} before the
 * backend sender can be used.
 *
 * @author  Alan Kaminsky
 * @version 30-Nov-2013
 */
public class BackendSender
	extends Sender
	implements BackendRef
	{

// Exported constructors.

	/**
	 * Construct a new backend sender.
	 */
	public BackendSender()
		{
		opcode = Opcode.BACKEND;
		}

// Exported operations.

	/**
	 * Tell this backend to start a task with the given information.
	 *
	 * @param  info  Task information object.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void startTask
		(TaskInfo info)
		throws IOException
		{
		out.writeByte (Opcode.BACKENDREF_START_TASK);
		out.writeFields (info);
		out.flush();
		}

	/**
	 * Tell this backend that the given tuple was taken out of tuple space.
	 *
	 * @param  taskID     Task ID.
	 * @param  requestID  Request ID.
	 * @param  tuple      Tuple.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void tupleTaken
		(long taskID,
		 long requestID,
		 Tuple tuple)
		throws IOException
		{
		out.writeByte (Opcode.BACKENDREF_TUPLE_TAKEN);
		out.writeLong (taskID);
		out.writeLong (requestID);
		out.writeObject (tuple);
		out.flush();
		}

	/**
	 * Tell this backend to stop the task.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void stopTask()
		throws IOException
		{
		out.writeByte (Opcode.BACKENDREF_STOP_TASK);
		out.flush();
		}

	/**
	 * Tell this backend that the job is still alive.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void heartbeatFromJob()
		throws IOException
		{
		out.writeByte (Opcode.BACKENDREF_HEARTBEAT_FROM_JOB);
		out.flush();
		}

	}
