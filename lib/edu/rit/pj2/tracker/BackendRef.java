//******************************************************************************
//
// File:    BackendRef.java
// Package: edu.rit.pj2.tracker
// Unit:    Interface edu.rit.pj2.tracker.BackendRef
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
 * Interface BackendRef specifies the interface for invoking a backend.
 *
 * @author  Alan Kaminsky
 * @version 04-Dec-2013
 */
public interface BackendRef
	{

// Exported operations.

	/**
	 * Tell this backend to start a task with the given information.
	 *
	 * @param  info  Task information object.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void startTask
		(TaskInfo info)
		throws IOException;

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
	public void tupleTaken
		(long taskID,
		 long requestID,
		 Tuple tuple)
		throws IOException;

	/**
	 * Tell this backend to stop the task.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void stopTask()
		throws IOException;

	/**
	 * Tell this backend that the job is still alive.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void heartbeatFromJob()
		throws IOException;

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
