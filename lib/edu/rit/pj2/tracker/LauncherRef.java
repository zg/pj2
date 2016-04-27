//******************************************************************************
//
// File:    LauncherRef.java
// Package: edu.rit.pj2.tracker
// Unit:    Interface edu.rit.pj2.tracker.LauncherRef
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

import java.io.IOException;

/**
 * Interface LauncherRef specifies the interface for invoking a launcher.
 *
 * @author  Alan Kaminsky
 * @version 04-Dec-2013
 */
public interface LauncherRef
	{

// Exported operations.

	/**
	 * Tell this launcher to launch a process for the given task in the given
	 * job.
	 *
	 * @param  jobID     Job ID.
	 * @param  taskID    Task ID.
	 * @param  jvmFlags  JVM flags for the task.
	 * @param  jobHost   Job's host name.
	 * @param  jobPort   Job's port number.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void launch
		(long jobID,
		 long taskID,
		 String[] jvmFlags,
		 String jobHost,
		 int jobPort)
		throws IOException;

	/**
	 * Tell this launcher that the tracker is still alive.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void heartbeatFromTracker()
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
