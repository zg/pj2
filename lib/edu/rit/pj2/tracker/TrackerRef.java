//******************************************************************************
//
// File:    TrackerRef.java
// Package: edu.rit.pj2.tracker
// Unit:    Interface edu.rit.pj2.tracker.TrackerRef
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

import edu.rit.util.AList;
import java.io.IOException;

/**
 * Interface TrackerRef specifies the interface for invoking a tracker.
 *
 * @author  Alan Kaminsky
 * @version 23-Mar-2014
 */
public interface TrackerRef
	{

// Exported operations.

	/**
	 * Tell this tracker that the given launcher started. The <TT>node</TT>
	 * gives the name and capabilities of the node where the launcher is
	 * running.
	 *
	 * @param  launcher  Launcher.
	 * @param  node      Node properties object.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void launcherStarted
		(LauncherRef launcher,
		 NodeProperties node)
		throws IOException;

	/**
	 * Tell this tracker that the given launcher stopped.
	 *
	 * @param  launcher  Launcher.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void launcherStopped
		(LauncherRef launcher)
		throws IOException;

	/**
	 * Tell this tracker to launch the given job.
	 *
	 * @param  job      Job.
	 * @param  user     User name.
	 * @param  jobHost  Host name at which job is listening for connections.
	 * @param  jobPort  Port number at which job is listening for connections.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void launchJob
		(JobRef job,
		 String user,
		 String jobHost,
		 int jobPort)
		throws IOException;

	/**
	 * Tell this tracker to launch the given task group in the given job.
	 *
	 * @param  jobID      Job ID.
	 * @param  taskGroup  Task group.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void launchTaskGroup
		(long jobID,
		 AList<TaskSpecInfo> taskGroup)
		throws IOException;

	/**
	 * Tell this tracker that the given task in the given job failed to launch.
	 *
	 * @param  jobID   Job ID.
	 * @param  taskID  Task ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void launchFailed
		(long jobID,
		 long taskID)
		throws IOException;

	/**
	 * Tell this tracker that the given task in the given job is done.
	 *
	 * @param  jobID   Job ID.
	 * @param  taskID  Task ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void taskDone
		(long jobID,
		 long taskID)
		throws IOException;

	/**
	 * Tell this tracker that the given job is done.
	 *
	 * @param  jobID  Job ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void jobDone
		(long jobID)
		throws IOException;

	/**
	 * Tell this tracker to stop the given job. The error message is included in
	 * the tracker's log.
	 *
	 * @param  jobID  Job ID.
	 * @param  msg    Error message.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void stopJob
		(long jobID,
		 String msg)
		throws IOException;

	/**
	 * Tell this tracker that the given job is still alive.
	 *
	 * @param  jobID  Job ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void heartbeatFromJob
		(long jobID)
		throws IOException;

	/**
	 * Tell this tracker that the given launcher is still alive.
	 *
	 * @param  launcher  Launcher.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void heartbeatFromLauncher
		(LauncherRef launcher)
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
