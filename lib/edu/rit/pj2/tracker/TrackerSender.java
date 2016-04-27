//******************************************************************************
//
// File:    TrackerSender.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.TrackerSender
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

import edu.rit.util.AList;
import java.io.IOException;

/**
 * Class TrackerSender provides an object for sending messages to a Tracker. A
 * tracker sender must be attached to a {@linkplain Proxy Proxy} before the
 * tracker sender can be used.
 *
 * @author  Alan Kaminsky
 * @version 17-Dec-2013
 */
public class TrackerSender
	extends Sender
	implements TrackerRef
	{

// Exported constructors.

	/**
	 * Construct a new tracker sender.
	 */
	public TrackerSender()
		{
		opcode = Opcode.TRACKER;
		}

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
	public synchronized void launcherStarted
		(LauncherRef launcher,
		 NodeProperties node)
		throws IOException
		{
		out.writeByte (Opcode.TRACKERREF_LAUNCHER_STARTED);
		out.writeFields (node);
		out.flush();
		}

	/**
	 * Tell this tracker that the given launcher stopped.
	 *
	 * @param  launcher  Launcher.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void launcherStopped
		(LauncherRef launcher)
		throws IOException
		{
		out.writeByte (Opcode.TRACKERREF_LAUNCHER_STOPPED);
		out.flush();
		}

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
	public synchronized void launchJob
		(JobRef job,
		 String user,
		 String jobHost,
		 int jobPort)
		throws IOException
		{
		out.writeByte (Opcode.TRACKERREF_LAUNCH_JOB);
		out.writeString (user);
		out.writeString (jobHost);
		out.writeInt (jobPort);
		out.flush();
		}

	/**
	 * Tell this tracker to launch the given task group in the given job.
	 *
	 * @param  jobID      Job ID.
	 * @param  taskGroup  Task group.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void launchTaskGroup
		(long jobID,
		 AList<TaskSpecInfo> taskGroup)
		throws IOException
		{
		out.writeByte (Opcode.TRACKERREF_LAUNCH_TASK_GROUP);
		out.writeLong (jobID);
		out.writeReference (taskGroup);
		out.clearCache();
		out.flush();
		}

	/**
	 * Tell this tracker that the given task in the given job failed to launch.
	 *
	 * @param  jobID   Job ID.
	 * @param  taskID  Task ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void launchFailed
		(long jobID,
		 long taskID)
		throws IOException
		{
		out.writeByte (Opcode.TRACKERREF_LAUNCH_FAILED);
		out.writeLong (jobID);
		out.writeLong (taskID);
		out.flush();
		}

	/**
	 * Tell this tracker that the given task in the given job is done.
	 *
	 * @param  jobID   Job ID.
	 * @param  taskID  Task ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void taskDone
		(long jobID,
		 long taskID)
		throws IOException
		{
		out.writeByte (Opcode.TRACKERREF_TASK_DONE);
		out.writeLong (jobID);
		out.writeLong (taskID);
		out.flush();
		}

	/**
	 * Tell this tracker that the given job is done.
	 *
	 * @param  jobID  Job ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void jobDone
		(long jobID)
		throws IOException
		{
		out.writeByte (Opcode.TRACKERREF_JOB_DONE);
		out.writeLong (jobID);
		out.flush();
		}

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
	public synchronized void stopJob
		(long jobID,
		 String msg)
		throws IOException
		{
		out.writeByte (Opcode.TRACKERREF_STOP_JOB);
		out.writeLong (jobID);
		out.writeString (msg);
		out.flush();
		}

	/**
	 * Tell this tracker that the given job is still alive.
	 *
	 * @param  jobID  Job ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void heartbeatFromJob
		(long jobID)
		throws IOException
		{
		out.writeByte (Opcode.TRACKERREF_HEARTBEAT_FROM_JOB);
		out.writeLong (jobID);
		out.flush();
		}

	/**
	 * Tell this tracker that the given launcher is still alive.
	 *
	 * @param  launcher  Launcher.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void heartbeatFromLauncher
		(LauncherRef launcher)
		throws IOException
		{
		out.writeByte (Opcode.TRACKERREF_HEARTBEAT_FROM_LAUNCHER);
		out.flush();
		}

	}
