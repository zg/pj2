//******************************************************************************
//
// File:    LauncherSender.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.LauncherSender
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
 * Class LauncherSender provides an object for sending messages to a Launcher. A
 * launcher sender must be attached to a {@linkplain Proxy Proxy} before the
 * launcher sender can be used.
 *
 * @author  Alan Kaminsky
 * @version 30-Nov-2013
 */
public class LauncherSender
	extends Sender
	implements LauncherRef
	{

// Exported constructors.

	/**
	 * Construct a new launcher sender.
	 */
	public LauncherSender()
		{
		opcode = Opcode.LAUNCHER;
		}

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
	public synchronized void launch
		(long jobID,
		 long taskID,
		 String[] jvmFlags,
		 String jobHost,
		 int jobPort)
		throws IOException
		{
		out.writeByte (Opcode.LAUNCHERREF_LAUNCH);
		out.writeLong (jobID);
		out.writeLong (taskID);
		out.writeStringArray (jvmFlags);
		out.writeString (jobHost);
		out.writeInt (jobPort);
		out.flush();
		}

	/**
	 * Tell this launcher that the tracker is still alive.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void heartbeatFromTracker()
		throws IOException
		{
		out.writeByte (Opcode.LAUNCHERREF_HEARTBEAT_FROM_TRACKER);
		out.flush();
		}

	}
