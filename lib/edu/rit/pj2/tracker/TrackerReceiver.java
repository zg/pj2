//******************************************************************************
//
// File:    TrackerReceiver.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.TrackerReceiver
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
import java.io.EOFException;
import java.net.SocketException;

/**
 * Class TrackerReceiver provides a thread that receives messages from a
 * Tracker. A tracker receiver must be attached to a {@linkplain Proxy Proxy}
 * before the tracker receiver can be used.
 *
 * @author  Alan Kaminsky
 * @version 23-Mar-2014
 */
public class TrackerReceiver
	extends Receiver
	{

// Hidden data members.

	private TrackerRef tracker;

// Exported constructors.

	/**
	 * Construct a new tracker receiver. Failures are reported to the given
	 * listener. Incoming messages invoke methods on the given Tracker.
	 *
	 * @param  listener  Receiver listener.
	 * @param  tracker   Tracker.
	 */
	public TrackerReceiver
		(ReceiverListener listener,
		 TrackerRef tracker)
		{
		super (listener);
		this.tracker = tracker;
		opcode = Opcode.TRACKER;
		}

// Exported operations.

	/**
	 * Run this tracker receiver thread.
	 */
	public void run()
		{
		byte opcode;
		NodeProperties node;
		long jobID, taskID;
		String user, jobHost, msg;
		int jobPort;
		AList<TaskSpecInfo> taskGroup;

		try
			{
			// Repeatedly read a message and invoke a method on the tracker.
			for (;;)
				{
				opcode = in.readByte();
				switch (opcode)
					{
					case Opcode.TRACKERREF_LAUNCHER_STARTED:
						node = in.readFields (new NodeProperties());
						tracker.launcherStarted ((LauncherRef)sender, node);
						node = null;
						break;
					case Opcode.TRACKERREF_LAUNCHER_STOPPED:
						tracker.launcherStopped ((LauncherRef)sender);
						break;
					case Opcode.TRACKERREF_LAUNCH_JOB:
						user = in.readString();
						jobHost = in.readString();
						jobPort = in.readInt();
						tracker.launchJob ((JobRef)sender, user, jobHost,
							jobPort);
						user = null;
						jobHost = null;
						break;
					case Opcode.TRACKERREF_LAUNCH_TASK_GROUP:
						jobID = in.readLong();
						taskGroup = (AList<TaskSpecInfo>) in.readReference();
						in.clearCache();
						tracker.launchTaskGroup (jobID, taskGroup);
						taskGroup = null;
						break;
					case Opcode.TRACKERREF_LAUNCH_FAILED:
						jobID = in.readLong();
						taskID = in.readLong();
						tracker.launchFailed (jobID, taskID);
						break;
					case Opcode.TRACKERREF_TASK_DONE:
						jobID = in.readLong();
						taskID = in.readLong();
						tracker.taskDone (jobID, taskID);
						break;
					case Opcode.TRACKERREF_JOB_DONE:
						jobID = in.readLong();
						tracker.jobDone (jobID);
						break;
					case Opcode.TRACKERREF_STOP_JOB:
						jobID = in.readLong();
						msg = in.readString();
						tracker.stopJob (jobID, msg);
						msg = null;
						break;
					case Opcode.TRACKERREF_HEARTBEAT_FROM_JOB:
						jobID = in.readLong();
						tracker.heartbeatFromJob (jobID);
						break;
					case Opcode.TRACKERREF_HEARTBEAT_FROM_LAUNCHER:
						tracker.heartbeatFromLauncher ((LauncherRef)sender);
						break;
					case Opcode.SHUTDOWN:
						throw new EOFException();
					default:
						throw new IllegalArgumentException (String.format
							("TrackerReceiver.run(): Opcode = %d illegal",
							 opcode));
					}
				}
			}

		catch (EOFException exc2)
			{
			proxy.farEndShutdown();
			}
		catch (SocketException exc2)
			{
			proxy.farEndShutdown();
			}
		catch (Throwable exc2)
			{
			listener.receiverFailed (this, exc2);
			}
		}

	}
