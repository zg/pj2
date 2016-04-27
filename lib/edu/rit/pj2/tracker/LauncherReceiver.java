//******************************************************************************
//
// File:    LauncherReceiver.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.LauncherReceiver
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

import java.io.EOFException;
import java.net.SocketException;

/**
 * Class LauncherReceiver provides a thread that receives messages from a
 * Launcher. A launcher receiver must be attached to a {@linkplain Proxy Proxy}
 * before the launcher receiver can be used.
 *
 * @author  Alan Kaminsky
 * @version 04-Dec-2013
 */
public class LauncherReceiver
	extends Receiver
	{

// Hidden data members.

	private LauncherRef launcher;

// Exported constructors.

	/**
	 * Construct a new launcher receiver. Failures are reported to the given
	 * listener. Incoming messages invoke methods on the given Launcher.
	 *
	 * @param  listener  Receiver listener.
	 * @param  launcher  Launcher.
	 */
	public LauncherReceiver
		(ReceiverListener listener,
		 LauncherRef launcher)
		{
		super (listener);
		this.launcher = launcher;
		opcode = Opcode.LAUNCHER;
		}

// Exported operations.

	/**
	 * Run this launcher receiver thread.
	 */
	public void run()
		{
		byte opcode;
		long jobID, taskID;
		int len;
		String[] jvmFlags;
		String jobHost;
		int jobPort;

		try
			{
			// Repeatedly read a message and invoke a method on the launcher.
			for (;;)
				{
				opcode = in.readByte();
				switch (opcode)
					{
					case Opcode.LAUNCHERREF_LAUNCH:
						jobID = in.readLong();
						taskID = in.readLong();
						jvmFlags = in.readStringArray();
						jobHost = in.readString();
						jobPort = in.readInt();
						launcher.launch (jobID, taskID, jvmFlags,
							jobHost, jobPort);
						jvmFlags = null;
						jobHost = null;
						break;
					case Opcode.LAUNCHERREF_HEARTBEAT_FROM_TRACKER:
						launcher.heartbeatFromTracker();
						break;
					case Opcode.SHUTDOWN:
						throw new EOFException();
					default:
						throw new IllegalArgumentException (String.format
							("LauncherReceiver.run(): Opcode = %d illegal",
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
