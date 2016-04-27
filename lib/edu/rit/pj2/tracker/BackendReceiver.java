//******************************************************************************
//
// File:    BackendReceiver.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.BackendReceiver
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
import java.io.EOFException;
import java.net.SocketException;

/**
 * Class BackendReceiver provides a thread that receives messages from a
 * Backend. A backend receiver must be attached to a {@linkplain Proxy Proxy}
 * before the backend receiver can be used.
 *
 * @author  Alan Kaminsky
 * @version 04-Dec-2013
 */
public class BackendReceiver
	extends Receiver
	{

// Hidden data members.

	private BackendRef backend;

// Exported constructors.

	/**
	 * Construct a new backend receiver. Failures are reported to the given
	 * listener. Incoming messages invoke methods on the given Backend.
	 *
	 * @param  listener  Receiver listener.
	 * @param  backend   Backend.
	 */
	public BackendReceiver
		(ReceiverListener listener,
		 BackendRef backend)
		{
		super (listener);
		this.backend = backend;
		opcode = Opcode.BACKEND;
		}

// Exported operations.

	/**
	 * Run this backend receiver thread.
	 */
	public void run()
		{
		byte opcode;
		TaskInfo info;
		long taskID, requestID;
		Tuple tuple;

		try
			{
			// Repeatedly read a message and invoke a method on the backend.
			for (;;)
				{
				opcode = in.readByte();
				switch (opcode)
					{
					case Opcode.BACKENDREF_START_TASK:
						info = in.readFields (new TaskInfo());
						backend.startTask (info);
						info = null;
						break;
					case Opcode.BACKENDREF_TUPLE_TAKEN:
						taskID = in.readLong();
						requestID = in.readLong();
						tuple = (Tuple) in.readObject();
						backend.tupleTaken (taskID, requestID, tuple);
						tuple = null;
						break;
					case Opcode.BACKENDREF_STOP_TASK:
						backend.stopTask();
						break;
					case Opcode.BACKENDREF_HEARTBEAT_FROM_JOB:
						backend.heartbeatFromJob();
						break;
					case Opcode.SHUTDOWN:
						throw new EOFException();
					default:
						throw new IllegalArgumentException (String.format
							("BackendReceiver.run(): Opcode = %d illegal",
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
