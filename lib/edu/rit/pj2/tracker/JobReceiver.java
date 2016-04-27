//******************************************************************************
//
// File:    JobReceiver.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.JobReceiver
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
import java.io.EOFException;
import java.net.SocketException;

/**
 * Class JobReceiver provides a thread that receives messages from a Job. A job
 * receiver must be attached to a {@linkplain Proxy Proxy} before the job
 * receiver can be used.
 *
 * @author  Alan Kaminsky
 * @version 17-Jun-2014
 */
public class JobReceiver
	extends Receiver
	{

// Hidden data members.

	private JobRef job;

// Exported constructors.

	/**
	 * Construct a new job receiver. Failures are reported to the given
	 * listener. Incoming messages invoke methods on the given Job.
	 *
	 * @param  listener  Receiver listener.
	 * @param  job       Job.
	 */
	public JobReceiver
		(ReceiverListener listener,
		 JobRef job)
		{
		super (listener);
		this.job = job;
		opcode = Opcode.JOB;
		}

// Exported operations.

	/**
	 * Run this job receiver thread.
	 */
	public void run()
		{
		byte opcode;
		long jobID, taskID, requestID;
		int[] devnum;
		boolean runInJobProcess, blocking, taking;
		String name;
		Tuple tuple, template;
		Throwable exc;
		int copies, stream, len;
		byte[] data = new byte [256];

		try
			{
			// Repeatedly read a message and invoke a method on the job.
//in.debug = true;
			for (;;)
				{
				opcode = in.readByte();
				switch (opcode)
					{
					case Opcode.JOBREF_JOB_LAUNCHED:
//System.out.printf ("JobReceiver(): JOBREF_JOB_LAUNCHED%n");
						jobID = in.readLong();
						job.jobLaunched (jobID);
						break;
					case Opcode.JOBREF_JOB_STARTED:
//System.out.printf ("JobReceiver(): JOBREF_JOB_STARTED%n");
						job.jobStarted();
						break;
					case Opcode.JOBREF_TASK_LAUNCHING:
//System.out.printf ("JobReceiver(): JOBREF_TASK_LAUNCHING%n");
						taskID = in.readLong();
						devnum = in.readIntArray();
						runInJobProcess = in.readBoolean();
						job.taskLaunching (taskID, devnum, runInJobProcess);
						devnum = null;
						break;
					case Opcode.JOBREF_TASK_LAUNCHED:
//System.out.printf ("JobReceiver(): JOBREF_TASK_LAUNCHED%n");
						taskID = in.readLong();
						name = in.readString();
						job.taskLaunched ((BackendRef)sender, taskID, name);
						name = null;
						break;
					case Opcode.JOBREF_TAKE_TUPLE:
//System.out.printf ("JobReceiver(): JOBREF_TAKE_TUPLE%n");
						taskID = in.readLong();
						requestID = in.readLong();
						template = (Tuple) in.readObject();
						blocking = in.readBoolean();
						taking = in.readBoolean();
						job.takeTuple (taskID, requestID, template, blocking,
							taking);
						template = null;
						break;
					case Opcode.JOBREF_WRITE_TUPLE:
//System.out.printf ("JobReceiver(): JOBREF_WRITE_TUPLE%n");
						taskID = in.readLong();
						tuple = (Tuple) in.readObject();
						copies = in.readInt();
						job.writeTuple (taskID, tuple, copies);
						tuple = null;
						break;
					case Opcode.JOBREF_TASK_FINISHED:
//System.out.printf ("JobReceiver(): JOBREF_TASK_FINISHED%n");
						taskID = in.readLong();
						job.taskFinished (taskID);
						break;
					case Opcode.JOBREF_TASK_FAILED:
//System.out.printf ("JobReceiver(): JOBREF_TASK_FAILED%n");
						taskID = in.readLong();
						exc = (Throwable) in.readObject();
						job.taskFailed (taskID, exc);
						exc = null;
						break;
					case Opcode.JOBREF_HEARTBEAT_FROM_TRACKER:
//System.out.printf ("JobReceiver(): JOBREF_HEARTBEAT_FROM_TRACKER%n");
						job.heartbeatFromTracker();
						break;
					case Opcode.JOBREF_HEARTBEAT_FROM_TASK:
//System.out.printf ("JobReceiver(): JOBREF_HEARTBEAT_FROM_TASK%n");
						taskID = in.readLong();
						job.heartbeatFromTask (taskID);
						break;
					case Opcode.JOBREF_WRITE_STANDARD_STREAM:
//System.out.printf ("JobReceiver(): JOBREF_WRITE_STANDARD_STREAM%n");
						stream = in.readByte();
						len = in.readInt();
						if (len > data.length)
							data = new byte [len];
						in.readByteArray (data, 0, len);
						job.writeStandardStream (stream, len, data);
						break;
					case Opcode.SHUTDOWN:
//System.out.printf ("JobReceiver(): SHUTDOWN%n");
						throw new EOFException();
					default:
						throw new IllegalArgumentException (String.format
							("JobReceiver.run(): Opcode = %d illegal",
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
