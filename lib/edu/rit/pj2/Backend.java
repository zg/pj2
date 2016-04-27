//******************************************************************************
//
// File:    Backend.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.Backend
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

package edu.rit.pj2;

import edu.rit.gpu.Gpu;
import edu.rit.io.ThreadedOutputStream;
import edu.rit.pj2.tracker.BackendReceiver;
import edu.rit.pj2.tracker.BackendRef;
import edu.rit.pj2.tracker.JobRef;
import edu.rit.pj2.tracker.JobSender;
import edu.rit.pj2.tracker.Proxy;
import edu.rit.pj2.tracker.Receiver;
import edu.rit.pj2.tracker.ReceiverListener;
import edu.rit.pj2.tracker.TaskInfo;
import edu.rit.pj2.tracker.TaskProperties;
import edu.rit.util.AList;
import edu.rit.util.Heartbeat;
import edu.rit.util.Instance;
import edu.rit.util.Logger;
import edu.rit.util.PrintStreamLogger;
import java.io.IOException;
import java.io.PrintStream;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Semaphore;

/**
 * Class Backend provides an object and main program that runs a {@linkplain
 * edu.rit.pj2.Task Task} on a certain node of a cluster. The Backend prints log
 * messages on the {@linkplain edu.rit.pj2.tracker.Launcher Launcher}'s standard
 * output.
 * <P>
 * Usage: <TT>java edu.rit.pj2.Backend <I>name</I> <I>jobID</I>
 * <I>taskID</I> <I>jobHost</I> <I>jobPort</I></TT>
 * <P>
 * <I>Note:</I> The Backend constructs an instance of the task's class using the
 * class's no-argument constructor, with access checks suppressed. This means
 * the task's class and/or the task's no-argument constructor need not be
 * public, and a new instance will still be constructed. However, this also
 * requires that either (a) a security manager is not installed, or (b) the
 * security manager allows ReflectPermission("suppressAccessChecks"). See the
 * <TT>java.lang.reflect.Constructor.setAccessible()</TT> method for further
 * information.
 *
 * @author  Alan Kaminsky
 * @version 19-Sep-2014
 */
public class Backend
	implements BackendRef
	{

// Hidden data members.

	private static String name;
	private static long jobID;
	private static long taskID;
	private static String jobHost;
	private static int jobPort;

	private static Logger logger;

	private static BackendRef backend;
	private static JobSender job;
	private static Proxy jobProxy;

	private static ScheduledExecutorService executor;
	private static Heartbeat heartbeat;

	private static Semaphore finishSema = new Semaphore (0);
	private static Throwable excThrown;

	private static TupleRequestMap tupleRequestMap = new TupleRequestMap();

	private static ThreadedOutputStream thrOut;
	private static ThreadedOutputStream thrErr;

// Hidden constructors.

	private Backend()
		{
		}

// Main program.

	/**
	 * Main program.
	 *
	 * @param  args  Array of command line arguments.
	 */
	public static void main
		(String[] args)
		{
		// Parse command line arguments.
		if (args.length != 5) usage();
		name = args[0];
		try
			{
			jobID = Long.parseLong (args[1]);
			if (jobID < 1)
				usage (String.format ("<jobID> = %s illegal", args[1]));
			}
		catch (NumberFormatException exc)
			{
			usage (String.format ("<jobID> = %s illegal", args[1]));
			}
		try
			{
			taskID = Long.parseLong (args[2]);
			if (taskID < 1)
				usage (String.format ("<taskID> = %s illegal", args[2]));
			}
		catch (NumberFormatException exc)
			{
			usage (String.format ("<taskID> = %s illegal", args[2]));
			}
		jobHost = args[3];
		try
			{
			jobPort = Integer.parseInt (args[4]);
			if (0 > jobPort || jobPort > 65535)
				usage (String.format ("<jobPort> = %s illegal", args[4]));
			}
		catch (NumberFormatException exc)
			{
			usage (String.format ("<jobPort> = %s illegal", args[4]));
			}

		// Log messages on standard output.
		logger = new PrintStreamLogger (System.out);
		logger.prefix (String.format ("Job %d task %d backend", jobID, taskID));

		// Create Backend.
		backend = new Backend();

		// Open a connection to the Job.
		job = new JobSender();
		try
			{
			jobProxy = new Proxy
				(jobHost,
				 jobPort,
				 job,
				 new BackendReceiver
					(new ReceiverListener()
						{
						public void receiverFailed
							(Receiver receiver,
							 Throwable exc)
							{
							try { job.taskFailed (taskID, exc); }
								catch (IOException exc2) {}
							stopBackend (exc);
							}
						},
					 backend));
			}
		catch (IOException exc)
			{
			logger.log (String.format ("Cannot connect to job at %s:%d",
				jobHost, jobPort));
			System.exit (0);
			}

		// Backend has started successfully.
		try
			{
			// Add a shutdown hook that will run when the Backend exits.
			Runtime.getRuntime().addShutdownHook (new Thread()
				{
				public void run()
					{
					flushConsoleStreams();
					if (heartbeat != null)
						heartbeat.cancel();
					if (executor != null)
						executor.shutdownNow();
					if (jobProxy != null)
						jobProxy.terminate();
					}
				});

			// Start heartbeats with the Job.
			executor = Executors.newSingleThreadScheduledExecutor();
			heartbeat = new Heartbeat()
				{
				protected void sendHeartbeat() throws IOException
					{
					job.heartbeatFromTask (taskID);
					}
				protected void died()
					{
					logger.log ("Job heartbeat died");
					stopBackend (null);
					}
				};
			heartbeat.enable (executor);

			// Set up multiple thread safe console streams.
			thrOut = new ThreadedOutputStream (new JobOutputStream (job, 0));
			thrErr = new ThreadedOutputStream (new JobOutputStream (job, 1));
			System.setOut (new PrintStream (thrOut, false));
			System.setErr (new PrintStream (thrErr, false));

			// Tell the Job that the Backend started.
			job.taskLaunched (backend, taskID, name);

			// All further work is done in response to incoming messages.
			}

		// Capture any exception thrown above.
		catch (Throwable exc)
			{
			stopBackend (exc);
			}

		// Wait until the backend is finished.
		finishSema.acquireUninterruptibly();
		if (excThrown != null)
			logger.log ("Unexpected exception", excThrown);
		if (jobProxy != null)
			{
			jobProxy.shutdown();
			jobProxy = null;
			}
		System.exit (0);
		}

// Exported operations.

	/**
	 * Tell this backend to start a task with the given information.
	 *
	 * @param  info  Task information object.
	 */
	public void startTask
		(TaskInfo info)
		{
		new BackendThread (info) .start();
		}

	/**
	 * Tell this backend that the given tuple was taken out of tuple space.
	 *
	 * @param  taskID     Task ID.
	 * @param  requestID  Request ID.
	 * @param  tuple      Tuple.
	 */
	public void tupleTaken
		(long taskID,
		 long requestID,
		 Tuple tuple)
		{
		tupleRequestMap.tupleTaken (taskID, requestID, tuple);
		}

	/**
	 * Tell this backend to stop the task.
	 */
	public void stopTask()
		throws IOException
		{
		logger.log ("Task stopped");
		stopBackend (null);
		}

	/**
	 * Tell this backend that the job is still alive.
	 */
	public void heartbeatFromJob()
		{
		heartbeat.receiveHeartbeat();
		}

	/**
	 * Gracefully shut down communication with this backend.
	 */
	public void shutdown()
		{
		}

	/**
	 * Forcibly terminate communication with this backend.
	 */
	public void terminate()
		{
		}

// Hidden helper thread.

	private class BackendThread
		extends Thread
		{
		private TaskInfo info;

		/**
		 * Construct a new thread for running the given task.
		 *
		 * @param  info  Task information record.
		 */
		public BackendThread
			(TaskInfo info)
			{
			this.info = info;
			}

		/**
		 * Run the task.
		 */
		public void run()
			{
			try
				{
				// Install JAR class loader if necessary.
				if (info.jar != null)
					{
					JarClassLoader loader =
						new JarClassLoader
							(Thread.currentThread().getContextClassLoader(),
							 info.jar);
					Thread.currentThread().setContextClassLoader (loader);
					jobProxy.setContextClassLoader (loader);
					}

				// Create instance of task subclass.
				Task task = (Task)
					Instance.newDefaultInstance (info.taskClassName, true);

				// Set up task's data.
				task.job = job;
				task.taskID = info.taskID;
				info.unmarshalInputTuples();
				task.inputTuples = info.inputTuples;
				task.tupleRequestMap = tupleRequestMap;
				task.properties = new TaskProperties (info.properties);
				task.groupSize = info.size;
				task.taskRank = info.rank;

				// Specify GPU device numbers the task is allowed to use.
				Gpu.setDeviceNumbers (info.devnum);

				// Run the task.
				task.main (info.args);
				flushConsoleStreams();

				// Inform the Job that the task finished successfully.
				job.taskFinished (info.taskID);

				// Exit the backend process.
				stopBackend (null);
				}

			catch (Throwable exc)
				{
				flushConsoleStreams();

				// Inform the Job that the task failed.
				try { job.taskFailed (info.taskID, exc); }
					catch (IOException exc2) {}

				// Exit the backend process.
				stopBackend (exc);
				}
			}
		}

// Hidden operations.

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.out.printf ("Usage: java edu.rit.pj2.Backend <name> <jobID> <taskID> <jobHost> <jobPort>%n");
		System.exit (0);
		}

	/**
	 * Print a usage message and exit.
	 *
	 * @param  msg  Error message.
	 */
	private static void usage
		(String msg)
		{
		System.out.printf ("Backend: %s%n", msg);
		usage();
		}

	/**
	 * Stop the Backend program.
	 *
	 * @param  exc  Exception that was thrown, or null if none.
	 */
	private static void stopBackend
		(Throwable exc)
		{
		excThrown = exc;
		finishSema.release();
		}

	/**
	 * Flush the console streams.
	 */
	private static void flushConsoleStreams()
		{
		if (thrOut != null)
			try
				{
				thrOut.flushAll();
				}
			catch (IOException exc)
				{
				}
		if (thrErr != null)
			try
				{
				thrErr.flushAll();
				}
			catch (IOException exc)
				{
				}
		}

	}
