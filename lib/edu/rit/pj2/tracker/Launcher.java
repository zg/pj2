//******************************************************************************
//
// File:    Launcher.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.Launcher
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

import edu.rit.util.Heartbeat;
import edu.rit.util.Logger;
import edu.rit.util.PrintStreamLogger;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Semaphore;

/**
 * Class Launcher provides a Launcher object and main program for launching
 * processes on a certain node of a cluster. These processes run the {@linkplain
 * edu.rit.pj2.Backend Backend} main program. The Launcher prints log messages
 * on the standard output. All Backends also print log messages on the
 * Launcher's standard output.
 * <P>
 * Usage: <TT>java edu.rit.pj2.tracker.Launcher
 * [tracker=<I>host</I>[:<I>port</I>]] [name=<I>name</I>] [cores=<I>cores</I>]
 * [gpus=<I>gpus</I>] [command="<I>command</I>"]</TT>
 * <P>
 * The following options may be specified:
 * <UL>
 * <P><LI>
 * <TT>tracker=<I>host</I></TT> -- The Launcher will connect to the Tracker at
 * the given host name and port 20618.
 * <P>
 * <TT>tracker=<I>host</I>:<I>port</I></TT> -- The Launcher will connect to the
 * Tracker at the given host name and port number.
 * <P>
 * If omitted, the Launcher will connect to the tracker at host
 * <TT>localhost</TT> and port 20618.
 * <P><LI>
 * <TT>name=<I>name</I></TT> -- The Launcher's node name is the given name. If
 * omitted, the Launcher's node name is the hostname of the node where the
 * Launcher is running.
 * <P><LI>
 * <TT>cores=<I>cores</I></TT> -- The Launcher's node has the given number of
 * CPU cores. If omitted, the default is 1.
 * <P><LI>
 * <TT>gpus=<I>gpus</I></TT> -- The Launcher's node has the given number of GPU
 * accelerators. If omitted, the default is 0.
 * <P><LI>
 * <TT>command="<I>command</I>"</TT> -- The Launcher will use the given command
 * to launch the Java Virtual Machine (JVM) process for the Backend main
 * program. If omitted, the Launcher will use the command <TT>"java"</TT> to
 * launch the JVM. The command can include arguments after the command name,
 * separated by whitespace; in this case the command must be enclosed in
 * quotation marks; for example:
 * <P>
 * <TT>java edu.rit.pj2.tracker.Launcher command="java -server" ...</TT>
 * </UL>
 *
 * @author  Alan Kaminsky
 * @version 23-Mar-2014
 */
public class Launcher
	implements LauncherRef
	{

// Hidden data members.

	private static String trackerHost = "localhost";
	private static int trackerPort = 20618;
	private static String name = null;
	private static int cores = 1;
	private static int gpus = 0;
	private static String command = "java";
	private static String[] splitCommand = new String[] { "java" };

	private static Logger logger;

	private static LauncherRef launcher;
	private static TrackerSender tracker;
	private static Proxy trackerProxy;

	private static ScheduledExecutorService executor;
	private static Heartbeat heartbeat;

	private static Semaphore finishSema = new Semaphore (0);
	private static Throwable excThrown;

// Hidden constructors.

	private Launcher()
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
		for (int i = 0; i < args.length; ++ i)
			{
			if (args[i].startsWith ("tracker="))
				{
				trackerHost = args[i].substring (8);
				int j = trackerHost.indexOf (':');
				if (j >= 0)
					try
						{
						trackerPort = Integer.parseInt
							(trackerHost.substring (j + 1));
						if (0 > trackerPort || trackerPort > 65535)
							usage (String.format ("%s illegal", args[i]));
						trackerHost = trackerHost.substring (0, j);
						}
					catch (NumberFormatException exc)
						{
						usage (String.format ("%s illegal", args[i]));
						}
				}
			else if (args[i].startsWith ("name="))
				name = args[i].substring (5);
			else if (args[i].startsWith ("cores="))
				try
					{
					cores = Integer.parseInt (args[i].substring (6));
					if (cores < 1)
						usage (String.format ("%s illegal", args[i]));
					}
				catch (NumberFormatException exc)
					{
					usage (String.format ("%s illegal", args[i]));
					}
			else if (args[i].startsWith ("gpus="))
				try
					{
					gpus = Integer.parseInt (args[i].substring (5));
					if (gpus < 0)
						usage (String.format ("%s illegal", args[i]));
					}
				catch (NumberFormatException exc)
					{
					usage (String.format ("%s illegal", args[i]));
					}
			else if (args[i].startsWith ("command="))
				{
				command = args[i].substring (8);
				splitCommand = command.trim().split ("\\s+");
				}
			else
				usage (String.format ("%s illegal", args[i]));
			}

		// Log messages on standard output.
		logger = new PrintStreamLogger (System.out);

		// Create Launcher.
		launcher = new Launcher();

		// Open a connection to the Tracker.
		tracker = new TrackerSender();
		try
			{
			trackerProxy = new Proxy
				(trackerHost,
				 trackerPort,
				 tracker,
				 new LauncherReceiver
					(new ReceiverListener()
						{
						public void receiverFailed
							(Receiver receiver,
							 Throwable exc)
							{
							stopLauncher (exc);
							}
						},
					 launcher));
			}
		catch (IOException exc)
			{
			usage (String.format ("Cannot connect to Tracker at %s:%d",
				trackerHost, trackerPort));
			}

		// Launcher has started successfully.
		try
			{
			logger.log (String.format
				("Launcher started, tracker=%s:%d name=%s cores=%d gpus=%d command=\"%s\"",
				 trackerHost, trackerPort, name, cores, gpus, command));

			// Add a shutdown hook that will run when the Launcher exits.
			Runtime.getRuntime().addShutdownHook (new Thread()
				{
				public void run()
					{
					if (heartbeat != null)
						heartbeat.cancel();
					if (executor != null)
						executor.shutdownNow();
					if (trackerProxy != null)
						{
						try { tracker.launcherStopped (launcher); }
							catch (IOException exc) {}
						trackerProxy.shutdown();
						}
					logger.log ("Launcher stopped");
					}
				});

			// If name was defaulted, set it to the local host name.
			if (name == null)
				name = trackerProxy.localHost();

			// Start heartbeats with the Tracker.
			executor = Executors.newSingleThreadScheduledExecutor();
			heartbeat = new Heartbeat()
				{
				protected void sendHeartbeat() throws IOException
					{
					tracker.heartbeatFromLauncher (launcher);
					}
				protected void died()
					{
					logger.log ("Tracker heartbeat died");
					stopLauncher (null);
					}
				};
			heartbeat.enable (executor);

			// Tell the Tracker that the Launcher started.
			tracker.launcherStarted
				(launcher, new NodeProperties (name, cores, gpus));

			// All further work is done in response to incoming messages.
			}

		// Capture any exception thrown above.
		catch (Throwable exc)
			{
			stopLauncher (exc);
			}

		// Wait until the launcher is finished.
		finishSema.acquireUninterruptibly();
		if (excThrown != null)
			logger.log ("Unexpected exception", excThrown);
		System.exit (0);
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
	 */
	public void launch
		(long jobID,
		 long taskID,
		 String[] jvmFlags,
		 String jobHost,
		 int jobPort)
		{
		try
			{
			ArrayList<String> args = new ArrayList<String>();
			for (int i = 0; i < splitCommand.length; ++ i)
				args.add (splitCommand[i]);
			for (int i = 0; i < jvmFlags.length; ++ i)
				args.add (jvmFlags[i]);
			args.add ("edu.rit.pj2.Backend");
			args.add (name);
			args.add (""+jobID);
			args.add (""+taskID);
			args.add (jobHost);
			args.add (""+jobPort);
			new ProcessBuilder (args) .inheritIO() .start();
			logger.log (String.format
				("Backend launched, jobID=%d taskID=%d jobHost=%s jobPort=%d",
				 jobID, taskID, jobHost, jobPort));
			}
		catch (IOException exc)
			{
			try { tracker.launchFailed (jobID, taskID); }
				catch (IOException exc2) {}
			logger.log (String.format
				("Backend launch failed, jobID=%d taskID=%d jobHost=%s jobPort=%d",
				 jobID, taskID, jobHost, jobPort), exc);
			stopLauncher (null);
			}
		}

	/**
	 * Tell this launcher that the tracker is still alive.
	 */
	public void heartbeatFromTracker()
		{
		heartbeat.receiveHeartbeat();
		}

	/**
	 * Gracefully shut down communication with this launcher.
	 */
	public void shutdown()
		{
		}

	/**
	 * Forcibly terminate communication with this launcher.
	 */
	public void terminate()
		{
		}

// Hidden operations.

	/**
	 * Print a usage message and exit.
	 *
	 * @param  msg  Error message.
	 */
	private static void usage
		(String msg)
		{
		System.err.printf ("Launcher: %s%n", msg);
		System.err.printf ("Usage: java edu.rit.pj2.tracker.Launcher [tracker=<host>[:<port>]] [name=<name>] [cores=<cores>] [gpus=<gpus>] [command=\"<command>\"]%n");
		System.exit (1);
		}

	/**
	 * Stop the Launcher program.
	 *
	 * @param  exc  Exception that was thrown, or null if none.
	 */
	private static void stopLauncher
		(Throwable exc)
		{
		excThrown = exc;
		finishSema.release();
		}

	}
