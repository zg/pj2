//******************************************************************************
//
// File:    Tracker.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.Tracker
//
// This Java source file is copyright (C) 2015 by Alan Kaminsky. All rights
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

import edu.rit.http.HttpRequest;
import edu.rit.http.HttpResponse;
import edu.rit.http.HttpServer;
import edu.rit.pj2.Version;
import edu.rit.util.AList;
import edu.rit.util.Action;
import edu.rit.util.DList;
import edu.rit.util.DListEntry;
import edu.rit.util.Heartbeat;
import edu.rit.util.IntList;
import edu.rit.util.Logger;
import edu.rit.util.Map;
import edu.rit.util.Plural;
import edu.rit.util.Predicate;
import edu.rit.util.PrintStreamLogger;
import edu.rit.util.Set;
import edu.rit.util.Sorting;
import java.io.CharArrayWriter;
import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Date;
import java.util.Iterator;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;

/**
 * Class Tracker provides a Tracker object and main program for keeping track of
 * {@linkplain Launcher Launcher}s, {@linkplain edu.rit.pj2.Job Job}s, and
 * {@linkplain edu.rit.pj2.Task Task}s in a cluster. The Tracker prints log
 * messages on the standard output. The Tracker has a web interface for
 * displaying the cluster's status; point a web browser at the Tracker's web
 * host and port (see below).
 * <P>
 * Usage: <TT>java edu.rit.pj2.tracker.Tracker
 * [tracker=<I>host</I>[:<I>port</I>]]
 * [web=<I>host</I>[:<I>port</I>]]
 * [name=<I>name</I>]
 * [node=<I>name,cores,gpus</I>]</TT>
 * <P>
 * The following options may be specified:
 * <UL>
 * <P><LI>
 * <TT>tracker=<I>host</I></TT> &mdash; The Tracker will listen for connections
 * from Launchers and Jobs at the given host name and port 20618.
 * <P>
 * <TT>tracker=<I>host</I>:<I>port</I></TT> &mdash; The Tracker will listen for
 * connections from Launchers and Jobs at the given host name and port number.
 * <P>
 * If omitted, the Tracker will listen for connections from Launchers and Jobs
 * at host <TT>localhost</TT> and port 20618.
 * <P><LI>
 * <TT>web=<I>host</I></TT> &mdash; The Tracker will listen for web requests at
 * the given host name and port 8080.
 * <P>
 * <TT>web=<I>host</I>:<I>port</I></TT> &mdash; The Tracker will listen for web
 * requests at the given host name and port number.
 * <P>
 * If omitted, the Tracker will listen for web requests at host
 * <TT>localhost</TT> and port 8080. Specifying the <TT>web</TT> option with an
 * Internet-accessible host name is recommended, though.
 * <P><LI>
 * <TT>name=<I>name</I></TT> &mdash; The tracker will use the given cluster
 * name. If omitted, the default is <TT>"PJ2 Cluster"</TT>.
 * <P><LI>
 * <TT>node=<I>name,cores,gpus</I></TT> &mdash; If specified, tasks scheduled by
 * the Tracker will run on the Tracker's node in the Job's process; this is
 * intended for a single-node multicore parallel computer. Specify
 * <TT><I>name</I></TT> as the name of the node, <TT><I>cores</I></TT> as the
 * number of CPU cores in the node (&ge; 1), and <TT><I>gpus</I></TT> as the
 * number of GPU accelerators in the node (&ge; 0). If omitted, tasks scheduled
 * by the Tracker will run in a Backend process on a separate node; this is
 * intended for a multi-node cluster parallel computer.
 * </UL>
 * <P>
 * <B>Job lifecycle.</B>
 * When a {@linkplain edu.rit.pj2.Job Job} starts, the job contacts the Tracker,
 * which records the job's presence. The job then executes {@linkplain
 * edu.rit.pj2.Task Task}s as described below. When all the tasks have finished
 * or failed, the Tracker removes the job.
 * <P>
 * <B>Task lifecycle.</B>
 * When a {@linkplain edu.rit.pj2.Job Job} fires a {@linkplain edu.rit.pj2.Rule
 * Rule} to launch a group of one or more {@linkplain edu.rit.pj2.Task Task}s,
 * the tasks enter the <I>launched</I> state. The tasks stay in the launched
 * state until resources are available to run all the tasks in the group. When
 * resources become available, the tasks are started and enter the
 * <I>running</I> state. When each task finishes successfully, the task enters
 * the <I>finished</I> state. Alternatively, if the task fails, the task enters
 * the <I>failed</I> state.
 * <P>
 * <B>Scheduling policy.</B>
 * When a {@linkplain Launcher Launcher} starts, the launcher specifies its
 * node's available computational resources (in a {@linkplain NodeProperties}
 * object). These include the node's name, the number of CPU cores, and the
 * number of GPU accelerators. Likewise, when a {@linkplain edu.rit.pj2.Job Job}
 * fires a {@linkplain edu.rit.pj2.Rule Rule} to launch a group of {@linkplain
 * edu.rit.pj2.Task Task}s, the rule specifies the computational resources
 * required for each task (in a {@linkplain NodeProperties} object). These
 * include:
 * <UL>
 * <P><LI>
 * The name of the node on which the task must run, or {@link
 * NodeProperties#ANY_NODE_NAME ANY_NODE_NAME} if the task can run on any node.
 * By default, the task can run on any node.
 * <P><LI>
 * The number of CPU cores the task needs, or {@link NodeProperties#ALL_CORES
 * ALL_CORES} if the task will use all the cores in whatever node the task runs
 * on. The default is all cores.
 * <P><LI>
 * The number of GPU accelerators the task needs, or {@link
 * NodeProperties#ALL_GPUS ALL_GPUS} if the task will use all the GPU
 * accelerators in whatever node the task runs on. The default is no GPU
 * accelerators.
 * </UL>
 * <P>
 * For each node, the Tracker maintains a FIFO queue of launched task groups
 * that must run on that node. The Tracker also maintains a FIFO queue of
 * launched task groups that can run on any node. The Tracker schedules tasks to
 * run on nodes as follows:
 * <UL>
 * <P><LI>
 * For each node, the Tracker starts as many tasks as possible from the node's
 * launched task queue, in FIFO order, until all the tasks have started, or
 * until the first task in the queue requires more cores or GPUs than there are
 * idle cores or GPUs on the node.
 * <P><LI>
 * The Tracker then starts as many tasks as possible from the any-node launched
 * task queue, in FIFO order, on any available nodes, until all the tasks have
 * started, or until the first task in the queue requires more cores or GPUs
 * than there are idle cores or GPUs on any node. However, a task from the
 * any-node launched task queue will not be started on a node that has tasks in
 * that node's own launched task queue. Nodes are chosen in a round-robin
 * fashion so as to distribute the load among all the available nodes.
 * <P><LI>
 * The Tracker ensures that all the tasks in a task group start together. If
 * some of the tasks in a task group can be started and others cannot, the
 * Tracker will not start any of the tasks in the task group.
 * </UL>
 * <P>
 * <B>Web interface.</B>
 * The Tracker maintains a web interface, listening for web requests at the host
 * and port specified by the <TT>web</TT> option (see above). The web interface
 * displays several web pages:
 * <UL>
 * <P><LI>
 * A summary of the nodes and the jobs in the system.
 * <P><LI>
 * Details about each node, including the running and launched tasks for the
 * node.
 * <P><LI>
 * Details about each job, including the running, launched, and finished tasks
 * for the job.
 * <P><LI>
 * A list of all running, launched, and finished tasks in the system.
 * </UL>
 * <P>
 * The URL for the summary web page is
 * <TT>http://<I>host</I>:<I>port</I>/summary</TT>. The other web pages are
 * linked off the summary web page.
 * <P>
 * The web interface displays status only; it does not have the capability to
 * cancel jobs or tasks, change the order of tasks, etc. The web interface
 * refreshes itself automatically every 20 seconds.
 *
 * @author  Alan Kaminsky
 * @version 10-Jan-2015
 */
public class Tracker
	implements TrackerRef
	{

// Hidden data members.

	private static String trackerHost = "localhost";
	private static int trackerPort = 20618;
	private static String webHost = "localhost";
	private static int webPort = 8080;
	private static String clusterName = "PJ2 Cluster";
	private static NodeProperties trackerNode = null;

	private static Logger logger;

	private static TrackerRef tracker;
	private static ServerSocket serverSocket;
	private static ScheduledExecutorService executor =
		Executors.newSingleThreadScheduledExecutor();

	private static HttpServer webServer;

	private static Object lock = new Object();

	// For keeping track of Launchers.
	private static enum LauncherState { OK, FAILED };
	private static class LauncherInfo
		{
		public LauncherRef launcher; // If null, runs tasks in Job process
		public NodeProperties node;
		public LauncherState state;
		public Heartbeat heartbeat;
		public long startTime;
		public long failedTime;
		public String name;
		public int coresTotal;
		public int coresAvailable;
		public int gpusTotal;
		public int gpusAvailable;
		public IntList devnumsAvailable;
		}
	private static Map<LauncherRef,LauncherInfo> launcherMap =
		new Map<LauncherRef,LauncherInfo>();
	private static Map<String,LauncherInfo> launcherNameMap =
		new Map<String,LauncherInfo>();
	private static LauncherInfo[] launcherArray = new LauncherInfo [0];
	private static int launcherIndex = 0;
	private static Sorting.Object<LauncherInfo> launcherSorting =
		new Sorting.Object<LauncherInfo>()
			{
			public boolean comesBefore (LauncherInfo[] x, int a, int b)
				{
				return x[a].name.compareTo (x[b].name) < 0;
				}
			};

	// For keeping track of Jobs.
	private static class JobInfo
		{
		public JobRef job;
		public long jobID;
		public String user;
		public String jobHost;
		public int jobPort;
		public long startTime;
		public long minTaskStartTime;
		public Heartbeat heartbeat;
		}
	private static Map<Long,JobInfo> jobMap = new Map<Long,JobInfo>();
	private static JobInfo[] jobArray = new JobInfo [0];
	private static Sorting.Object<JobInfo> jobSorting =
		new Sorting.Object<JobInfo>()
			{
			public boolean comesBefore (JobInfo[] x, int a, int b)
				{
				return x[a].jobID < x[b].jobID;
				}
			};
	private static long nextJobID = 1L;

	// For keeping track of tasks.
	private static enum JobTaskState { Pending, Running, Finished, Failed };
	private static class JobTaskInfo
		{
		public JobInfo jobInfo;
		public TaskGroupInfo taskGroupInfo;
		public LauncherInfo launcherInfo;
		public long jobID;
		public long taskID;
		public String user;
		public NodeProperties node;
		public String[] jvmFlags;
		public JobTaskState state;
		public long launchTime;
		public long startTime;
		public long finishTime;
		public IntList devnum;
		}
	private static DList<JobTaskInfo> finishedTaskList =
		new DList<JobTaskInfo>();
	private static DList<JobTaskInfo> runningTaskList =
		new DList<JobTaskInfo>();
	private static DList<JobTaskInfo> pendingTaskList =
		new DList<JobTaskInfo>();

	// For keeping track of pending task groups.
	private static class TaskGroupInfo
		{
		public JobInfo jobInfo;
		public long jobID;
		public DList<JobTaskInfo> taskList;
		public boolean launchReady;
		}
	private static DList<TaskGroupInfo> pendingTaskGroupList =
		new DList<TaskGroupInfo>();

	// For searching a task list for a certain node name.
	private static class FindNode implements Predicate<JobTaskInfo>
		{
		private String name;
		public FindNode (String name)
			{
			this.name = name;
			}
		public boolean test (JobTaskInfo info)
			{
			return info.node.nodeName().equals (this.name);
			}
		}

	// For searching a task list for a certain job ID.
	private static class FindJob implements Predicate<JobTaskInfo>
		{
		private long jobID;
		public FindJob (long jobID)
			{
			this.jobID = jobID;
			}
		public boolean test (JobTaskInfo info)
			{
			return info.jobID == this.jobID;
			}
		}

	// For searching a task list for a certain job ID and task ID.
	private static class FindJobAndTask implements Predicate<JobTaskInfo>
		{
		private long jobID;
		private long taskID;
		public FindJobAndTask (long jobID, long taskID)
			{
			this.jobID = jobID;
			this.taskID = taskID;
			}
		public boolean test (JobTaskInfo info)
			{
			return info.jobID == this.jobID && info.taskID == this.taskID;
			}
		}

	// Matches all tasks in a task list.
	private static class AllTasks implements Predicate<JobTaskInfo>
		{
		public boolean test (JobTaskInfo info)
			{
			return true;
			}
		}

	// Statistics.
	private static Date trackerStartDate = new Date();
	private static long jobCount = 0L;
	private static long taskCount = 0L;
	private static long cpuMsec = 0L;

	// Predefined content for web interface.
	private static class ContentInfo
		{
		public String type;
		public byte[] content;
		public ContentInfo
			(String type,
			 byte[] content)
			{
			this.type = type;
			this.content = content;
			}
		}
	private static Map<String,ContentInfo> contentMap =
		new Map<String,ContentInfo>();

// Hidden constructors.

	private Tracker()
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
							usageIllegal (args[i]);
						trackerHost = trackerHost.substring (0, j);
						}
					catch (NumberFormatException exc)
						{
						usageIllegal (args[i]);
						}
				}
			else if (args[i].startsWith ("web="))
				{
				webHost = args[i].substring (4);
				int j = webHost.indexOf (':');
				if (j >= 0)
					try
						{
						webPort = Integer.parseInt
							(webHost.substring (j + 1));
						if (0 > webPort || webPort > 65535)
							usageIllegal (args[i]);
						webHost = webHost.substring (0, j);
						}
					catch (NumberFormatException exc)
						{
						usageIllegal (args[i]);
						}
				}
			else if (args[i].startsWith ("name="))
				{
				clusterName = args[i].substring (5);
				if (clusterName.length() == 0)
					usageIllegal (args[i]);
				}
			else if (args[i].startsWith ("node="))
				try
					{
					trackerNode = new NodeProperties (args[i].substring (5));
					}
				catch (IllegalArgumentException exc)
					{
					usageIllegal (args[i]);
					}
			else
				usageIllegal (args[i]);
			}

		// Log messages on standard output.
		logger = new PrintStreamLogger (System.out);

		// Create Tracker.
		tracker = new Tracker();

		// Listen for connections from Launchers and Jobs.
		try
			{
			serverSocket = new ServerSocket();
			serverSocket.bind
				(new InetSocketAddress (trackerHost, trackerPort));
			}
		catch (IOException exc)
			{
			usage (String.format
				("Cannot listen for tracker connections at %s:%d",
				 trackerHost, trackerPort));
			}

		// Set up web interface.
		try
			{
			webServer = new HttpServer
				(new InetSocketAddress (webHost, webPort), logger)
				{
				protected void process
					(HttpRequest request,
					 HttpResponse response)
					throws IOException
					{
					processWebRequest (request, response);
					}
				};
			}
		catch (IOException exc)
			{
			usage (String.format
				("Cannot listen for web connections at %s:%d",
				 webHost, webPort));
			}

		// Tracker has started successfully.
		try
			{
			logger.log (String.format
				("Tracker started, tracker=%s:%d web=%s:%d name=%s version=%s",
				 trackerHost, trackerPort, webHost, webPort, clusterName,
				 Version.PJ2_VERSION));

			// If Tracker is scheduling a single node, set up Launcher.
			if (trackerNode != null)
				tracker.launcherStarted (null, trackerNode);

			// Add predefined web content.
			for (int percent = 0; percent <= 100; percent += 5)
				addContent (String.format ("bar%02d.png", percent),
					"image/png");

			// Add a shutdown hook that will run when the Tracker exits.
			Runtime.getRuntime().addShutdownHook (new Thread()
				{
				public void run()
					{
					try { webServer.close(); } catch (IOException exc) {}
					if (executor != null)
						executor.shutdownNow();
					logger.log ("Tracker stopped");
					}
				});

			// Repeatedly accept incoming connections. If an error occurs while
			// accepting a connection, go to the catch-all exception handler.
			for (;;)
				{
				Socket socket = serverSocket.accept();

				// Attach a proxy to the incoming connection.
				try
					{
					new Proxy
						(socket,
						 new TrackerReceiver
							(new ReceiverListener()
								{
								public void receiverFailed
									(Receiver receiver,
									 Throwable exc)
									{
									// Log the error and terminate the proxy.
									logger.log
										("Exception while receiving message",
										 exc);
									receiver.terminate();
									}
								},
							 tracker));
					}

				// If an error occurs while creating the proxy, log the error,
				// close the socket, and keep running.
				catch (Throwable exc)
					{
					logger.log
						(String.format
							("Exception while creating proxy, socket=%s",
							 socket),
						exc);
					try { socket.close(); }
						catch (IOException exc2) {}
					}
				}

			// All further work is done in response to incoming messages.
			}

		// Catch-all exception handler. Log the error and exit.
		catch (Throwable exc)
			{
			logger.log ("Unexpected exception", exc);
			System.exit (0);
			}
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
	public void launcherStarted
		(LauncherRef launcher,
		 NodeProperties node)
		throws IOException
		{
		synchronized (lock)
			{
			logger.log (String.format ("Node %s launcher started, %s",
				node.nodeName(), node));

			// Set up new launcher information record.
			final LauncherInfo info = new LauncherInfo();
			info.launcher = launcher;
			info.node = node;
			info.state = LauncherState.OK;
			info.heartbeat = new Heartbeat()
				{
				protected void sendHeartbeat() throws IOException
					{
					info.launcher.heartbeatFromTracker();
					}
				protected void died()
					{
					synchronized (lock)
						{
						logger.log (String.format
							("Node %s launcher heartbeat died", info.name));
						stopLauncher (info);
						info.state = LauncherState.FAILED;
						info.failedTime = System.currentTimeMillis();
							// Keep failed launcher in launcher maps
						scheduleTaskGroups();
						}
					}
				};
			if (launcher != null)
				info.heartbeat.enable (executor);
			info.startTime = System.currentTimeMillis();
			info.name = node.nodeName();
			info.coresTotal = node.cores();
			info.coresAvailable = info.coresTotal;
			info.gpusTotal = node.gpus();
			info.gpusAvailable = info.gpusTotal;
			info.devnumsAvailable = new IntList();
			for (int i = 0; i < info.gpusTotal; ++ i)
				info.devnumsAvailable.addLast (i);

			// Remove any launchers with the same name as the new launcher.
			LauncherInfo oldinfo = launcherNameMap.get (info.name);
			if (oldinfo != null)
				{
				launcherMap.remove (oldinfo.launcher);
				launcherNameMap.remove (oldinfo.name);
				}

			// Put new launcher in launcher maps.
			launcherMap.put (info.launcher, info);
			launcherNameMap.put (info.name, info);
			makeLauncherArray();

			// Schedule pending tasks if possible.
			scheduleTaskGroups();
			}
		}

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
		throws IOException
		{
		synchronized (lock)
			{
			LauncherInfo info = launcherMap.get (launcher);
			if (info == null)
				logger.log (String.format
					("launcherStopped(%s): Unknown launcher", launcher));
			else
				{
				logger.log (String.format ("Launcher stopped, %s", info.node));
				stopLauncher (info);
				launcherMap.remove (info.launcher);
				launcherNameMap.remove (info.name);
				makeLauncherArray();
				scheduleTaskGroups();
				}
			}
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
	public void launchJob
		(JobRef job,
		 String user,
		 String jobHost,
		 int jobPort)
		throws IOException
		{
		synchronized (lock)
			{
			++ jobCount;
			long jobID = nextJobID ++;
			logger.log (String.format ("Job %d launched, user %s",
				jobID, user));

			// Set up job information record.
			final JobInfo info = new JobInfo();
			info.job = job;
			info.jobID = jobID;
			info.user = user;
			info.jobHost = jobHost;
			info.jobPort = jobPort;
			info.startTime = System.currentTimeMillis();
			info.heartbeat = new Heartbeat()
				{
				protected void sendHeartbeat() throws IOException
					{
					info.job.heartbeatFromTracker();
					}
				protected void died()
					{
					synchronized (lock)
						{
						logger.log (String.format
							("Job %d heartbeat died", info.jobID));
						stopJob (info);
						scheduleTaskGroups();
						}
					}
				};
			info.heartbeat.enable (executor);

			// Record job in job map.
			jobMap.put (jobID, info);
			makeJobArray();

			// Inform Job.
			job.jobLaunched (jobID);
			job.jobStarted();
			}
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
	public void launchTaskGroup
		(long jobID,
		 AList<TaskSpecInfo> taskGroup)
		throws IOException
		{
		synchronized (lock)
			{
			final JobInfo jobInfo = jobMap.get (jobID);
			if (jobInfo == null)
				logger.log (String.format ("launchTaskGroup(%d): Unknown job",
					jobID));
			else
				{
				final TaskGroupInfo tgInfo = new TaskGroupInfo();
				tgInfo.jobInfo = jobInfo;
				tgInfo.jobID = jobID;
				tgInfo.taskList = new DList<JobTaskInfo>();
				taskGroup.forEachItemDo (new Action<TaskSpecInfo>()
					{
					public void run (TaskSpecInfo spec)
						{
						tgInfo.taskList.addLast
							(launchTask (jobInfo, tgInfo, spec.taskID,
								spec.node, spec.jvmFlags));
						}
					});
				tgInfo.launchReady = false;
				pendingTaskGroupList.addLast (tgInfo);
				scheduleTaskGroups();
				}
			}
		}

	/**
	 * Tell this tracker to launch the given task in the given job.
	 *
	 * @param  jobInfo        Job info object.
	 * @param  taskGroupInfo  Task group info object.
	 * @param  taskID         Task ID.
	 * @param  node           Node requirements for the task.
	 * @param  jvmFlags       JVM flags for the task.
	 *
	 * @return  Job task info object.
	 */
	private JobTaskInfo launchTask
		(JobInfo jobInfo,
		 TaskGroupInfo taskGroupInfo,
		 long taskID,
		 NodeProperties node,
		 String[] jvmFlags)
		{
		synchronized (lock)
			{
			logger.log (String.format ("Job %d task %d launched, %s",
				jobInfo.jobID, taskID, node));
			++ taskCount;
			JobTaskInfo info = new JobTaskInfo();
			info.jobInfo = jobInfo;
			info.taskGroupInfo = taskGroupInfo;
			info.jobID = jobInfo.jobID;
			info.taskID = taskID;
			info.user = jobInfo.user;
			info.node = new NodeProperties (node);
			info.jvmFlags = jvmFlags;
			info.state = JobTaskState.Pending;
			info.launchTime = System.currentTimeMillis();
			info.devnum = new IntList();
			pendingTaskList.addLast (info);
			return info;
			}
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
	public void launchFailed
		(long jobID,
		 long taskID)
		throws IOException
		{
		synchronized (lock)
			{
			DListEntry<JobTaskInfo> entry = runningTaskList.find
				(new FindJobAndTask (jobID, taskID));
			if (entry == null)
				logger.log (String.format ("launchFailed(%d,%d): Unknown task",
					jobID, taskID));
			else
				{
				logger.log (String.format ("Job %d task %d launch failed",
					jobID, taskID));
				failTask (entry, new LaunchException ("Launch failed"));
				scheduleTaskGroups();
				}
			}
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
	public void taskDone
		(long jobID,
		 long taskID)
		throws IOException
		{
		synchronized (lock)
			{
			DListEntry<JobTaskInfo> entry = runningTaskList.find
				(new FindJobAndTask (jobID, taskID));
			if (entry == null)
				logger.log (String.format ("taskDone(%d,%d): Unknown task",
					jobID, taskID));
			else
				{
				logger.log (String.format ("Job %d task %d done",
					jobID, taskID));
				stopTask (entry);
				scheduleTaskGroups();
				}
			}
		}

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
		throws IOException
		{
		synchronized (lock)
			{
			JobInfo info = jobMap.get (jobID);
			if (info == null)
				logger.log (String.format ("jobDone(%d): Unknown job", jobID));
			else
				{
				logger.log (String.format ("Job %d done", jobID));
				stopJob (info);
				scheduleTaskGroups();
				}
			}
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
	public void stopJob
		(long jobID,
		 String msg)
		throws IOException
		{
		synchronized (lock)
			{
			JobInfo info = jobMap.get (jobID);
			if (info == null)
				logger.log (String.format ("stopJob(%d): Unknown job", jobID));
			else
				{
				logger.log (String.format ("Job %d stopped %s", jobID, msg));
				stopJob (info);
				scheduleTaskGroups();
				}
			}
		}

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
		throws IOException
		{
		synchronized (lock)
			{
			JobInfo info = jobMap.get (jobID);
			if (info == null)
				logger.log (String.format
					("heartbeatFromJob(%d): Unknown job", jobID));
			else
				info.heartbeat.receiveHeartbeat();
			}
		}

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
		throws IOException
		{
		synchronized (lock)
			{
			LauncherInfo info = launcherMap.get (launcher);
			if (info == null)
				logger.log (String.format
					("heartbeatFromLauncher(%s): Unknown launcher", launcher));
			else
				info.heartbeat.receiveHeartbeat();
			}
		}

	/**
	 * Gracefully shut down communication with this tracker.
	 */
	public void shutdown()
		{
		}

	/**
	 * Forcibly terminate communication with this tracker.
	 */
	public void terminate()
		{
		}

// Hidden operations.

	/**
	 * Print an illegal argument usage message and exit.
	 *
	 * @param  arg  Argument string.
	 */
	private static void usageIllegal
		(String arg)
		{
		usage (arg + " illegal");
		}

	/**
	 * Print a usage message and exit.
	 *
	 * @param  msg  Error message.
	 */
	private static void usage
		(String msg)
		{
		System.err.printf ("Tracker: %s%n", msg);
		System.err.printf ("Usage: java edu.rit.pj2.tracker.Tracker [tracker=<host>[:<port>]] [web=<host>[:<port>]] [name=<name>] [node=<name>,<cores>,<gpus>]%n");
		System.exit (1);
		}

	/**
	 * Make an array of launcher information records sorted in ascending
	 * lexicographic order of node name.
	 */
	private static void makeLauncherArray()
		{
		launcherArray = launcherMap.valuesToArray
			(new LauncherInfo [launcherMap.size()]);
		Sorting.sort (launcherArray, launcherSorting);
		}

	/**
	 * Make an array of job information records sorted in ascending order of job
	 * ID.
	 */
	private static void makeJobArray()
		{
		jobArray = jobMap.valuesToArray (new JobInfo [jobMap.size()]);
		Sorting.sort (jobArray, jobSorting);
		}

	/**
	 * Stop the given Launcher.
	 *
	 * @param  info  Launcher information record.
	 */
	private static void stopLauncher
		(LauncherInfo info)
		{
		synchronized (lock)
			{
			info.heartbeat.cancel();
			info.launcher.terminate();
			}
		}

	/**
	 * Stop the given Job.
	 *
	 * @param  info  Job information record.
	 */
	private static void stopJob
		(JobInfo info)
		{
		synchronized (lock)
			{
			info.heartbeat.cancel();
			info.job.terminate();

			// Remove all task groups for this job from the system.
			DListEntry<TaskGroupInfo> pp, qq;
			pp = pendingTaskGroupList.first();
			while (pp != null)
				{
				qq = pp.succ();
				if (pp.item().jobID == info.jobID)
					pp.remove();
				pp = qq;
				}

			// Remove all tasks for this job from the system.
			DListEntry<JobTaskInfo> p, q;
			p = pendingTaskList.first();
			while (p != null)
				{
				q = p.succ();
				if (p.item().jobID == info.jobID)
					p.remove();
				p = q;
				}
			p = runningTaskList.first();
			while (p != null)
				{
				q = p.succ();
				if (p.item().jobID == info.jobID)
					{
					releaseResources (p.item());
					p.remove();
					}
				p = q;
				}
			p = finishedTaskList.first();
			while (p != null)
				{
				q = p.succ();
				if (p.item().jobID == info.jobID)
					p.remove();
				p = q;
				}

			// Remove job from the system.
			jobMap.remove (info.jobID);
			makeJobArray();
			}
		}

	/**
	 * Stop the task contained in the given task list entry.
	 *
	 * @param  entry  Task list entry.
	 */
	private static void stopTask
		(DListEntry<JobTaskInfo> entry)
		{
		// Idle task's resources.
		JobTaskInfo info = entry.item();
		releaseResources (info);

		// Update task state and task lists.
		entry.remove();
		info.state = JobTaskState.Finished;
		info.finishTime = System.currentTimeMillis();
		finishedTaskList.addLast (entry);

		// Update CPU seconds statistic.
		if (info.startTime != 0L)
			cpuMsec += (info.finishTime - info.startTime)*info.node.cores();
		}

	/**
	 * Fail the task contained in the given task list entry.
	 *
	 * @param  entry  Task list entry.
	 * @param  exc    Exception that was thrown.
	 */
	private static void failTask
		(DListEntry<JobTaskInfo> entry,
		 Throwable exc)
		{
		// Inform Job.
		JobTaskInfo info = entry.item();
		try { info.jobInfo.job.taskFailed (info.taskID, exc); }
			catch (IOException exc2) {}

		stopTask (entry);
		}

	/**
	 * Schedule as many pending task groups as possible.
	 */
	private static void scheduleTaskGroups()
		{
		DListEntry<TaskGroupInfo> p, q;
		DListEntry<JobTaskInfo> pp, qq;
		Exception exc;
		JobTaskInfo taskInfo;
		String name;

		// PHASE 1. RESOURCE CHECK

		// For each pending task group, ensure resources exist for each task in
		// the group; if not, fail all tasks in the group.
		p = pendingTaskGroupList.first();
		while (p != null)
			{
			q = p.succ();
			exc = resourcesExist (p.item());
			if (exc != null)
				{
				failTaskGroup (p.item(), exc);
				p.remove();
				}
			p = q;
			}

		// PHASE 2. RESOURCE RESERVATION

		// Set of blocked node names (those that have pending tasks that must
		// run on the node).
		Set<String> emptySet = new Set<String>();
		Set<String> blockedNodes = new Set<String>();

		// Phase 2 first pass: Attempt to reserve resources for pending tasks
		// that must run on a certain node.
		pp = pendingTaskList.first();
		while (pp != null)
			{
			qq = pp.succ();
			taskInfo = pp.item();
			name = taskInfo.node.nodeName();
			if (! name.equals (NodeProperties.ANY_NODE_NAME))
				{
				if (reserveResources (taskInfo, emptySet))
					blockedNodes.add (name);
				}
			pp = qq;
			}

		// Phase 2 second pass: Attempt to reserve resources for pending tasks
		// that can run on any node. Omit blocked nodes.
		pp = pendingTaskList.first();
		while (pp != null)
			{
			qq = pp.succ();
			taskInfo = pp.item();
			name = taskInfo.node.nodeName();
			if (name.equals (NodeProperties.ANY_NODE_NAME))
				{
				if (! reserveResources (taskInfo, blockedNodes))
					qq = null;
				}
			pp = qq;
			}

		// Phase 2 third pass: Determine which task groups are launch ready;
		// i.e., have resources reserved for all tasks in the group.
		p = pendingTaskGroupList.first();
		while (p != null)
			{
			checkLaunchReady (p.item());
			p = p.succ();
			}

		// PHASE 3. TASK LAUNCH

		// Phase 3 first pass: For each pending task, if its task group is
		// launch ready, run the task; otherwise, release the task's resources.
		pp = pendingTaskList.first();
		while (pp != null)
			{
			qq = pp.succ();
			taskInfo = pp.item();
			if (taskInfo.taskGroupInfo.launchReady)
				{
				runTask (taskInfo);
				pp.remove();
				runningTaskList.addLast (pp);
				}
			else
				releaseResources (taskInfo);
			pp = qq;
			}

		// Phase 3 second pass: Remove all launch ready task groups from the
		// pending task group list.
		p = pendingTaskGroupList.first();
		while (p != null)
			{
			q = p.succ();
			if (p.item().launchReady)
				p.remove();
			p = q;
			}
		}

	/**
	 * Determine if resources exist for all tasks in the given task group.
	 *
	 * @param  group  Task group info object.
	 *
	 * @return  If sufficient resources exist, null is returned. If sufficient
	 *          resources do not exist, a LaunchException giving the reason is
	 *          returned.
	 */
	private static Exception resourcesExist
		(TaskGroupInfo group)
		{
		LauncherInfo linfo;
		JobTaskInfo tinfo;
		int index, i;
		DListEntry<JobTaskInfo> p;
		String nameNeeded;
		int coresNeeded;
		int gpusNeeded;

		// Determine first and last task IDs in task group.
		p = group.taskList.first();
		long firstTaskID = p.item().taskID;
		long lastTaskID = firstTaskID;
		while (p != null)
			{
			lastTaskID = p.item().taskID;
			p = p.succ();
			}

		// Special case if there are no launchers.
		int numLaunchers = launcherArray.length;
		if (numLaunchers == 0)
			return getLaunchException (firstTaskID, lastTaskID);

		// Make copies of the Launcher info objects, in order to make tentative
		// assignments of resources to tasks.
		LauncherInfo[] launcherInfo = new LauncherInfo [numLaunchers];
		for (i = 0; i < numLaunchers; ++ i)
			{
			LauncherInfo launcherArray_i = launcherArray[i];
			linfo = launcherInfo[i] = new LauncherInfo();
			linfo.state = launcherArray_i.state;
			linfo.name = launcherArray_i.name;
			linfo.coresTotal = launcherArray_i.coresTotal;
			linfo.coresAvailable = launcherArray_i.coresTotal;
			linfo.gpusTotal = launcherArray_i.gpusTotal;
			linfo.gpusAvailable = launcherArray_i.gpusTotal;
			}
		index = 0;

		// Try to find sufficient resources for each task in the task group.
		p = group.taskList.first();
		while (p != null)
			{
			// Determine task's resource needs.
			tinfo = p.item();
			nameNeeded = tinfo.node.nodeName();
			coresNeeded = tinfo.node.cores();
			gpusNeeded = tinfo.node.gpus();

			// Try to find a node with sufficient resources for the task.
			i = index;
			linfo = null;
			do
				{
				linfo = launcherInfo[i];
				i = (i + 1) % numLaunchers;
				if (linfo.state == LauncherState.OK &&
						nameSuffices (nameNeeded, linfo) &&
						coresSuffice (coresNeeded, linfo) &&
						gpusSuffice (gpusNeeded, linfo))
					break;
				linfo = null;
				}
			while (i != index);
			index = i;

			// If we found a node, update its resource usage.
			if (linfo != null)
				{
				if (coresNeeded == NodeProperties.ALL_CORES)
					linfo.coresAvailable = 0;
				else
					linfo.coresAvailable -= coresNeeded;
				if (gpusNeeded == NodeProperties.ALL_GPUS)
					linfo.gpusAvailable = 0;
				else
					linfo.gpusAvailable -= gpusNeeded;
				}

			// If we didn't find a node, return failure.
			else
				return getLaunchException (firstTaskID, lastTaskID);

			// Check next task.
			p = p.succ();
			}

		// We found resources for all tasks. Return success.
		return null;
		}

	/**
	 * Returns a LaunchException with the proper error message.
	 *
	 * @param  firstTaskID  First task ID.
	 * @param  lastTaskID   Last task ID.
	 *
	 * @return  LaunchException.
	 */
	private static Exception getLaunchException
		(long firstTaskID,
		 long lastTaskID)
		{
		if (firstTaskID == lastTaskID)
			return new LaunchException (String.format
				("Insufficient resources to launch task %d",
				 firstTaskID));
		else
			return new LaunchException (String.format
				("Insufficient resources to launch tasks %d..%d",
				 firstTaskID, lastTaskID));
		}

	/**
	 * Fail all pending tasks in the given task group.
	 *
	 * @param  group  Task group info object.
	 * @param  exc    Exception giving reason for failure.
	 */
	private static void failTaskGroup
		(TaskGroupInfo group,
		 Exception exc)
		{
		DListEntry<JobTaskInfo> p = group.taskList.first();
		while (p != null)
			{
			DListEntry<JobTaskInfo> q = pendingTaskList.find
				(new FindJobAndTask (p.item().jobID, p.item().taskID));
			if (q != null)
				failTask (q, exc);
			p = p.succ();
			}
		}

	/**
	 * Find a node that has enough idle resources to run the given task, and
	 * reserve resources for that task.
	 *
	 * @param  taskInfo      Job task info object.
	 * @param  blockedNodes  Set of blocked nodes. These nodes are omitted from
	 *                       the search.
	 *
	 * @return  True if resources were reserved, false otherwise.
	 */
	private static boolean reserveResources
		(JobTaskInfo taskInfo,
		 Set<String> blockedNodes)
		{
		// Special case.
		if (launcherArray.length == 0) return false;

		String nameNeeded = taskInfo.node.nodeName();
		int coresNeeded = taskInfo.node.cores();
		int gpusNeeded = taskInfo.node.gpus();
		LauncherInfo launcherInfo = null;

		// Search for nodes (launchers) in a round-robin fashion, picking up
		// where we left off the last time.
		int i = launcherIndex;
		int numLaunchers = launcherArray.length;
		do
			{
			launcherInfo = launcherArray[i];
			i = (i + 1) % numLaunchers;
			if (launcherInfo.state == LauncherState.OK &&
					! blockedNodes.contains (launcherInfo.name) &&
					nameSuffices (nameNeeded, launcherInfo) &&
					coresSuffice (coresNeeded, launcherInfo) &&
					gpusSuffice (gpusNeeded, launcherInfo))
				break;
			launcherInfo = null;
			}
		while (i != launcherIndex);
		launcherIndex = i;

		// If we found a suitable node, reserve resources.
		if (launcherInfo != null)
			{
			taskInfo.launcherInfo = launcherInfo;
			if (coresNeeded == NodeProperties.ALL_CORES)
				launcherInfo.coresAvailable = 0;
			else
				launcherInfo.coresAvailable -= coresNeeded;
			if (gpusNeeded == NodeProperties.ALL_GPUS)
				{
				launcherInfo.gpusAvailable = 0;
				for (int j = 0; j < launcherInfo.gpusTotal; ++ j)
					taskInfo.devnum.addLast
						(launcherInfo.devnumsAvailable.removeFirst());
				}
			else
				{
				launcherInfo.gpusAvailable -= gpusNeeded;
				for (int j = 0; j < gpusNeeded; ++ j)
					taskInfo.devnum.addLast
						(launcherInfo.devnumsAvailable.removeFirst());
				}
			return true;
			}
		else
			{
			taskInfo.launcherInfo = null;
			return false;
			}
		}

	private static boolean nameSuffices
		(String nameNeeded,
		 LauncherInfo info)
		{
		return nameNeeded.equals (NodeProperties.ANY_NODE_NAME) ||
			nameNeeded.equals (info.name);
		}

	private static boolean coresSuffice
		(int coresNeeded,
		 LauncherInfo info)
		{
		return
			(coresNeeded == NodeProperties.ALL_CORES &&
				info.coresAvailable == info.coresTotal) ||
			(coresNeeded != NodeProperties.ALL_CORES &&
				coresNeeded <= info.coresAvailable);
		}

	private static boolean gpusSuffice
		(int gpusNeeded,
		 LauncherInfo info)
		{
		return
			(gpusNeeded == NodeProperties.ALL_GPUS &&
				info.gpusAvailable == info.gpusTotal) ||
			(gpusNeeded != NodeProperties.ALL_GPUS &&
				gpusNeeded <= info.gpusAvailable);
		}

	/**
	 * Determine if the given task group is launch ready; i.e., resources are
	 * reserved for all its tasks.
	 *
	 * @param  group  Task group info object.
	 */
	private static void checkLaunchReady
		(TaskGroupInfo group)
		{
		group.launchReady = true;
		DListEntry<JobTaskInfo> p = group.taskList.first();
		while (p != null && group.launchReady)
			{
			group.launchReady = p.item().launcherInfo != null;
			p = p.succ();
			}
		}

	/**
	 * Run the given task.
	 *
	 * @param  taskInfo  Job task info object.
	 */
	private static void runTask
		(JobTaskInfo taskInfo)
		{
		LauncherInfo launcherInfo = taskInfo.launcherInfo;
		logger.log (String.format ("Job %d task %d started on node %s",
			taskInfo.jobID, taskInfo.taskID, launcherInfo.name));

		// Update task information.
		taskInfo.node.nodeName (launcherInfo.name);
		if (taskInfo.node.cores() == NodeProperties.ALL_CORES)
			taskInfo.node.cores (launcherInfo.coresTotal);
		if (taskInfo.node.gpus() == NodeProperties.ALL_GPUS)
			taskInfo.node.gpus (launcherInfo.gpusTotal);
		taskInfo.state = JobTaskState.Running;
		taskInfo.startTime = System.currentTimeMillis();
		if (taskInfo.jobInfo.minTaskStartTime == 0L)
			taskInfo.jobInfo.minTaskStartTime = taskInfo.startTime;
		int ngpus = taskInfo.devnum.size();

		// If task is to run in a Backend process ...
		if (launcherInfo.launcher != null)
			{
			// Inform Launcher.
			try
				{
				launcherInfo.launcher.launch
					(taskInfo.jobID,
					 taskInfo.taskID,
					 taskInfo.jvmFlags,
					 taskInfo.jobInfo.jobHost,
					 taskInfo.jobInfo.jobPort);
				}
			catch (IOException exc)
				{
				}

			// Inform Job.
			try
				{
				taskInfo.jobInfo.job.taskLaunching
					(taskInfo.taskID,
					 taskInfo.devnum.toArray (new int [ngpus]),
					 false);
				}
			catch (IOException exc)
				{
				}
			}

		// If task is to run in the Job's process ...
		else
			{
			// Inform Job.
			try
				{
				taskInfo.jobInfo.job.taskLaunching
					(taskInfo.taskID,
					 taskInfo.devnum.toArray (new int [ngpus]),
					 true);
				}
			catch (IOException exc)
				{
				}
			}
		}

	/**
	 * Release the given task's reserved resources, if any.
	 *
	 * @param  taskInfo  Task information record.
	 */
	private static void releaseResources
		(JobTaskInfo taskInfo)
		{
		LauncherInfo launcherInfo = taskInfo.launcherInfo;
		if (launcherInfo != null)
			{
			if (taskInfo.node.cores() == NodeProperties.ALL_CORES)
				launcherInfo.coresAvailable = launcherInfo.coresTotal;
			else
				launcherInfo.coresAvailable += taskInfo.node.cores();
			if (taskInfo.node.cores() == NodeProperties.ALL_CORES)
				{
				launcherInfo.gpusAvailable = launcherInfo.gpusTotal;
				for (int i = 0; i < launcherInfo.gpusTotal; ++ i)
					launcherInfo.devnumsAvailable.addLast
						(taskInfo.devnum.get (i));
				}
			else
				{
				launcherInfo.gpusAvailable += taskInfo.node.gpus();
				for (int i = 0; i < taskInfo.node.gpus(); ++ i)
					launcherInfo.devnumsAvailable.addLast
						(taskInfo.devnum.get (i));
				}
			taskInfo.launcherInfo = null;
			}
		}

// Hidden operations for processing web requests.

	/**
	 * Add the given predefined content.
	 *
	 * @param  name  Content name.
	 * @param  type  Content type.
	 */
	private static void addContent
		(String name,
		 String type)
		{
		try
			{
			InputStream in = Tracker.class.getClassLoader()
				.getResourceAsStream (name);
			if (in == null) return;
			BufferedInputStream bis = new BufferedInputStream (in);
			ByteArrayOutputStream baos = new ByteArrayOutputStream();
			int b;
			while ((b = bis.read()) != -1)
				baos.write (b);
			bis.close();
			contentMap.put ("/"+name,
				new ContentInfo (type, baos.toByteArray()));
			}
		catch (Throwable exc)
			{
			}
		}

	/**
	 * Process the given web request.
	 *
	 * @param  request   Web request.
	 * @param  response  Web response.
	 */
	private static void processWebRequest
		(HttpRequest request,
		 HttpResponse response)
		throws IOException
		{
		long now = System.currentTimeMillis();
		CharArrayWriter body = new CharArrayWriter();
		PrintWriter bodyout = new PrintWriter (body);
		HttpResponse.Status status = HttpResponse.Status.STATUS_200_OK;
		ContentInfo info = null;
		String pageUrl = null;

		// Reject an invalid HTTP request.
		if (! request.isValid())
			status = HttpResponse.Status.STATUS_400_BAD_REQUEST;

		// Reject all methods except GET.
		else if (! request.getMethod().equals (HttpRequest.GET_METHOD))
			status = HttpResponse.Status.STATUS_501_NOT_IMPLEMENTED;

		// Attempt to process a predefined content URI.
		else if ((info = contentMap.get (request.getUri())) != null)
			response.setContent (info.type, info.content);

		// Attempt to parse and process URI; reject a bad URI.
		else if ((pageUrl = processUri (bodyout, now, request.getUri()))
				== null)
			status = HttpResponse.Status.STATUS_404_NOT_FOUND;

		// Send the response.
		if (info == null)
			{
			PrintWriter out = response.getPrintWriter();
			if (status == HttpResponse.Status.STATUS_200_OK)
				{
				printHeader (out, now, pageUrl);
				bodyout.flush();
				out.write (body.toCharArray());
				printTrailer (out);
				}
			else
				{
				response.setStatusCode (status);
				response.setContentType ("text/plain");
				out.printf ("%s%n", status);
				}
			}
		response.close();
		}

	/**
	 * Print the web page header.
	 *
	 * @param  out  Print writer on which to print.
	 * @param  now  Current time and date.
	 * @param  url  URL of the web page.
	 */
	private static void printHeader
		(PrintWriter out,
		 long now,
		 String url)
		{
		out.printf ("<HTML>%n");
		out.printf ("<HEAD>%n");
		out.printf ("<TITLE>%s</TITLE>%n", clusterName);
		out.printf ("<META HTTP-EQUIV=\"refresh\" CONTENT=\"20;url=%s\">%n",
			url);
		out.printf ("<STYLE TYPE=\"text/css\">%n");
		out.printf ("<!--%n");
		out.printf ("* {font-family: Arial, Helvetica, Sans-Serif;}%n");
		out.printf ("body {font-size: small;}%n");
		out.printf ("h1 {font-size: 160%%; font-weight: bold;}%n");
		out.printf ("h2 {font-size: 130%%; font-weight: bold;}%n");
		out.printf ("table {font-size: 100%%;}%n");
		out.printf ("input {font-size: 100%%;}%n");
		out.printf ("-->%n");
		out.printf ("</STYLE>%n");
		out.printf ("</HEAD>%n");
		out.printf ("<BODY>%n");
		out.printf ("<H1>%s</H1>%n", clusterName);
		out.printf ("<P>%n");
		out.printf ("<FORM ACTION=\"%s\" METHOD=\"get\">%n", url);
		out.printf ("<INPUT TYPE=\"submit\" NAME=\"btn\" VALUE=\"Summary\">%n");
		out.printf ("<INPUT TYPE=\"submit\" NAME=\"btn\" VALUE=\"Tasks\">%n");
		out.printf ("<INPUT TYPE=\"submit\" NAME=\"btn\" VALUE=\"Refresh\">%n");
		out.printf ("%s -- %s%n", new Date (now), Version.PJ2_VERSION);
		out.printf ("</FORM>%n");
		out.printf ("<P><HR>%n");
		}

	/**
	 * Print the web page trailer.
	 *
	 * @param  out  Print writer on which to print.
	 */
	private static void printTrailer
		(PrintWriter out)
		{
		out.printf ("<P><HR>%n");
		out.printf ("<P>%n");
		out.printf ("<TABLE BORDER=0 CELLPADDING=0 CELLSPACING=0>%n");
		out.printf ("<TR>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%n");
		out.printf ("Tracker host and port:&nbsp;&nbsp;%n");
		out.printf ("</TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%n");
		out.printf ("%s:%d%n", trackerHost, trackerPort);
		out.printf ("</TD>%n");
		out.printf ("</TR>%n");
		out.printf ("<TR>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%n");
		out.printf ("Tracker web interface:&nbsp;&nbsp;%n");
		out.printf ("</TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%n");
		out.printf ("<A HREF=\"%s\">%s</A>%n", summaryUrl(), summaryUrl());
		out.printf ("</TD>%n");
		out.printf ("</TR>%n");
		out.printf ("<TR>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%n");
		out.printf ("Powered by Parallel Java 2:&nbsp;&nbsp;%n");
		out.printf ("</TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%n");
		out.printf ("<A HREF=\"http://www.cs.rit.edu/~ark/pj2.shtml\">http://www.cs.rit.edu/~ark/pj2.shtml</A>%n");
		out.printf ("</TD>%n");
		out.printf ("</TR>%n");
		out.printf ("<TR>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%n");
		out.printf ("Developed by Alan Kaminsky:&nbsp;&nbsp;%n");
		out.printf ("</TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%n");
		out.printf ("<A HREF=\"http://www.cs.rit.edu/~ark/\">http://www.cs.rit.edu/~ark/</A>%n");
		out.printf ("</TD>%n");
		out.printf ("</TR>%n");
		out.printf ("</TABLE>%n");
		out.printf ("</BODY>%n");
		out.printf ("</HTML>%n");
		}

	/**
	 * Process the given URI.
	 *
	 * @param  out  Print writer for HTML body.
	 * @param  now  Current time and date.
	 * @param  uri  Request URI.
	 *
	 * @return  If the request URI is good, the web page URL is returned,
	 *          otherwise null is returned.
	 */
	private static String processUri
		(PrintWriter out,
		 long now,
		 String uri)
		{
		// Split URI around slashes and question marks.
		String[] piece = uri.substring(1).split ("[/?]+");
//System.out.printf ("*** processUri(\"%s\"), %d pieces,", uri, piece.length); for (int i = 0; i < piece.length; ++ i) System.out.printf (" \"%s\"", piece[i]); System.out.println();

		synchronized (lock)
			{
			int n = piece.length;
			if (n == 0)
				return null;
			else if (piece[n-1].equals ("btn=Summary"))
				return displaySummary (out, now);
			else if (piece[n-1].equals ("btn=Tasks"))
				return displayTaskList (out, now);
			else if (piece[0].equals ("summary"))
				return displaySummary (out, now);
			else if (piece[0].startsWith ("node="))
				return displayNodeDetails (out, now, piece[0].substring (5));
			else if (piece[0].startsWith ("job="))
				return displayJobDetails (out, now, piece[0].substring (4));
			else if (piece[0].equals ("tasks"))
				return displayTaskList (out, now);
			else
				return null;
			}
		}

	/**
	 * Display the summary web page.
	 *
	 * @param  out  Print writer for HTML body.
	 * @param  now  Current time and date.
	 *
	 * @return  Web page URL.
	 */
	private static String displaySummary
		(PrintWriter out,
		 long now)
		{
		int row;
		out.printf ("<H2>Summary</H2>%n");

		// Print node summary.
		row = 0;
		out.printf ("<H3>Nodes</H3>%n");
		out.printf ("<P>%n");
		out.printf ("<TABLE BORDER=1 CELLPADDING=2 CELLSPACING=0>%n");
		out.printf ("<TR>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%n");
		out.printf ("<TABLE BORDER=0 CELLPADDING=2 CELLSPACING=0>%n");
		out.printf ("<TR BGCOLOR=\"%s\">%n", stripeColor (row ++));
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;Node&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;Status&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;Cores in use&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;GPUs in use&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;Tasks&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;Details&nbsp;</I></TD>%n");
		out.printf ("</TR>%n");
		for (LauncherInfo info : launcherArray)
			{
			int[] count = countTasksForNode (info.name);
			out.printf ("<TR BGCOLOR=\"%s\">%n", stripeColor (row ++));
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;%s&nbsp;</TD>%n",
				info.name);
			if (info.state == LauncherState.OK)
				{
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;OK&nbsp;</TD>%n");
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;<IMG SRC=\"%s\"> %d of %d&nbsp;</TD>%n",
					inUseBarUrl (info.coresTotal, info.coresAvailable),
					info.coresTotal - info.coresAvailable,
					info.coresTotal);
				if (info.gpusTotal == 0)
					out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;n/a&nbsp;</TD>%n");
				else
					out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;<IMG SRC=\"%s\"> %d of %d&nbsp;</TD>%n",
						inUseBarUrl (info.gpusTotal, info.gpusAvailable),
						info.gpusTotal - info.gpusAvailable,
						info.gpusTotal);
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;%d running, %d pending&nbsp;</TD>%n",
					count[0], count[1]);
				}
			else
				{
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\" BGCOLOR=\"#FF0000\">&nbsp;<FONT COLOR=\"#FFFFFF\"><B>FAILED</B></FONT>&nbsp;</TD>%n");
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;&nbsp;</TD>%n");
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;&nbsp;</TD>%n");
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;&nbsp;</TD>%n");
				}
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;<A HREF=\"%s\">Details</A>&nbsp;</TD>%n",
				nodeDetailsUrl (info.name));
			out.printf ("</TR>%n");
			}
		out.printf ("</TABLE>%n");
		out.printf ("</TD>%n");
		out.printf ("</TR>%n");
		out.printf ("</TABLE>%n");

		// Print job summary.
		row = 0;
		out.printf ("<H3>Jobs</H3>%n");
		out.printf ("<P>%n");
		out.printf ("<TABLE BORDER=1 CELLPADDING=2 CELLSPACING=0>%n");
		out.printf ("<TR>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%n");
		out.printf ("<TABLE BORDER=0 CELLPADDING=2 CELLSPACING=0>%n");
		out.printf ("<TR BGCOLOR=\"%s\">%n", stripeColor (row ++));
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;Job&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;User&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;Started&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;Status&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;Tasks&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;Details&nbsp;</I></TD>%n");
		out.printf ("</TR>%n");
		for (JobInfo info : jobArray)
			{
			int[] count = countTasksForJob (info.jobID);
			out.printf ("<TR BGCOLOR=\"%s\">%n", stripeColor (row ++));
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;%d&nbsp;</TD>%n",
				info.jobID);
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;%s&nbsp;</TD>%n",
				info.user);
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;%s&nbsp;</TD>%n",
				new Date (info.startTime));
			if (info.minTaskStartTime == 0L)
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;Waiting %d sec&nbsp;</TD>%n",
					(now - info.startTime + 500L)/1000L);
			else
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;Running %d sec&nbsp;</TD>%n",
					(now - info.minTaskStartTime + 500L)/1000L);
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;%d running, %d pending, %d finished&nbsp;</TD>%n",
				count[0], count[1], count[2]);
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;<A HREF=\"%s\">Details</A>&nbsp;</TD>%n",
				jobDetailsUrl (info.jobID));
			out.printf ("</TR>%n");
			}
		out.printf ("</TABLE>%n");
		out.printf ("</TD>%n");
		out.printf ("</TR>%n");
		out.printf ("</TABLE>%n");

		// Print statistics.
		out.printf ("<P>%n");
		out.printf ("%s served%n", new Plural (jobCount, "job"));
		out.printf ("<BR>%s served%n", new Plural (taskCount, "task"));
		long cpuSec = (cpuMsec + 999L)/1000L;
		if (cpuSec < 1000L)
			out.printf ("<BR>%s served%n", new Plural (cpuSec, "CPU second"));
		else if (cpuSec < 1000000L)
			out.printf ("<BR>%d thousand CPU seconds served%n",
				cpuSec/1000L);
		else
			out.printf ("<BR>%.1f million CPU seconds served%n",
				cpuSec/1000000.0);
		out.printf ("<BR>Since %s%n", trackerStartDate);

		return summaryUrl();
		}

	/**
	 * Display the node details web page.
	 *
	 * @param  out   Print writer for HTML body.
	 * @param  now   Current time and date.
	 * @param  name  Node name.
	 *
	 * @return  Web page URL.
	 */
	private static String displayNodeDetails
		(PrintWriter out,
		 long now,
		 String name)
		{
		int[] count = countTasksForNode (name);
		out.printf ("<H2>Node %s Details</H2>%n", name);
		out.printf ("<P>%n");
		LauncherInfo info = launcherNameMap.get (name);
		if (info == null)
			out.printf ("Node %s nonexistent%n", name);
		else
			{
			out.printf ("<TABLE BORDER=0 CELLPADDING=0 CELLSPACING=0>%n");
			out.printf ("<TR>%n");
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">Name:</TD>%n");
			out.printf ("<TD WIDTH=10> </TD>%n");
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%s</TD>%n", name);
			out.printf ("</TR>%n");
			out.printf ("<TR>%n");
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">Status:</TD>%n");
			out.printf ("<TD WIDTH=10> </TD>%n");
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%s</TD>%n",
				info.state);
			out.printf ("</TR>%n");
			out.printf ("<TR>%n");
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">Started:</TD>%n");
			out.printf ("<TD WIDTH=10> </TD>%n");
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%s</TD>%n",
				new Date (info.startTime));
			out.printf ("</TR>%n");
			if (info.state == LauncherState.FAILED)
				{
				out.printf ("<TR>%n");
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">Failed:</TD>%n");
				out.printf ("<TD WIDTH=10> </TD>%n");
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%s</TD>%n",
					new Date (info.failedTime));
				out.printf ("</TR>%n");
				out.printf ("</TABLE>%n");
				}
			else
				{
				out.printf ("<TR>%n");
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">CPU cores:</TD>%n");
				out.printf ("<TD WIDTH=10> </TD>%n");
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%d</TD>%n",
					info.coresTotal);
				out.printf ("</TR>%n");
				out.printf ("<TR>%n");
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">GPU accelerators:</TD>%n");
				out.printf ("<TD WIDTH=10> </TD>%n");
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%s</TD>%n",
					numberOrNone (info.gpusTotal));
				out.printf ("</TR>%n");
				out.printf ("<TR>%n");
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">Cores in use:</TD>%n");
				out.printf ("<TD WIDTH=10> </TD>%n");
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><IMG SRC=\"%s\"> %d of %d</TD>%n",
					inUseBarUrl (info.coresTotal, info.coresAvailable),
					info.coresTotal - info.coresAvailable,
					info.coresTotal);
				out.printf ("</TR>%n");
				out.printf ("<TR>%n");
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">GPUs in use:</TD>%n");
				out.printf ("<TD WIDTH=10> </TD>%n");
				if (info.gpusTotal == 0)
					out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">n/a</TD>%n");
				else
					out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><IMG SRC=\"%s\"> %d of %d</TD>%n",
						inUseBarUrl (info.gpusTotal, info.gpusAvailable),
						info.gpusTotal - info.gpusAvailable,
						info.gpusTotal);
				out.printf ("</TR>%n");
				out.printf ("<TR>%n");
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">Running tasks:</TD>%n");
				out.printf ("<TD WIDTH=10> </TD>%n");
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%s</TD>%n",
					numberOrNone (count[0]));
				out.printf ("</TR>%n");
				out.printf ("<TR>%n");
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">Pending tasks:</TD>%n");
				out.printf ("<TD WIDTH=10> </TD>%n");
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%s</TD>%n",
					numberOrNone (count[1]));
				out.printf ("</TR>%n");
				out.printf ("</TABLE>%n");
				out.printf ("<H3>Tasks</H3>");
				displayTaskList (out, now, false, true,
					new FindNode (info.name),
					runningTaskList, pendingTaskList);
				}
			}

		return nodeDetailsUrl (name);
		}

	/**
	 * Display the job details web page.
	 *
	 * @param  out   Print writer for HTML body.
	 * @param  now   Current time and date.
	 * @param  id    Job ID string.
	 *
	 * @return  Web page URL.
	 */
	private static String displayJobDetails
		(PrintWriter out,
		 long now,
		 String id)
		{
		out.printf ("<H2>Job %s Details</H2>%n", id);
		out.printf ("<P>%n");
		long jobID = 0;
		try { jobID = Long.parseLong (id); }
			catch (NumberFormatException exc) {}
		JobInfo info = jobMap.get (jobID);
		if (info == null)
			out.printf ("Job %s nonexistent%n", id);
		else
			{
			out.printf ("<TABLE BORDER=0 CELLPADDING=0 CELLSPACING=0>%n");
			out.printf ("<TR>%n");
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">Job:</TD>%n");
			out.printf ("<TD WIDTH=10> </TD>%n");
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%d</TD>%n",
				info.jobID);
			out.printf ("</TR>%n");
			out.printf ("<TR>%n");
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">User:</TD>%n");
			out.printf ("<TD WIDTH=10> </TD>%n");
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%s</TD>%n",
				info.user);
			out.printf ("</TR>%n");
			out.printf ("<TR>%n");
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">Started:</TD>%n");
			out.printf ("<TD WIDTH=10> </TD>%n");
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%s</TD>%n",
				new Date (info.startTime));
			out.printf ("</TR>%n");
			out.printf ("<TR>%n");
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">Running time:</TD>%n");
			out.printf ("<TD WIDTH=10> </TD>%n");
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%d sec</TD>%n",
				msecToSec (now - info.startTime));
			out.printf ("</TR>%n");
			out.printf ("</TABLE>%n");
			out.printf ("<H3>Tasks</H3>");
			displayTaskList (out, now, true, false,
				new FindJob (info.jobID),
				runningTaskList, pendingTaskList, finishedTaskList);
			}

		return jobDetailsUrl (id);
		}

	/**
	 * Display the task list web page.
	 *
	 * @param  out   Print writer for HTML body.
	 * @param  now   Current time and date.
	 *
	 * @return  Web page URL.
	 */
	private static String displayTaskList
		(PrintWriter out,
		 long now)
		{
		out.printf ("<H2>Task List</H2>%n");
		displayTaskList (out, now, true, true, new AllTasks(),
			runningTaskList, pendingTaskList, finishedTaskList);
		return taskListUrl();
		}

	/**
	 * Display a task list.
	 *
	 * @param  out        Print writer for HTML body.
	 * @param  now        Current time and date.
	 * @param  nodeLinks  True to include links to node details.
	 * @param  jobLinks   True to include links to job details.
	 * @param  predicate  Predicate indicating which tasks to display.
	 * @param  taskLists  Task list(s) containing tasks to display.
	 */
	private static void displayTaskList
		(PrintWriter out,
		 long now,
		 boolean nodeLinks,
		 boolean jobLinks,
		 Predicate<JobTaskInfo> predicate,
		 DList<JobTaskInfo>... taskLists)
		{
		int row = 0;
		out.printf ("<P>%n");
		out.printf ("<TABLE BORDER=1 CELLPADDING=2 CELLSPACING=0>%n");
		out.printf ("<TR>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">%n");
		out.printf ("<TABLE BORDER=0 CELLPADDING=2 CELLSPACING=0>%n");
		out.printf ("<TR BGCOLOR=\"%s\">%n", stripeColor (row ++));
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;Job&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;Task&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;User&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;Status&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;Node&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;Cores&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;GPUs&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;History&nbsp;</I></TD>%n");
		out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;&nbsp;</I></TD>%n");
		if (jobLinks)
			out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\"><I>&nbsp;Job details&nbsp;</I></TD>%n");
		out.printf ("</TR>%n");
		for (DList<JobTaskInfo> taskList : taskLists)
			{
			DListEntry<JobTaskInfo> entry = taskList.first();
			while (entry != null)
				{
				JobTaskInfo info = entry.item();
				entry = entry.succ();
				if (! predicate.test (info)) continue;
				out.printf ("<TR BGCOLOR=\"%s\">%n", stripeColor (row ++));
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;%d&nbsp;</TD>%n",
					info.jobID);
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;%d&nbsp;</TD>%n",
					info.taskID);
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;%s&nbsp;</TD>%n",
					info.user);
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;%s&nbsp;</TD>%n",
					info.state);
				if (info.node.nodeName().equals (NodeProperties.ANY_NODE_NAME))
					out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;(Any)&nbsp;</TD>%n");
				else if (nodeLinks)
					out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;<A HREF=\"%s\">%s</A>&nbsp;</TD>%n",
						nodeDetailsUrl (info.node.nodeName()), info.node.nodeName());
				else
					out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;%s&nbsp;</TD>%n",
						info.node.nodeName());
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;%s&nbsp;</TD>%n",
					info.node.cores() == NodeProperties.ALL_CORES ?
						"(All)" : "" + info.node.cores());
				out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;%s&nbsp;</TD>%n",
					info.node.gpus() == NodeProperties.ALL_GPUS ?
						"(All)" : "" + numberOrNone (info.node.gpus()));
				switch (info.state)
					{
					case Pending:
						out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;Launched %s&nbsp;</TD>%n",
							new Date (info.launchTime));
						out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;Waiting %d sec&nbsp;</TD>%n",
							msecToSec (now - info.launchTime));
						break;
					case Running:
						out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;Launched %s&nbsp;<BR>&nbsp;Started %s&nbsp;</TD>%n",
							new Date (info.launchTime),
							new Date (info.startTime));
						out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;Waited %d sec&nbsp;<BR>&nbsp;Running %d sec&nbsp;</TD>%n",
							msecToSec (info.startTime - info.launchTime),
							msecToSec (now - info.startTime));
						break;
					case Finished:
						out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;Launched %s&nbsp;<BR>&nbsp;Started %s&nbsp;<BR>&nbsp;Finished %s&nbsp;</TD>%n",
							new Date (info.launchTime),
							new Date (info.startTime),
							new Date (info.finishTime));
						out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;Waited %d sec&nbsp;<BR>&nbsp;Ran %d sec&nbsp;</TD>%n",
							msecToSec (info.startTime - info.launchTime),
							msecToSec (info.finishTime - info.startTime));
						break;
					case Failed:
						if (info.startTime == 0L)
							{
							out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;Launched %s&nbsp;<BR>&nbsp;Failed %s&nbsp;</TD>%n",
								new Date (info.launchTime),
								new Date (info.finishTime));
							out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;Waited %d sec&nbsp;</TD>%n",
								msecToSec (info.finishTime - info.launchTime));
							}
						else
							{
							out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;Launched %s&nbsp;<BR>&nbsp;Started %s&nbsp;<BR>&nbsp;Failed %s&nbsp;</TD>%n",
								new Date (info.launchTime),
								new Date (info.startTime),
								new Date (info.finishTime));
							out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;Waited %d sec&nbsp;<BR>&nbsp;Ran %d sec&nbsp;</TD>%n",
								msecToSec (info.startTime - info.launchTime),
								msecToSec (info.finishTime - info.startTime));
							}
						break;
					}
				if (jobLinks)
					out.printf ("<TD ALIGN=\"left\" VALIGN=\"top\">&nbsp;<A HREF=\"%s\">Details</A>&nbsp;</TD>%n",
						jobDetailsUrl (info.jobID));
				out.printf ("</TR>%n");
				}
			}
		out.printf ("</TABLE>%n");
		out.printf ("</TD>%n");
		out.printf ("</TR>%n");
		out.printf ("</TABLE>%n");
		}

	/**
	 * Returns the summary web page URL.
	 *
	 * @return  URL.
	 */
	private static String summaryUrl()
		{
		return url ("/summary");
		}

	/**
	 * Returns the node details web page URL for the given launcher ID.
	 *
	 * @param  name  Node name.
	 *
	 * @return  URL.
	 */
	private static String nodeDetailsUrl
		(String name)
		{
		return url (String.format ("/node=%s", name));
		}

	/**
	 * Returns the job details web page URL for the given job ID.
	 *
	 * @param  jobID  Job ID.
	 *
	 * @return  URL.
	 */
	private static String jobDetailsUrl
		(long jobID)
		{
		return url (String.format ("/job=%d", jobID));
		}

	/**
	 * Returns the job details web page URL for the given job ID string.
	 *
	 * @param  id  Job ID string.
	 *
	 * @return  URL.
	 */
	private static String jobDetailsUrl
		(String id)
		{
		return url (String.format ("/job=%s", id));
		}

	/**
	 * Returns the task list web page URL.
	 *
	 * @return  URL.
	 */
	private static String taskListUrl()
		{
		return url ("/tasks");
		}

	/**
	 * Returns the in-use bar URL for the given total and available counts.
	 *
	 * @param  total  Total count.
	 * @param  available  Available count.
	 */
	private static String inUseBarUrl
		(int total,
		 int available)
		{
		int inUse = total - available;
		int percent = (int)(Math.ceil(20.0*inUse/total))*5;
		return url (String.format ("/bar%02d.png", percent));
		}

	/**
	 * Returns the URL for the given web page URI.
	 *
	 * @param  uri  URI.
	 *
	 * @return  URL.
	 */
	private static String url
		(String uri)
		{
		return String.format ("http://%s:%d%s", webHost, webPort, uri);
		}

	/**
	 * Returns the stripe color for the given row. Even rows are gray, odd rows
	 * are white.
	 *
	 * @param  row  Row.
	 *
	 * @return  Color string.
	 */
	private static String stripeColor
		(int row)
		{
		return row % 2 == 0 ? "#E2E2E2" : "#FFFFFF";
		}

	/**
	 * Returns the numbers of running and pending tasks for the given node.
	 *
	 * @param  name  Node name.
	 *
	 * @return  Two-element array; element 0 = number of running tasks; element
	 *          1 = number of pending tasks.
	 */
	private static int[] countTasksForNode
		(final String name)
		{
		final int[] count = new int [2];
		runningTaskList.forEachItemDo (new Action<JobTaskInfo>()
			{
			public void run (JobTaskInfo info)
				{
				if (info.node.nodeName().equals (name))
					++ count[0];
				}
			});
		pendingTaskList.forEachItemDo (new Action<JobTaskInfo>()
			{
			public void run (JobTaskInfo info)
				{
				if (info.node.nodeName().equals (name))
					++ count[1];
				}
			});
		return count;
		}

	/**
	 * Returns the numbers of running, pending, and finished tasks for the given
	 * job ID.
	 *
	 * @param  jobID  Job ID.
	 *
	 * @return  Three-element array; element 0 = number of running tasks;
	 *          element 1 = number of pending tasks; element 2 = number of
	 *          finished tasks.
	 */
	private static int[] countTasksForJob
		(final long jobID)
		{
		final int[] count = new int [3];
		runningTaskList.forEachItemDo (new Action<JobTaskInfo>()
			{
			public void run (JobTaskInfo info)
				{
				if (info.jobID == jobID)
					++ count[0];
				}
			});
		pendingTaskList.forEachItemDo (new Action<JobTaskInfo>()
			{
			public void run (JobTaskInfo info)
				{
				if (info.jobID == jobID)
					++ count[1];
				}
			});
		finishedTaskList.forEachItemDo (new Action<JobTaskInfo>()
			{
			public void run (JobTaskInfo info)
				{
				if (info.jobID == jobID)
					++ count[2];
				}
			});
		return count;
		}

	/**
	 * Returns the given number, or <TT>"None"</TT> if the number is 0.
	 *
	 * @param  n  Number.
	 *
	 * @return  String.
	 */
	private static String numberOrNone
		(int n)
		{
		return n == 0 ? "None" : "" + n;
		}

	/**
	 * Converts the given milliseconds to seconds.
	 *
	 * @param  msec  Milliseconds.
	 *
	 * @return  Seconds.
	 */
	private static long msecToSec
		(long msec)
		{
		return (msec + 500L)/1000L;
		}

	}
