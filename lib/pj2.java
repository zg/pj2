//******************************************************************************
//
// File:    pj2.java
// Package: ---
// Unit:    Class pj2
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

import edu.rit.pj2.Debug;
import edu.rit.pj2.JarClassLoader;
import edu.rit.pj2.Job;
import edu.rit.pj2.Rule;
import edu.rit.pj2.Schedule;
import edu.rit.pj2.Task;
import edu.rit.pj2.TaskSpec;
import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.util.EnumSet;

/**
 * Class pj2 is the PJ2 job and task launcher main program.
 * <P>
 * Usage: <TT>java pj2 [threads=<I>threads</I>] [schedule=<I>schedule</I>]
 * [chunk=<I>chunk</I>] [nodeName=<I>nodeName</I>] [cores=<I>cores</I>]
 * [gpus=<I>gpus</I>] [workers=<I>workers</I>]
 * [masterSchedule=<I>masterSchedule</I>] [masterChunk=<I>masterChunk</I>]
 * [tracker=<I>host</I>[:<I>port</I>]] [listen=<I>host</I>[:<I>port</I>]]
 * [debug=<I>message</I>[,<I>message</I>...]]
 * [jvmFlags=<I>flag</I>[,<I>flag</I>]] [jar=<I>file</I>]
 * [timelimit=<I>timelimit</I>] <I>jobOrTaskClass</I> [<I>arg</I> ...]</TT>
 * <P>
 * PJ2 is designed for writing two kinds of parallel programs: tightly coupled
 * single-node and loosely coupled multi-node.
 * <P>
 * <B>Tightly coupled single-node parallel programs.</B>
 * A tightly coupled single-node parallel program is intended to run on one
 * multicore node. The program consists of multiple threads that run in parallel
 * on separate cores. The threads communicate and coordinate by reading and
 * writing variables in shared memory.
 * <P>
 * This kind of PJ2 program is expressed as a subclass of class {@linkplain
 * edu.rit.pj2.Task Task}. The program is executed by running the <TT>pj2</TT>
 * program as follows:
 * <P>
 * <TT>$ java pj2 MainTask <I>arg</I> ...</TT>
 * <P>
 * The above command runs class MainTask, a subclass of class {@linkplain
 * edu.rit.pj2.Task Task}. Replace <TT>MainTask</TT> with the fully-qualified
 * class name of the desired task subclass. The <TT>pj2</TT> program creates an
 * instance of the task subclass, then calls that task's {@link
 * edu.rit.pj2.Task#main(String[]) main()} method, passing in an array of the
 * command line argument strings (zero or more). The {@link
 * edu.rit.pj2.Task#main(String[]) main()} method carries out the computation.
 * When the {@link edu.rit.pj2.Task#main(String[]) main()} method returns, the
 * <TT>pj2</TT> program terminates.
 * <P>
 * The computation can contain sequential statements executed by a single
 * thread, interspersed with parallel statements executed by multiple threads.
 * Running on a multicore computer, the Java Virtual Machine (JVM) schedules
 * each thread on a separate core, thus executing the program in parallel.
 * <P>
 * <B>Loosely coupled multi-node parallel programs.</B>
 * A loosely coupled multi-node parallel program is intended to run on multiple
 * nodes. The nodes may be located in a cluster parallel computer, in a grid, or
 * in a cloud. The program consists of multiple {@linkplain edu.rit.pj2.Task
 * Task}s that run in parallel on separate nodes. The tasks coordinate and
 * communicate with each other by reading and writing {@linkplain
 * edu.rit.pj2.Tuple Tuple}s in <I>tuple space.</I> The tasks are intended to be
 * <I>coarse-grained,</I> meaning each task runs for minutes or hours, not
 * milliseconds. The tasks are intended to communicate <I>small to medium
 * amounts of data</I> via tuple space, not enormous quantities of data.
 * <P>
 * This kind of PJ2 program is expressed as a subclass of class {@linkplain
 * edu.rit.pj2.Job Job}. The program is executed by running the <TT>pj2</TT>
 * program as follows:
 * <P>
 * <TT>$ java pj2 jar=<I>file</I> MainJob <I>arg</I> ...</TT>
 * <P>
 * The above command runs class MainJob, a subclass of class {@linkplain
 * edu.rit.pj2.Job Job}. Replace <TT>MainJob</TT> with the fully-qualified class
 * name of the desired job subclass. The <TT>pj2</TT> program creates an
 * instance of the job subclass, then calls that job's {@link
 * edu.rit.pj2.Job#main(String[]) main()} method, passing in an array of the
 * command line argument strings (zero or more). The {@link
 * edu.rit.pj2.Job#main(String[]) main()} method defines {@linkplain
 * edu.rit.pj2.Rule Rule}s that specify which {@linkplain edu.rit.pj2.Task
 * Task}s are to be executed as part of the job. The <TT>pj2</TT> program
 * creates and executes those tasks, which carry out the job's computation. The
 * tasks may execute in the <TT>pj2</TT> program's process or on computational
 * nodes that are part of a cluster, grid, or cloud. When all the tasks have
 * finished executing, the <TT>pj2</TT> program terminates.
 * <P>
 * Running on a cluster, grid, or cloud, the program executes each task on a
 * separate node, thus executing the program in parallel. Each task receives
 * input tuples that are taken out of tuple space, performs computations, and
 * generates output tuples that are written into tuple space. Each task executes
 * independently of other tasks. All inter-task coordination and communication
 * goes through tuple space.
 * <P>
 * <B>Properties.</B>
 * Certain properties of the job or task can be specified on the <TT>pj2</TT>
 * command line by including the appropriate option before the job or task class
 * name. These override the default property values. The properties are:
 * <UL>
 * <P><LI>
 * <TT>threads</TT> specifies the number of threads that will execute a parallel
 * for loop. Specify <TT>threads=all</TT> to indicate that a parallel for loop
 * will be executed by as many threads as there are cores on the machine on
 * which the task is running. The default is <TT>threads=all</TT>. See the
 * {@link edu.rit.pj2.Job#threads(int) Job.threads()} or {@link
 * edu.rit.pj2.Task#threads(int) Task.threads()} method for further information.
 * <P><LI>
 * <TT>schedule</TT> specifies how the iterations of a parallel for loop are
 * partitioned among the threads executing the parallel for loop. The default is
 * a fixed schedule. See the {@link
 * edu.rit.pj2.Job#schedule(edu.rit.pj2.Schedule) Job.schedule()} or {@link
 * edu.rit.pj2.Task#schedule(edu.rit.pj2.Schedule) Task.schedule()} method for
 * further information.
 * <P><LI>
 * <TT>chunk</TT> specifies how the iterations of a parallel for loop are
 * partitioned among the threads executing the parallel for loop. The
 * interpretation of the default <TT>chunk</TT> property depends on the
 * <TT>schedule</TT> property. See the {@link edu.rit.pj2.Job#chunk(int)
 * Job.chunk()} or {@link edu.rit.pj2.Task#chunk(int) Task.chunk()} method for
 * further information.
 * <P><LI>
 * <TT>nodeName</TT> specifies the name of the node on which a task must run.
 * Specify <TT>nodeName=any</TT> to indicate that a task can run on any node.
 * The default is <TT>nodeName=any</TT>. See the {@link
 * edu.rit.pj2.Job#nodeName(String) Job.nodeName()} or {@link
 * edu.rit.pj2.Task#nodeName(String) Task.nodeName()} method for further
 * information.
 * <P><LI>
 * <TT>cores</TT> specifies the number of CPU cores a task requires. Specify
 * <TT>cores=all</TT> to indicate that a task requires all the CPU cores on the
 * node. The default is <TT>cores=all</TT>. See the {@link
 * edu.rit.pj2.Job#cores(int) Job.cores()} or {@link edu.rit.pj2.Task#cores(int)
 * Task.cores()} method for further information.
 * <P><LI>
 * <TT>gpus</TT> specifies the number of GPU accelerators a task requires. The
 * default is none (0). See the {@link edu.rit.pj2.Job#gpus(int) Job.gpus()} or
 * {@link edu.rit.pj2.Task#gpus(int) Task.gpus()} method for further
 * information.
 * <P><LI>
 * <TT>workers</TT> specifies the number of worker tasks that will execute a
 * master-worker cluster parallel for loop. The default is one worker task. See
 * the {@link edu.rit.pj2.Job#workers(int) Job.workers()} method for further
 * information.
 * <P><LI>
 * <TT>masterSchedule</TT> specifies how the iterations of a master-worker
 * cluster parallel for loop are partitioned among the worker tasks executing
 * the parallel for loop. The default is a fixed schedule. See the {@link
 * edu.rit.pj2.Job#masterSchedule(edu.rit.pj2.Schedule) Job.masterSchedule()}
 * method for further information.
 * <P><LI>
 * <TT>masterChunk</TT> specifies how the iterations of a master-worker cluster
 * parallel for loop are partitioned among the worker tasks executing the
 * parallel for loop. The interpretation of the default <TT>masterChunk</TT>
 * property depends on the <TT>masterSchedule</TT> property. See the {@link
 * edu.rit.pj2.Job#masterChunk(int) Job.masterChunk()} method for further
 * information.
 * <P><LI>
 * <TT>tracker</TT> specifies the host and port at which the Tracker is
 * listening for connections from jobs. If <TT>tracker=none</TT> is specified,
 * the job will not connect to the Tracker. See the {@link
 * edu.rit.pj2.Job#trackerHost(String) Job.trackerHost()} and {@link
 * edu.rit.pj2.Job#trackerPort(int) Job.trackerPort()} methods for further
 * information.
 * <P><LI>
 * <TT>listen</TT> specifies the host and port at which the job itself will
 * listen for connections from the nodes that will run the job's tasks. See the
 * {@link edu.rit.pj2.Job#listenHost(String) Job.listenHost()} and {@link
 * edu.rit.pj2.Job#listenPort(int) Job.listenPort()} methods for further
 * information.
 * <P><LI>
 * <TT>debug</TT> tells a job to print a certain debugging message or messages.
 * Multiple messages may be specified, separated by commas (no spaces). Specify
 * <TT>debug=all</TT> to turn on all debugging messages. Specify
 * <TT>debug=none</TT> to turn off all debugging messages. See the {@link
 * edu.rit.pj2.Job#debug(edu.rit.pj2.Debug[]) Job.debug()} method for further
 * information.
 * <P><LI>
 * <TT>jvmFlags</TT> specifies one or more flags that are included on the
 * command lines of the JVMs that run each of the tasks in the job. Multiple
 * flags may be specified, separated by commas (no spaces). See the {@link
 * edu.rit.pj2.Job#jvmFlags(String[]) Job.jvmFlags()} method for further
 * information.
 * <P><LI>
 * <TT>jar</TT> specifies the Java archive (JAR) file that contains the class
 * files needed by a job and its tasks. For further information, see class
 * {@linkplain edu.rit.pj2.Job Job}.
 * <P><LI>
 * <TT>timelimit</TT> specifies a time limit in seconds for the job. The default
 * is no time limit. See the {@link edu.rit.pj2.Job#timeLimit(int)
 * Job.timeLimit()} method for further information.
 * </UL>
 * <P>
 * <B>Abnormal termination.</B>
 * If code in a job or task throws an exception that propagates out of the
 * <TT>main()</TT> method, the <TT>pj2</TT> program prints an exception stack
 * trace and terminates.
 * <P>
 * Code in a job or task must <I>not</I> call the <TT>System.exit()</TT> method.
 * If a job or task needs to terminate itself, the job or task should throw an
 * exception that propagates out of every method without being caught, such as a
 * {@linkplain edu.rit.pj2.TerminateException TerminateException} or any other
 * runtime exception. The exception's detail message can explain why the job or
 * task terminated. This exception propagates out of the <TT>main()</TT> method
 * and causes the <TT>pj2</TT> program to terminate.
 *
 * @see  edu.rit.pj2
 * @see  edu.rit.pj2.Job
 * @see  edu.rit.pj2.Task
 *
 * @author  Alan Kaminsky
 * @version 01-Oct-2015
 */
public class pj2
	{

// Prevent construction.

	private pj2()
		{
		}

// Command line arguments.

	private static Integer threads = null;
	private static Schedule schedule = null;
	private static Integer chunk = null;
	private static String nodeName = null;
	private static Integer cores = null;
	private static Integer gpus = null;
	private static Integer workers = null;
	private static Schedule masterSchedule = null;
	private static Integer masterChunk = null;
	private static String tracker = null;
	private static String listen = null;
	private static EnumSet<Debug> debug = null;
	private static String[] jvmFlags = null;
	private static String jar = null;
	private static Integer timeLimit = null;
	private static String jobOrTaskClass = null;
	private static Class<?> jtclass = null;
	private static String[] taskArgs = null;

// Exported operations.

	/**
	 * PJ2 launcher main program.
	 *
	 * @param  args  Property settings (optional), job or task class name, job's
	 *               or task's command line arguments (zero or more).
	 */
	public static void main
		(String[] args)
		{
		try
			{
			// Parse command line arguments.
			int argi = 0;
			while (argi < args.length && jobOrTaskClass == null)
				{
				if (args[argi].startsWith ("threads="))
					{
					String t = args[argi].substring (8);
					if (t.equals ("all"))
						threads = new Integer (Job.THREADS_EQUALS_CORES);
					else
						{
						try
							{
							threads = new Integer (t);
							if (threads < 1)
								usageIllegal (args[argi]);
							}
						catch (NumberFormatException exc)
							{
							usageIllegal (args[argi]);
							}
						}
					}
				else if (args[argi].startsWith ("schedule="))
					{
					try
						{
						schedule = Schedule.valueOf (args[argi].substring (9));
						}
					catch (IllegalArgumentException exc)
						{
						usageIllegal (args[argi]);
						}
					}
				else if (args[argi].startsWith ("chunk="))
					{
					try
						{
						chunk = new Integer (args[argi].substring (6));
						if (chunk < 1)
							usageIllegal (args[argi]);
						}
					catch (NumberFormatException exc)
						{
						usageIllegal (args[argi]);
						}
					}
				else if (args[argi].startsWith ("nodeName="))
					{
					nodeName = args[argi].substring (9);
					if (nodeName.equals ("any"))
						nodeName = Job.ANY_NODE_NAME;
					}
				else if (args[argi].startsWith ("cores="))
					{
					String c = args[argi].substring (6);
					if (c.equals ("all"))
						cores = new Integer (Job.ALL_CORES);
					else
						{
						try
							{
							cores = new Integer (c);
							if (cores < 1)
								usageIllegal (args[argi]);
							}
						catch (NumberFormatException exc)
							{
							usageIllegal (args[argi]);
							}
						}
					}
				else if (args[argi].startsWith ("gpus="))
					{
					String g = args[argi].substring (5);
					try
						{
						gpus = new Integer (g);
						if (gpus < 0)
							usageIllegal (args[argi]);
						}
					catch (NumberFormatException exc)
						{
						usageIllegal (args[argi]);
						}
					}
				else if (args[argi].startsWith ("workers="))
					{
					try
						{
						workers = new Integer (args[argi].substring (8));
						if (workers < 1)
							usageIllegal (args[argi]);
						}
					catch (NumberFormatException exc)
						{
						usageIllegal (args[argi]);
						}
					}
				else if (args[argi].startsWith ("masterSchedule="))
					{
					try
						{
						masterSchedule =
							Schedule.valueOf (args[argi].substring (15));
						}
					catch (IllegalArgumentException exc)
						{
						usageIllegal (args[argi]);
						}
					}
				else if (args[argi].startsWith ("masterChunk="))
					{
					try
						{
						masterChunk = new Integer (args[argi].substring (12));
						if (masterChunk < 1)
							usageIllegal (args[argi]);
						}
					catch (NumberFormatException exc)
						{
						usageIllegal (args[argi]);
						}
					}
				else if (args[argi].startsWith ("tracker="))
					{
					tracker = args[argi].substring (8);
					}
				else if (args[argi].startsWith ("listen="))
					{
					listen = args[argi].substring (7);
					}
				else if (args[argi].startsWith ("debug="))
					{
					String d = args[argi].substring (6);
					if (d.equals ("all"))
						debug = EnumSet.allOf (Debug.class);
					else if (d.equals ("none"))
						debug = EnumSet.noneOf (Debug.class);
					else
						{
						try
							{
							debug = EnumSet.noneOf (Debug.class);
							for (String s : d.split (","))
								debug.add (Debug.valueOf (s));
							}
						catch (IllegalArgumentException exc)
							{
							usageIllegal (args[argi]);
							}
						}
					}
				else if (args[argi].startsWith ("jvmFlags="))
					{
					jvmFlags = args[argi] .substring (9) .split (",");
					}
				else if (args[argi].startsWith ("jar="))
					{
					jar = args[argi].substring (4);
					}
				else if (args[argi].startsWith ("timelimit="))
					{
					try
						{
						timeLimit = new Integer (args[argi].substring (10));
						if (timeLimit < 1)
							usageIllegal (args[argi]);
						}
					catch (NumberFormatException exc)
						{
						usageIllegal (args[argi]);
						}
					}
				else
					{
					jobOrTaskClass = args[argi];
					}
				++ argi;
				}
			int numTaskArgs = args.length - argi;
			taskArgs = new String [numTaskArgs];
			System.arraycopy (args, argi, taskArgs, 0, numTaskArgs);

			// Install JAR class loader if specified.
			if (jar != null)
				{
				BufferedInputStream in = new BufferedInputStream
					(new FileInputStream (jar));
				ByteArrayOutputStream out = new ByteArrayOutputStream();
				int b;
				while ((b = in.read()) != -1)
					out.write (b);
				in.close();
				Thread.currentThread().setContextClassLoader
					(new JarClassLoader
						(Thread.currentThread().getContextClassLoader(),
						 out.toByteArray()));
				}

			// Get job or task class.
			if (jobOrTaskClass == null)
				usage ("Missing jobOrTaskClass");
			try
				{
				jtclass = Class.forName (jobOrTaskClass, true,
					Thread.currentThread().getContextClassLoader());
				}
			catch (ClassNotFoundException exc)
				{
				usage (String.format
					("Job or task class %s not found", jobOrTaskClass));
				}

			// Job to be executed.
			Job job = null;

			// Execute a parallel program consisting of a task.
			if (Task.class.isAssignableFrom (jtclass))
				{
				// Create instance of task class. This is just to make sure the
				// task class can be instantiated.
				Task task = null;
				try
					{
					task = (Task) jtclass.newInstance();
					}
				catch (IllegalAccessException exc)
					{
					usage (String.format
						("Task class %s or its nullary constructor cannot be accessed",
						 jobOrTaskClass));
					}
				catch (InstantiationException exc)
					{
					usage (String.format
						("Task class %s cannot be instantiated",
						 jobOrTaskClass));
					}
				task = null;

				// Set up a single-task Job.
				job = new Job()
					{
					public void main (String[] args)
						{
						rule() .task ((Class<Task>) jtclass) .args (taskArgs);
						}
					};
				job.debug (EnumSet.noneOf (Debug.class));
				}

			// Execute a parallel program consisting of a job.
			else if (Job.class.isAssignableFrom (jtclass))
				{
				// Create instance of job class.
				try
					{
					job = (Job) jtclass.newInstance();
					}
				catch (IllegalAccessException exc)
					{
					usage (String.format
						("Job class %s or its nullary constructor cannot be accessed",
						 jobOrTaskClass));
					}
				catch (InstantiationException exc)
					{
					usage (String.format
						("Job class %s cannot be instantiated",
						 jobOrTaskClass));
					}
				}

			// Class is not a task class or a job class.
			else
				usage (String.format
					("Class %s is not a Job or a Task", jobOrTaskClass));

			// Override job's property settings if any.
			if (threads != null)
				job.threads (threads);
			if (schedule != null)
				job.schedule (schedule);
			if (chunk != null)
				job.chunk (chunk);
			if (nodeName != null)
				job.nodeName (nodeName);
			if (cores != null)
				job.cores (cores);
			if (gpus != null)
				job.gpus (gpus);
			if (workers != null)
				job.workers (workers);
			if (masterSchedule != null)
				job.masterSchedule (masterSchedule);
			if (masterChunk != null)
				job.masterChunk (masterChunk);
			if (tracker != null)
				try
					{
					int i = tracker.indexOf (':');
					if (i == -1)
						job.trackerHost (tracker);
					else
						{
						job.trackerHost (tracker.substring (0, i));
						job.trackerPort (Integer.parseInt
							(tracker.substring (i + 1)));
						}
					}
				catch (Exception exc)
					{
					usage (String.format ("tracker=%s illegal", tracker));
					}
			if (listen != null)
				try
					{
					int i = listen.indexOf (':');
					if (i == -1)
						job.listenHost (listen);
					else
						{
						job.listenHost (listen.substring (0, i));
						job.listenPort (Integer.parseInt
							(listen.substring (i + 1)));
						}
					}
				catch (Exception exc)
					{
					usage (String.format ("listen=%s illegal", listen));
					}
			if (debug != null)
				job.debug (debug);
			if (jvmFlags != null)
				job.jvmFlags (jvmFlags);
			if (jar != null)
				job.jar (new File (jar));
			if (timeLimit != null)
				job.timeLimit (timeLimit);

			// Execute job.
			job.main (taskArgs);
			job.execute();

			// Exit program.
			System.exit (0);
			}

		// On any exception, print stack trace and terminate.
		catch (Throwable exc)
			{
			exc.printStackTrace (System.err);
			System.exit (1);
			}
		}

// Hidden operations.

	/**
	 * Print an illegal argument usage message and exit.
	 *
	 * @param  arg  Command line argument.
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
		System.err.printf ("pj2: %s%n", msg);
		System.err.println ("Usage: java pj2 [threads=<threads>] [schedule=<schedule>] [chunk=<chunk>] [nodeName=<nodeName>] [cores=<cores>] [gpus=<gpus>] [workers=<workers>] [masterSchedule=<masterSchedule>] [masterChunk=<masterChunk>] [tracker=<host>[:<port>]] [listen=<host>[:<port>]] [debug=<message>[,<message>...]] [jar=<file>] [timelimit=<timelimit>] <jobOrTaskClass> [<arg> ...]");
		System.exit (1);
		}

	}
