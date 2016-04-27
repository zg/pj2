//******************************************************************************
//
// File:    Job.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.Job
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

package edu.rit.pj2;

import edu.rit.gpu.Gpu;
import edu.rit.io.ThreadedOutputStream;
import edu.rit.numeric.Int96;
import edu.rit.pj2.tracker.BackendRef;
import edu.rit.pj2.tracker.HeartbeatFailedException;
import edu.rit.pj2.tracker.JobProperties;
import edu.rit.pj2.tracker.JobReceiver;
import edu.rit.pj2.tracker.JobRef;
import edu.rit.pj2.tracker.LaunchException;
import edu.rit.pj2.tracker.LauncherRef;
import edu.rit.pj2.tracker.Proxy;
import edu.rit.pj2.tracker.Receiver;
import edu.rit.pj2.tracker.ReceiverListener;
import edu.rit.pj2.tracker.TaskInfo;
import edu.rit.pj2.tracker.TaskMap;
import edu.rit.pj2.tracker.TaskProperties;
import edu.rit.pj2.tracker.TaskSpecInfo;
import edu.rit.pj2.tracker.TrackerRef;
import edu.rit.pj2.tracker.TrackerSender;
import edu.rit.util.AList;
import edu.rit.util.Action;
import edu.rit.util.DList;
import edu.rit.util.DListEntry;
import edu.rit.util.ElapsedTime;
import edu.rit.util.Heartbeat;
import edu.rit.util.Instance;
import edu.rit.util.Predicate;
import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.InterfaceAddress;
import java.net.NetworkInterface;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.util.EnumSet;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Semaphore;

/**
 * Class Job is the abstract base class for a PJ2 job. A PJ2 parallel program is
 * either a single task (a subclass of class {@linkplain Task Task}) or is a
 * number of tasks grouped together into a job (a subclass of class Job). A PJ2
 * program is executed by running the PJ2 launcher program, {@link pj2 pj2}.
 * <P>
 * <B>Programming a job.</B>
 * To set up a job:
 * <UL>
 * <P><LI>
 * Write a subclass of class Job.
 * <P><LI>
 * In the Job subclass's {@link #main(String[]) main()} method, call the {@link
 * #rule() rule()} method one or more times to define the job's rules. See class
 * {@linkplain Rule Rule} for further information about defining rules.
 * <P><LI>
 * Optionally, call the {@link #putTuple(Tuple) putTuple()} method to put
 * initial tuples into tuple space.
 * <P><LI>
 * Optionally, specify the job's properties by calling the {@link #threads(int)
 * threads()}, {@link #schedule(Schedule) schedule()}, {@link #chunk(int)
 * chunk()}, {@link #nodeName(String) nodeName()}, {@link #cores(int) cores()},
 * {@link #gpus(int) gpus()}, {@link #workers(int) workers()}, {@link
 * #masterSchedule(Schedule) masterSchedule()}, {@link #masterChunk(int)
 * masterChunk()}, {@link #trackerHost(String) trackerHost()}, {@link
 * #trackerPort(int) trackerPort()}, {@link #listenHost(String) listenHost()},
 * {@link #listenPort(int) listenPort()}, {@link #debug(Debug[]) debug()},
 * {@link #jvmFlags(String[]) jvmFlags()}, and {@link #timeLimit(int)
 * timeLimit()} methods.
 * </UL>
 * <P>
 * <B>Running a job.</B>
 * To run a job, type this command:
 * <P>
 * <TT>$ java pj2 jar=<I>file</I> MainJob <I>arg</I> ...</TT>
 * <P>
 * The above command runs class MainJob, a subclass of class Job. Replace
 * <TT>MainJob</TT> with the fully-qualified class name of the desired job
 * subclass. The {@link pj2 pj2} program creates an instance of the job
 * subclass, then calls the job's {@link #main(String[]) main()} method, passing
 * in an array of the command line argument strings (zero or more). The {@link
 * #main(String[]) main()} method defines one or more {@linkplain Rule Rule}s
 * that specify {@linkplain Task Task}s that carry out the computation. (The
 * <TT>jar=<I>file</I></TT> option is described later.)
 * <P>
 * <B>Job execution.</B>
 * After calling the job's {@linkplain #main(String[]) main()} method, the
 * {@link pj2 pj2} program proceeds to execute the job. The job maintains a
 * repository of {@linkplain Tuple Tuple}s known as <I>tuple space.</I> The
 * job's execution is driven by <I>firing</I> the job's rules, based on tuples
 * that are written into tuple space. There are three kinds of rules:
 * <UL>
 * <P><LI>
 * <I>Start rule</I>&mdash;A rule that fires once when the job starts.
 * <P><LI>
 * <I>On-demand rule</I>&mdash;A rule that fires when a certain tuple or tuples
 * are present in tuple space.
 * <P><LI>
 * <I>Finish rule</I>&mdash;A rule that fires once when the job finishes.
 * </UL>
 * <P>
 * Each rule has an associated <I>task group</I> consisting of one or more
 * tasks. When a rule fires, for each task in the rule's task group, an instance
 * of the task is constructed, and the task's {@link Task#main(String[]) main()}
 * method is called to carry out the task's computation. For an on-demand rule,
 * the tuples that caused the rule to fire are automatically taken out of tuple
 * space and are provided to all tasks in the rule's task group. Any task may
 * also read or take tuples from tuple space as it executes. Any task may put
 * tuples into tuple space, which may trigger further rules to fire. For further
 * information about the criteria for firing rules and the actions performed
 * when firing rules, see class {@linkplain Rule Rule}.
 * <P>
 * The tasks in a rule's task group are guaranteed to start at the same time. If
 * there are not sufficient idle resources to execute all the tasks when the
 * rule fires, the tasks are put in a queue. Later, when sufficient idle
 * resources become available, the tasks are taken out of the queue and started
 * all at once.
 * <P>
 * Tuple space occupies memory solely inside the job's process. The Java Virtual
 * Machine that runs the job must be configured with a heap size large enough to
 * hold the tuples that exist in tuple space at each instant.
 * <P>
 * <B>Tracker host and port.</B>
 * A job's tasks are executed on one or more <I>nodes.</I> There is a
 * <I>Tracker</I> that keeps track of which jobs and tasks are executing on
 * which nodes, that assigns newly created tasks to execute on available nodes,
 * and that maintains the queue of tasks waiting for resources to become
 * available. By default, the Tracker listens for connections on one of the
 * local machine's network interfaces and port 20618. By default, the Job tries
 * to set up a connection to the Tracker on each of the local machine's network
 * interfaces and port 20618, stopping as soon as it successfully establishes a
 * connection. You can override the default and specify a particular Tracker
 * host on the Job's {@link pj2 pj2} command line as follows:
 * <P>
 * <TT>$ java pj2 tracker=<I>host</I> MainJob <I>arg</I> ...</TT>
 * <P>
 * You can specify a particular Tracker host and port on the Job's {@link pj2
 * pj2} command line as follows:
 * <P>
 * <TT>$ java pj2 tracker=<I>host</I>:<I>port</I> MainJob <I>arg</I> ...</TT>
 * <P>
 * If the job cannot set up a connection to the Tracker host and port, or if
 * <TT>tracker=none</TT> was specified, the job falls back to executing all
 * tasks in the same process as the job itself. In this case there is no
 * coordination among separate jobs. For further information about the Tracker,
 * see package {@linkplain edu.rit.pj2.tracker edu.rit.pj2.tracker}.
 * <P>
 * <B>Listening host and port.</B>
 * The job itself listens to a certain host and port for connections from the
 * nodes executing the job's tasks. By default, the job listens to the same IP
 * address as the Tracker connection and an unused port. You can specify a
 * particular listening host on the Job's {@link pj2 pj2} command line as
 * follows:
 * <P>
 * <TT>$ java pj2 listen=<I>host</I> MainJob <I>arg</I> ...</TT>
 * <P>
 * You can specify a particular listening host and port on the Job's {@link pj2
 * pj2} command line as follows:
 * <P>
 * <TT>$ java pj2 listen=<I>host</I>:<I>port</I> MainJob <I>arg</I> ...</TT>
 * <P>
 * <B>Firewalls.</B>
 * If the job's host is running a firewall, the firewall must be configured to
 * allow incoming connections to the job's listening host and port.
 * <P>
 * <B>JAR file.</B>
 * The <TT>jar=<I>file</I></TT> option specifies the Java archive (JAR) file
 * that contains the Java class files for the job and all its tasks. This JAR
 * file does not include the PJ2 Library classes. This JAR file is automatically
 * added to the {@link pj2 pj2} program's Java class path. In addition, when a
 * task is executed on a node, this JAR file is sent to the node and is
 * automatically added to the Java class path of the process that is executing
 * the task. An instance of class {@linkplain JarClassLoader JarClassLoader} is
 * used to load classes and resources from the JAR file. If the
 * <TT>jar=<I>file</I></TT> option is not specified, no JAR file is sent to the
 * node, which means the node might not be able to load the classes needed to
 * execute the task. (The <TT>jar=<I>file</I></TT> option might not be needed if
 * the nodes can obtain the necessary classes another way.) The JAR file can
 * also be specified by calling the {@link #jar(File) jar()} method.
 * <P>
 * <B>Debugging messages.</B>
 * A job can print various debugging messages on the standard error stream. Call
 * the {@link #debug(Debug[]) debug()} method to specify the debugging
 * message(s) to print. You can also specify debugging message(s) on the {@link
 * pj2 pj2} command line as follows, where <TT><I>message</I></TT> is one of the
 * debugging messages in enum {@linkplain Debug Debug}:
 * <P>
 * <TT>$ java pj2 debug=<I>message</I>[,<I>message</I>...] MainJob <I>arg</I> ...</TT>
 * <P>
 * <B>I/O in a job.</B>
 * Code in a job can print on <TT>System.out</TT> or <TT>System.err</TT>, and
 * the printouts appear on the job's process's console as usual. Code in a task
 * launched by a job can also print on <TT>System.out</TT> or
 * <TT>System.err</TT>, and the printouts are automatically redirected to the
 * job's process's console. The printouts actually appear on the console when
 * the job terminates; to make the printouts appear earlier, call
 * <TT>System.out.flush()</TT> or <TT>System.err.flush()</TT>.
 * <P>
 * Other I/O, such as reading from the job's standard input, reading files, and
 * writing files, might need to be done in a task that runs in the job's
 * process. This would be necessary, for example, if tasks running on separate
 * nodes do not run in the user's account and so cannot access the user's files.
 * In such cases:
 * <UL>
 * <P><LI>
 * To read the standard input or the contents of a file, include rule(s) in the
 * job that execute task(s) that read the data from the desired source. These
 * task(s) must be specified with {@link TaskSpec#runInJobProcess()
 * runInJobProcess()} to ensure that the task(s) have access to the job's
 * standard input or the files in the user's account. These task(s) put the data
 * into their output tuple(s). Also include rule(s) in the job that match these
 * tuple(s) and that execute task(s) to process the data. These task(s) obtain
 * the data from their input tuple(s).
 * <P><LI>
 * To write the contents of a file, include rule(s) in the job that that execute
 * task(s) to generate the data to be written. These task(s) put the data into
 * their output tuple(s). Also include rule(s) in the job that match these
 * tuple(s) and that execute task(s) to write the data to the desired
 * destination. These task(s) must be specified with {@link
 * TaskSpec#runInJobProcess() runInJobProcess()} to ensure that the
 * task(s) have access to files in the user's account. These task(s) obtain the
 * data from their input tuple(s).
 * </UL>
 * <P>
 * <B>JVM heap size in a job.</B>
 * When one of the job's tasks is executed on a node, by default, the Java
 * Virtual Machine (JVM) that will run the task is created with the
 * <TT>java</TT> command. This means that the maximum JVM heap size will be set
 * to the default size. If the task needs, say, a 2-gigabyte heap instead, call
 * the {@link TaskSpec#jvmFlags(String[]) jvmFlags("-Xmx2000m")} method in the
 * {@linkplain TaskSpec TaskSpec} in the {@linkplain Rule Rule} that launches
 * the task. Then the JVM that will run the task is created with the
 * <TT>java&nbsp;-Xmx2000m</TT> command, which sets the maximum JVM heap size to
 * 2 gigabytes. (You can specify any desired JVM flags for the task in this
 * manner.)
 * <P>
 * You can specify JVM flags for all tasks in the job by calling the job's
 * {@link #jvmFlags(String[]) jvmFlags()} method. You can also specify JVM flags
 * for all tasks in the job by including the <TT>jvmFlags</TT> parameter on the
 * {@link pj2 pj2} command line. (Specifying an individual task spec's JVM flags
 * overrides those of the job.)
 * <P>
 * <B>Abnormal termination.</B>
 * If code in a job or task throws an exception that propagates out of the
 * <TT>main()</TT> method, the {@link pj2 pj2} program prints an exception stack
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
 * @author  Alan Kaminsky
 * @version 01-Oct-2015
 */
public abstract class Job
	{

// Exported constants.

	/**
	 * Indicates that the <TT>workers</TT> property is defaulted.
	 */
	public static final int DEFAULT_WORKERS = JobProperties.DEFAULT_THREADS;

	/**
	 * Indicates that the <TT>threads</TT> property is defaulted.
	 */
	public static final int DEFAULT_THREADS = JobProperties.DEFAULT_THREADS;

	/**
	 * Indicates that the <TT>masterSchedule</TT> or <TT>schedule</TT> property
	 * is defaulted.
	 */
	public static final Schedule DEFAULT_SCHEDULE =
		JobProperties.DEFAULT_SCHEDULE;

	/**
	 * Indicates that the <TT>masterChunk</TT> or <TT>chunk</TT> property is
	 * defaulted.
	 */
	public static final int DEFAULT_CHUNK = JobProperties.DEFAULT_CHUNK;

	/**
	 * Indicates that a parallel for loop will be executed by as many threads as
	 * there are cores on the machine.
	 */
	public static final int THREADS_EQUALS_CORES =
		JobProperties.THREADS_EQUALS_CORES;

	/**
	 * Indicates to use the standard chunk size for the <TT>schedule</TT>
	 * property.
	 */
	public static final int STANDARD_CHUNK = JobProperties.STANDARD_CHUNK;

	/**
	 * Indicates that the <TT>nodeName</TT> property is defaulted.
	 */
	public static final String DEFAULT_NODE_NAME =
		JobProperties.DEFAULT_NODE_NAME;

	/**
	 * Indicates that the <TT>cores</TT> property is defaulted.
	 */
	public static final int DEFAULT_CORES = JobProperties.DEFAULT_CORES;

	/**
	 * Indicates that the <TT>gpus</TT> property is defaulted.
	 */
	public static final int DEFAULT_GPUS = JobProperties.DEFAULT_GPUS;

	/**
	 * Indicates that the task can run on any node of the cluster.
	 */
	public static final String ANY_NODE_NAME = JobProperties.ANY_NODE_NAME;

	/**
	 * Indicates that the task requires all the cores on the node.
	 */
	public static final int ALL_CORES = JobProperties.ALL_CORES;

	/**
	 * Indicates that the task requires all the GPU accelerators on the node.
	 */
	public static final int ALL_GPUS = JobProperties.ALL_GPUS;

	/**
	 * Fixed schedule. The iterations are partitioned into as many chunks as
	 * there are threads. Each chunk is the same size (except possibly the last
	 * chunk). Each thread performs a different chunk of iterations. The
	 * <TT>chunk</TT> property is not used.
	 * <P>
	 * A fixed schedule is appropriate when each loop iteration takes the same
	 * amount of time, so load balancing is not needed; and when each thread
	 * should do a contiguous range of loop indexes.
	 * @see Schedule#fixed
	 */
	public static final Schedule fixed = Schedule.fixed;

	/**
	 * Leapfrog schedule. Let the loop index lower bound be <I>L</I> and the
	 * number of threads be <I>K</I>; then thread 0 performs the iterations for
	 * indexes <I>L</I>, <I>L</I>+<I>K</I>, <I>L</I>+2<I>K</I>, ...; thread 1
	 * performs the iterations for indexes <I>L</I>+1, <I>L</I>+1+<I>K</I>,
	 * <I>L</I>+1+2<I>K</I>, ...; thread 2 performs the iterations for indexes
	 * <I>L</I>+2, <I>L</I>+2+<I>K</I>, <I>L</I>+2+2<I>K</I>, ...; and so on.
	 * The <TT>chunk</TT> property is not used.
	 * <P>
	 * A leapfrog schedule is appropriate when each loop iteration takes the
	 * same amount of time, so load balancing is not needed; and when each
	 * thread should do noncontiguous loop indexes with a stride of <I>K</I>.
	 * @see Schedule#leapfrog
	 */
	public static final Schedule leapfrog = Schedule.leapfrog;

	/**
	 * Dynamic schedule. The iterations are partitioned into chunks. Each chunk
	 * is the same size (except possibly the last chunk) as specified by the
	 * <TT>chunk</TT> property. If the <TT>chunk</TT> property is {@link
	 * Task#STANDARD_CHUNK STANDARD_CHUNK}, a chunk size of 1 is used. The
	 * threads execute the chunks in a dynamic fashion: when a thread has
	 * finished executing a chunk, the thread gets the next available chunk and
	 * executes it.
	 * <P>
	 * A dynamic schedule is appropriate when each loop iteration takes a
	 * different amount of time, so load balancing is needed. However, be
	 * careful when choosing the chunk size. A smaller chunk size can yield a
	 * better balanced load; a larger chunk size requires less thread
	 * synchronization overhead.
	 * @see Schedule#dynamic
	 */
	public static final Schedule dynamic = Schedule.dynamic;

	/**
	 * Proportional schedule. The iterations are partitioned into chunks. Each
	 * chunk is the same size (except possibly the last chunk). The
	 * <TT>chunk</TT> property specifies the <I>chunk factor.</I> If the
	 * <TT>chunk</TT> property is {@link Task#STANDARD_CHUNK STANDARD_CHUNK}, a
	 * chunk factor of 10 is used. The number of chunks is the number of threads
	 * times the chunk factor. The chunk size is the number of iterations
	 * divided by the number of chunks. The threads execute the chunks in a
	 * dynamic fashion: when a thread has finished executing a chunk, the thread
	 * gets the next available chunk and executes it.
	 * <P>
	 * A proportional schedule is appropriate when each loop iteration takes a
	 * different amount of time, so load balancing is needed. However, be
	 * careful when choosing the chunk factor. A larger chunk factor can yield a
	 * better balanced load; a smaller chunk factor requires less thread
	 * synchronization overhead.
	 * @see Schedule#proportional
	 */
	public static final Schedule proportional = Schedule.proportional;

	/**
	 * Guided schedule. The iterations are partitioned into chunks. The chunks
	 * are of different sizes; earlier chunks are larger, later chunks are
	 * smaller. The minimum chunk size is specified by the <TT>chunk</TT>
	 * property. If the <TT>chunk</TT> property is {@link Task#STANDARD_CHUNK
	 * STANDARD_CHUNK}, a minimum chunk size of 1 is used. The threads execute
	 * the chunks in a dynamic fashion: when a thread has finished executing a
	 * chunk, the thread gets the next available chunk and executes it.
	 * <P>
	 * A guided schedule is appropriate when each loop iteration takes a
	 * different amount of time, so load balancing is needed. However, be
	 * careful when choosing the minimum chunk size. A smaller minimum chunk
	 * size can yield a better balanced load; a larger minimum chunk size
	 * requires less thread synchronization overhead.
	 * @see Schedule#guided
	 */
	public static final Schedule guided = Schedule.guided;

	/**
	 * Debug printout when the job is launched. Includes the job ID, time, and
	 * date. On by default.
	 * @see Debug#jobLaunch
	 */
	public static final Debug jobLaunch = Debug.jobLaunch;

	/**
	 * Debug printout when the job's first task starts executing. Includes the
	 * job ID, time, and date. On by default.
	 * @see Debug#jobStart
	 */
	public static final Debug jobStart = Debug.jobStart;

	/**
	 * Debug printout when a task is launched. Includes the job ID, task ID,
	 * time, and date. Off by default.
	 * @see Debug#taskLaunch
	 */
	public static final Debug taskLaunch = Debug.taskLaunch;

	/**
	 * Debug printout of a task's class name and command line arguments when a
	 * task is launched. Off by default.
	 * @see Debug#taskArguments
	 */
	public static final Debug taskArguments = Debug.taskArguments;

	/**
	 * Debug printout of a task's input tuples. This includes tuples the task
	 * takes or reads from tuple space, as well as the matching tuples that
	 * fired an on-demand task. Off by default.
	 * @see Debug#taskInputTuples
	 */
	public static final Debug taskInputTuples = Debug.taskInputTuples;

	/**
	 * Debug printout when a task starts executing. Includes the job ID, task
	 * ID, time, date, and node. Off by default.
	 * @see Debug#taskStart
	 */
	public static final Debug taskStart = Debug.taskStart;

	/**
	 * Debug printout when a task finishes executing. Includes the job ID, task
	 * ID, time, date, and elapsed time from task start to task finish. Off by
	 * default.
	 * @see Debug#taskFinish
	 */
	public static final Debug taskFinish = Debug.taskFinish;

	/**
	 * Debug printout of a task's output tuples. This includes tuples the task
	 * puts into tuple space. Off by default.
	 * @see Debug#taskOutputTuples
	 */
	public static final Debug taskOutputTuples = Debug.taskOutputTuples;

	/**
	 * Debug printout when the job finishes executing. Includes the job ID,
	 * time, date, and elapsed time from job launch to job finish. On by
	 * default.
	 * @see Debug#jobFinish
	 */
	public static final Debug jobFinish = Debug.jobFinish;

	/**
	 * Debug printout of any tuples remaining in tuple space when the job
	 * finishes executing. Off by default.
	 * @see Debug#remainingTuples
	 */
	public static final Debug remainingTuples = Debug.remainingTuples;

	/**
	 * Debug printout of the job's makespan. This is the elapsed time from when
	 * the first task actually started until when the last task actually
	 * finished. This might be different from the job's elapsed time if the job
	 * sat in the Tracker queue for a while. Off by default.
	 * @see Debug#makespan
	 */
	public static final Debug makespan = Debug.makespan;

// Hidden data members.

	// Rules of various kinds.
	private AList<Rule> allRules = new AList<Rule>();
	private AList<Rule> startRules = new AList<Rule>();
	private AList<Rule> onDemandRules = new AList<Rule>();
	private AList<Rule> finishRules = new AList<Rule>();

	// Job properties.
	private JobProperties properties = new JobProperties();

	// Tracker and its host and port.
	private String trackerHost = null;
	private int trackerPort = -1;
	private TrackerRef tracker;
	private Proxy trackerProxy;

	// Listening host and port, and server socket.
	private String listenHost = null;
	private int listenPort = 0;
	private ServerSocket serverSocket;

	// Activated debugging messages.
	private EnumSet<Debug> debugs = EnumSet.of (jobLaunch, jobStart, jobFinish);

	// Default JVM flags for all tasks.
	private String[] jvmFlags = new String [0];

	// JAR file and its contents.
	private File jarfile;
	private byte[] jar;

	// Time limit in seconds, and timeout thread.
	private int timeLimit = 0;
	private Thread timeoutThread = null;

	// Job user name.
	private String user;

	// Job ID.
	private long jobID;

	// Job elapsed time.
	private ElapsedTime jobET;
	private long minTaskStartTime = Long.MAX_VALUE;
	private long maxTaskFinishTime = Long.MIN_VALUE;

	// Task map and per-task extra data.
	private TaskMap taskMap;
	private static class ExtraData
		{
		public ElapsedTime ET;
		public Heartbeat heartbeat;
		public BackendRef backend;
		public EnumSet<Debug> debugs;
		public boolean informTracker;
		}

	// Tuple space.
	private TupleSpace tupleSpace = new TupleSpace();
	private TupleRequestMap tupleRequestMap = new TupleRequestMap();
	private static class TupleTakeInfo
		{
		public long taskID;
		public long requestID;
		public Tuple template;
		public boolean blocking;
		public boolean taking;
		}
	private static DList<TupleTakeInfo> tupleTakeList =
		new DList<TupleTakeInfo>();

	// Flag denoting the job is finished.
	private boolean jobIsFinished = false;

	// For blocking the PJ2 launcher program until the job is finished.
	private Semaphore finishSema = new Semaphore (0);
	private Throwable excThrown;
	private String errorMsg = "by user";

	// For printing a tuple on System.err.
	private static class PrintTuple implements Action<Tuple>
		{
		private int i = 0;
		public void run (Tuple tuple)
			{
			System.err.printf ("      [%d] = ", i ++);
			tuple.dump (System.err, 3);
			System.err.flush();
			}
		};

	// For multiple thread safe console printing.
	private ThreadedOutputStream thrOut;
	private ThreadedOutputStream thrErr;

	// For exchanging heartbeats with the Tracker.
	private ScheduledExecutorService executor;
	private TrackerHeartbeat trackerHeartbeat;
	private class TrackerHeartbeat extends Heartbeat
		{
		public TrackerHeartbeat()
			{
			super();
			}
		protected void sendHeartbeat() throws IOException
			{
			tracker.heartbeatFromJob (jobID);
			}
		protected void died()
			{
			jobET.stop();
			stopJob (new HeartbeatFailedException (String.format
				("Job %d tracker heartbeat failed", jobID)));
			}
		}

	// For timing out waiting for a task to start.
	private class TaskStartHeartbeat extends Heartbeat
		{
		private long taskID;
		public TaskStartHeartbeat
			(long taskID)
			{
			super();
			this.taskID = taskID;
			}
		protected void sendHeartbeat()
			{
			}
		protected void died()
			{
			jobET.stop();
			stopJob (new LaunchException (String.format
				("Job %d task %d launch failed", jobID, taskID)));
			}
		}

	// For exchanging heartbeats with a Backend.
	private class BackendHeartbeat extends Heartbeat
		{
		private BackendRef backend;
		private long taskID;
		public BackendHeartbeat
			(BackendRef backend,
			 long taskID)
			{
			super();
			this.backend = backend;
			this.taskID = taskID;
			}
		protected void sendHeartbeat() throws IOException
			{
			if (backend != null)
				backend.heartbeatFromJob();
			}
		protected void died()
			{
			jobET.stop();
			stopJob (new HeartbeatFailedException (String.format
				("Job %d task %d backend heartbeat failed", jobID, taskID)));
			}
		}

	// Glue object to let the Tracker, Launcher, and Backend invoke methods in
	// interface JobRef on this job.
	private JobRef jobRef = new JobRef()
		{
		public void jobLaunched
			(long jobID)
			throws IOException
			{
			Job.this.jobLaunched (jobID);
			}
		public void jobStarted()
			throws IOException
			{
			Job.this.jobStarted();
			}
		public void taskLaunching
			(long taskID,
			 int[] devnum,
			 boolean runInJobProcess)
			throws IOException
			{
			Job.this.taskLaunching (taskID, devnum, runInJobProcess);
			}
		public void taskLaunched
			(BackendRef backend,
			 long taskID,
			 String name)
			throws IOException
			{
			Job.this.taskLaunched (backend, taskID, name);
			}
		public void takeTuple
			(long taskID,
			 long requestID,
			 Tuple template,
			 boolean blocking,
			 boolean taking)
			throws IOException
			{
			Job.this.takeTuple (taskID, requestID, template, blocking, taking);
			}
		public void writeTuple
			(long taskID,
			 Tuple tuple,
			 int copies)
			throws IOException
			{
			Job.this.writeTuple (taskID, tuple, copies);
			}
		public void taskFinished
			(long taskID)
			throws IOException
			{
			Job.this.taskFinished (taskID);
			}
		public void taskFailed
			(long taskID,
			 Throwable exc)
			throws IOException
			{
			Job.this.taskFailed (taskID, exc);
			}
		public void heartbeatFromTracker()
			throws IOException
			{
			Job.this.heartbeatFromTracker();
			}
		public void heartbeatFromTask
			(long taskID)
			throws IOException
			{
			Job.this.heartbeatFromTask (taskID);
			}
		public void writeStandardStream
			(int stream,
			 int len,
			 byte[] data)
			throws IOException
			{
			Job.this.writeStandardStream (stream, len, data);
			}
		public void jobFailed
			(Throwable exc)
			{
			Job.this.jobFailed (exc);
			}
		public String host()
			{
			return null;
			}
		public int port()
			{
			return 0;
			}
		public void shutdown()
			{
			}
		public void terminate()
			{
			}
		};

	// For cleaning up when the job process exits.
	private Thread shutdownHook = new Thread()
		{
		public void run()
			{
//System.err.printf ("Running Job shutdownHook%n");
//System.err.flush();
			// Stop job timeout thread if any.
			if (timeoutThread != null)
				timeoutThread.interrupt();

			// For each task (if any remain), stop backend heartbeats and tell
			// the backend to stop the task.
//System.err.printf ("Stopping tasks%n");
//System.err.flush();
			if (taskMap != null)
				taskMap.forEachItemDo (new Action<TaskInfo>()
					{
					public void run (TaskInfo info)
						{
//System.err.printf ("Stopping task %d%n", info.taskID);
//System.err.flush();
						ExtraData extra = (ExtraData) info.moreData;
						if (extra.heartbeat != null)
							extra.heartbeat.cancel();
						if (extra.backend != null)
							{
							try { extra.backend.stopTask(); }
								catch (IOException exc) {}
							extra.backend.shutdown();
							}
						}
					});

			// Stop tracker heartbeats.
//System.err.printf ("Stopping tracker heartbeats%n");
//System.err.flush();
			if (trackerHeartbeat != null)
				trackerHeartbeat.cancel();

			// Tell the tracker that the job is done.
			if (tracker != null)
				{
				try
					{
					if (errorMsg == null)
						{
//System.err.printf ("Calling tracker.jobDone(%d)%n", jobID);
//System.err.flush();
						tracker.jobDone (jobID);
						}
					else
						{
//System.err.printf ("Calling tracker.stopJob(%d,\"%s\")%n", jobID, errorMsg);
//System.err.flush();
						tracker.stopJob (jobID, errorMsg);
						}
					}
				catch (IOException exc)
					{
					}

				// Shut down the tracker connection.
//System.err.printf ("Calling tracker.close()%n");
//System.err.flush();
				tracker.shutdown();
				}

			// Shut down the executor.
//System.err.printf ("Shutting down executor%n");
//System.err.flush();
			if (executor != null)
				executor.shutdownNow();

			// Flush console streams.
			flushConsoleStreams();
			}
		};

// Exported constructors.

	/**
	 * Construct a new job.
	 */
	public Job()
		{
		}

// Exported operations.

	/**
	 * Configure the rules and other settings for this job.
	 * <P>
	 * <B><I>Warning:</I></B> The <TT>main()</TT> method is intended to be
	 * called only by the {@link pj2 pj2} launcher program. Do not call the
	 * <TT>main()</TT> method yourself.
	 *
	 * @param  args  Array of zero or more command line argument strings.
	 *
	 * @exception  Exception
	 *     The <TT>main()</TT> method can throw any exception.
	 */
	public abstract void main
		(String[] args)
		throws Exception;

	/**
	 * Execute this job.
	 * <P>
	 * <B><I>Warning:</I></B> The <TT>execute()</TT> method is intended to be
	 * called only by the {@link pj2 pj2} launcher program. Do not call the
	 * <TT>execute()</TT> method yourself.
	 *
	 * @exception  Exception
	 *     The <TT>execute()</TT> method can throw any exception.
	 */
	public final void execute()
		throws Exception
		{
		try
			{
			// Validate rules.
			if (allRules.isEmpty())
				throw new IllegalArgumentException
					("No rules specified");
			allRules.forEachItemDo (new Action<Rule>()
				{
				private int rulenum = 0;
				public void run (Rule rule)
					{
					++ rulenum;
					if (rule.taskGroup.isEmpty())
						throw new IllegalArgumentException (String.format
							("Rule %d does not specify any tasks",
							 rulenum));
					if (rule.atStart)
						startRules.addLast (rule);
					else if (rule.atFinish)
						finishRules.addLast (rule);
					else if (rule.templates == null)
						throw new IllegalArgumentException (String.format
							("Rule %d does not specify firing condition",
							 rulenum));
					else
						onDemandRules.addLast (rule);
					}
				});

			// Set up multiple thread safe console streams.
			thrOut = new ThreadedOutputStream (System.out);
			thrErr = new ThreadedOutputStream (System.err);
			System.setOut (new PrintStream (thrOut, false));
			System.setErr (new PrintStream (thrErr, false));

			// Set up to terminate job on exit.
			Runtime.getRuntime().addShutdownHook (shutdownHook);

			// Get job's user name.
			user = System.getProperty ("user.name");
			if (user == null) user = "&lt;unknown&gt;";

			// Record job start time.
			jobET = new ElapsedTime();

			// Set up tracker.
			if ("none".equals (trackerHost))
				{
				if (debugs.contains (Debug.jobLaunch))
					{
					System.err.printf
						("No Tracker; job will run in this process%n");
					System.err.flush();
					}
				tracker = null; // No tracker
				executor = null; // Do not exchange heartbeats
				}
			else
				{
				try
					{
					tracker = new TrackerSender();
					trackerProxy = new Proxy
						(connectToTracker(),
						 (TrackerSender)tracker,
						 new JobReceiver
							(new ReceiverListener()
								{
								public void receiverFailed
									(Receiver receiver,
									 Throwable exc)
									{
									System.err.printf
										("Failure while receiving a Tracker message%n");
									exc.printStackTrace (System.err);
									System.err.flush();
									stopJob (exc);
									}
								},
							 jobRef));
					executor = Executors.newSingleThreadScheduledExecutor();
					}
				catch (IOException exc)
					{
					if (debugs.contains (Debug.jobLaunch))
						{
						System.err.printf
							("No Tracker at %s:%d; job will run in this process%n",
							 trackerHost, trackerPort);
						System.err.flush();
						}
					tracker = null; // No tracker
					executor = null; // Do not exchange heartbeats
					}
				}

			// If remote tracker exists,
			if (tracker != null)
				{
				// If listen host was not specified, use same host as Tracker.
				if (listenHost == null) listenHost = trackerHost;

				// Listen for connections from Backends.
				try
					{
					serverSocket = new ServerSocket();
					serverSocket.bind (new InetSocketAddress
						(listenHost, listenPort));
					listenPort = serverSocket.getLocalPort();
					new ListenThread() .start();
					}
				catch (IOException exc)
					{
					System.err.printf ("Cannot listen to %s:%d%n",
						listenHost, listenPort);
					System.err.flush();
					throw exc;
					}

				// Launch job.
				tracker.launchJob (jobRef, user, listenHost, listenPort);
				}

			// If remote tracker does not exist,
			else
				{
				jobLaunched (1L);
				jobStarted();
				}

			// Set up tracker heartbeats.
			trackerHeartbeat = new TrackerHeartbeat();
			trackerHeartbeat.enable (executor);
			}

		// Capture any exception thrown above.
		catch (Throwable exc)
			{
			stopJob (exc);
			}

		// Wait until the job is finished.
		finishSema.acquireUninterruptibly();

		// If there was an exception, re-throw it with an error message,
		// otherwise return normally.
		if (excThrown == null)
			errorMsg = null;
		else
			{
			errorMsg = "due to exception: " + excThrown;
			if (excThrown instanceof Error)
				throw (Error)excThrown;
			else if (excThrown instanceof RuntimeException)
				throw (RuntimeException)excThrown;
			else
				throw (Exception)excThrown;
			}
		}

	/**
	 * Add a rule to this job. The <TT>rule()</TT> method returns the rule. Call
	 * methods on the returned {@linkplain Rule} object to configure the rule.
	 *
	 * @return  Rule.
	 */
	public Rule rule()
		{
		Rule rule = new Rule (properties, jvmFlags);
		allRules.addLast (rule);
		return rule;
		}

	/**
	 * Add a rule to this job and configure the rule. The given {@linkplain
	 * Action Action}'s {@link Action#run(Object) run()} method is called,
	 * passing in the rule; the <TT>run()</TT> method can configure the rule.
	 * The <TT>rule()</TT> method then returns the rule. If necessary, call
	 * methods on the returned {@linkplain Rule} object to further configure the
	 * rule.
	 *
	 * @param  action  Action.
	 *
	 * @return  Rule.
	 */
	public Rule rule
		(Action<Rule> action)
		{
		Rule rule = new Rule (properties, jvmFlags);
		allRules.addLast (rule);
		action.run (rule);
		return rule;
		}

	/**
	 * Put the given tuple into tuple space.
	 *
	 * @param  tuple  Tuple.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>tuple</TT> is null.
	 */
	public void putTuple
		(Tuple tuple)
		{
		tupleSpace.putTuple (tuple);
		}

	/**
	 * Put multiple copies of the given tuple into tuple space. If
	 * <TT>copies</TT> = 0, this method does nothing.
	 *
	 * @param  copies  Number of copies (&ge; 0).
	 * @param  tuple   Tuple.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>tuple</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>copies</TT> &lt; 0.
	 */
	public void putTuple
		(int copies,
		 Tuple tuple)
		{
		if (copies < 0)
			throw new IllegalArgumentException (String.format
				("Job.putTuple(): copies = %d illegal", copies));
		for (int i = 0; i < copies; ++ i)
			tupleSpace.putTuple (tuple);
		}

	/**
	 * Add a master-worker cluster parallel for loop to this job, with a loop
	 * index of type <TT>int</TT>. To complete the master-worker cluster
	 * parallel for loop, the worker task must call the {@link Task#workerFor()
	 * workerFor()} method. For further information, see class {@linkplain
	 * WorkerParallelForLoop}.
	 * <P>
	 * If <TT>lb</TT> &gt; <TT>ub</TT>, it represents a loop with no iterations.
	 * If <TT>lb</TT> &le; <TT>ub</TT>, it represents a loop with one or more
	 * iterations.
	 * <P>
	 * <B><I>Warning:</I></B> There can be only one master-worker cluster
	 * parallel for loop in a job.
	 *
	 * @param  <T>        Worker task data type.
	 * @param  lb         Loop index lower bound.
	 * @param  ub         Loop index upper bound.
	 * @param  taskClass  Worker task class.
	 *
	 * @return  Task specification for the worker task class.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>taskClass</TT> is null.
	 */
	public <T extends Task> TaskSpec masterFor
		(int lb,
		 int ub,
		 Class<T> taskClass)
		{
		// Verify preconditions.
		if (taskClass == null)
			throw new NullPointerException
				("Job.masterFor(): taskClass is null");

		// Determine number of workers.
		int W = workers();

		// Partition loop index range and put chunk tuples into tuple space.
		masterSchedule().putChunks (this, W, masterChunk(), lb, ub);

		// Add rule and return task specification.
		return rule() .task (W, taskClass);
		}

	/**
	 * Add a master-worker cluster parallel for loop to this job, with a loop
	 * index of type <TT>long</TT>. To complete the master-worker cluster
	 * parallel for loop, the worker task must call the {@link Task#workerFor()
	 * workerFor()} method. For further information, see class {@linkplain
	 * WorkerParallelForLoop}.
	 * <P>
	 * If <TT>lb</TT> &gt; <TT>ub</TT>, it represents a loop with no iterations.
	 * If <TT>lb</TT> &le; <TT>ub</TT>, it represents a loop with one or more
	 * iterations.
	 * <P>
	 * <B><I>Warning:</I></B> There can be only one master-worker cluster
	 * parallel for loop in a job.
	 *
	 * @param  <T>        Worker task data type.
	 * @param  lb         Loop index lower bound.
	 * @param  ub         Loop index upper bound.
	 * @param  taskClass  Worker task class.
	 *
	 * @return  Task specification for the worker task class.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>taskClass</TT> is null.
	 */
	public <T extends Task> TaskSpec masterFor
		(long lb,
		 long ub,
		 Class<T> taskClass)
		{
		// Verify preconditions.
		if (taskClass == null)
			throw new NullPointerException
				("Job.masterFor(): taskClass is null");

		// Determine number of workers.
		int W = workers();

		// Partition loop index range and put chunk tuples into tuple space.
		masterSchedule().putChunks (this, W, masterChunk(), lb, ub);

		// Add rule and return task specification.
		return rule() .task (W, taskClass);
		}

	/**
	 * Set this job's <TT>threads</TT> property. The <TT>threads</TT> property
	 * specifies the number of threads that will execute a parallel for loop.
	 * <P>
	 * The job's <TT>threads</TT> property specifies the <TT>threads</TT>
	 * property for each task in the job. Specifying the <TT>threads</TT>
	 * property for a particular task overrides the job's <TT>threads</TT>
	 * property for that task.
	 *
	 * @param  threads  Number of threads (&ge; 1), {@link
	 *                  #THREADS_EQUALS_CORES}, or {@link #DEFAULT_THREADS}.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>threads</TT> is illegal.
	 *
	 * @see  Task#threads(int)
	 * @see  #threads()
	 */
	public void threads
		(int threads)
		{
		properties.threads (threads);
		}

	/**
	 * Get this job's <TT>threads</TT> property. The <TT>threads</TT> property
	 * specifies the number of threads that will execute a parallel for loop.
	 * The default is one thread for each core of the machine on which the
	 * program is running.
	 *
	 * @return  Number of threads (&ge; 1), or {@link #THREADS_EQUALS_CORES}.
	 *
	 * @see  Task#threads()
	 * @see  #threads(int)
	 */
	public int threads()
		{
		return properties.threads();
		}

	/**
	 * Set this job's <TT>schedule</TT> property. The <TT>schedule</TT>
	 * property, along with the <TT>chunk</TT> property, specifies how the
	 * iterations of a parallel for loop are partitioned among the threads
	 * executing the parallel for loop. Refer to enum {@linkplain Schedule} for
	 * descriptions of the possible schedules.
	 * <P>
	 * The job's <TT>schedule</TT> property specifies the <TT>schedule</TT>
	 * property for each task in the job. Specifying the <TT>schedule</TT>
	 * property for a particular task overrides the job's <TT>schedule</TT>
	 * property for that task.
	 *
	 * @param  schedule  Parallel for loop schedule, or {@link
	 *                   #DEFAULT_SCHEDULE}.
	 *
	 * @see  Task#schedule(Schedule)
	 * @see  #schedule()
	 * @see  #chunk(int)
	 * @see  #chunk()
	 */
	public void schedule
		(Schedule schedule)
		{
		properties.schedule (schedule);
		}

	/**
	 * Get this job's <TT>schedule</TT> property. The <TT>schedule</TT>
	 * property, along with the <TT>chunk</TT> property, specifies how the
	 * iterations of a parallel for loop are partitioned among the threads
	 * executing the parallel for loop. The default is a {@link Schedule#fixed
	 * fixed} schedule.
	 *
	 * @return  Parallel for loop schedule.
	 *
	 * @see  Task#schedule()
	 * @see  #schedule(Schedule)
	 * @see  #chunk()
	 */
	public Schedule schedule()
		{
		return properties.schedule();
		}

	/**
	 * Set this job's <TT>chunk</TT> property. The <TT>chunk</TT> property,
	 * along with the <TT>schedule</TT> property, specifies how the iterations
	 * of a parallel for loop are partitioned among the threads executing the
	 * parallel for loop. Refer to enum {@linkplain Schedule} for descriptions
	 * of the possible schedules.
	 * <P>
	 * The job's <TT>chunk</TT> property specifies the <TT>chunk</TT> property
	 * for each task in the job. Specifying the <TT>chunk</TT> property for a
	 * particular task overrides the job's <TT>chunk</TT> property for that
	 * task.
	 *
	 * @param  chunk  Chunk size (&ge; 1), {@link #STANDARD_CHUNK}, or {@link
	 *                #DEFAULT_CHUNK}.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>chunk</TT> is illegal.
	 *
	 * @see  Task#chunk(int)
	 * @see  #schedule(Schedule)
	 * @see  #schedule()
	 * @see  #chunk()
	 */
	public void chunk
		(int chunk)
		{
		properties.chunk (chunk);
		}

	/**
	 * Get this job's <TT>chunk</TT> property. The <TT>chunk</TT> property,
	 * along with the <TT>schedule</TT> property, specifies how the iterations
	 * of a parallel for loop are partitioned among the threads executing the
	 * parallel for loop. The default is the standard chunk size for the
	 * <TT>schedule</TT> property; see enum {@linkplain Schedule Schedule} for
	 * further information.
	 *
	 * @return  Chunk size (&ge; 1), or {@link #STANDARD_CHUNK}.
	 *
	 * @see  Task#chunk()
	 * @see  #chunk(int)
	 * @see  #schedule()
	 */
	public int chunk()
		{
		return properties.chunk();
		}

	/**
	 * Set this job's <TT>nodeName</TT> property. The <TT>nodeName</TT> property
	 * specifies the name of the cluster node on which a task must run.
	 * <P>
	 * The job's <TT>nodeName</TT> property specifies the <TT>nodeName</TT>
	 * property for each task in the job. Specifying the <TT>nodeName</TT>
	 * property for a particular task overrides the job's <TT>nodeName</TT>
	 * property for that task.
	 *
	 * @param  nodeName  Node name, {@link #ANY_NODE_NAME}, or {@link
	 *                   #DEFAULT_NODE_NAME}.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>nodeName</TT> is null.
	 *
	 * @see  #nodeName()
	 */
	public void nodeName
		(String nodeName)
		{
		properties.nodeName (nodeName);
		}

	/**
	 * Get this job's <TT>nodeName</TT> property. The <TT>nodeName</TT> property
	 * specifies the name of the cluster node on which a task must run. The
	 * default is that a task can run on any node of the cluster.
	 *
	 * @return  Node name, or {@link #ANY_NODE_NAME}.
	 *
	 * @see  #nodeName(String)
	 */
	public String nodeName()
		{
		return properties.nodeName();
		}

	/**
	 * Set this job's <TT>cores</TT> property. The <TT>cores</TT> property
	 * specifies the number of CPU cores a task requires.
	 * <P>
	 * The job's <TT>cores</TT> property specifies the <TT>cores</TT> property
	 * for each task in the job. Specifying the <TT>cores</TT> property for a
	 * particular task overrides the job's <TT>cores</TT> property for that
	 * task.
	 *
	 * @param  cores  Number of cores (&ge; 1), {@link #ALL_CORES}, or {@link
	 *                #DEFAULT_CORES}.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>cores</TT> is illegal.
	 *
	 * @see  #cores()
	 */
	public void cores
		(int cores)
		{
		properties.cores (cores);
		}

	/**
	 * Get this task's <TT>cores</TT> property. The <TT>cores</TT> property
	 * specifies the number of CPU cores a task requires. The default is that a
	 * task requires all the cores on the node.
	 *
	 * @return  Number of cores (&ge; 1), or {@link #ALL_CORES}.
	 *
	 * @see  #cores(int)
	 */
	public int cores()
		{
		return properties.cores();
		}

	/**
	 * Set this job's <TT>gpus</TT> property. The <TT>gpus</TT> property
	 * specifies the number of GPU accelerators a task requires.
	 * <P>
	 * The job's <TT>gpus</TT> property specifies the <TT>gpus</TT> property for
	 * each task in the job. Specifying the <TT>gpus</TT> property for a
	 * particular task overrides the job's <TT>gpus</TT> property for that task.
	 *
	 * @param  gpus  Number of GPUs (&ge; 0), {@link #ALL_GPUS}, or {@link
	 *               #DEFAULT_GPUS}.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>gpus</TT> is illegal.
	 *
	 * @see  #gpus()
	 */
	public void gpus
		(int gpus)
		{
		properties.gpus (gpus);
		}

	/**
	 * Get this job's <TT>gpus</TT> property. The <TT>gpus</TT> property
	 * specifies the number of GPU accelerators a task requires. The default is
	 * none (0).
	 *
	 * @return  Number of GPUs (&ge; 0), or {@link #ALL_GPUS}.
	 *
	 * @see  #gpus(int)
	 */
	public int gpus()
		{
		return properties.gpus();
		}

	/**
	 * Set this job's <TT>workers</TT> property. The <TT>workers</TT> property
	 * specifies the number of worker tasks that will execute a master-worker
	 * cluster parallel for loop.
	 *
	 * @param  workers  Number of worker tasks (&ge; 1), or {@link
	 *                  #DEFAULT_WORKERS}.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>workers</TT> is illegal.
	 *
	 * @see  #workers()
	 */
	public void workers
		(int workers)
		{
		properties.workers (workers);
		}

	/**
	 * Get this job's <TT>workers</TT> property. The <TT>workers</TT> property
	 * specifies the number of worker tasks that will execute a master-worker
	 * cluster parallel for loop. The default is one worker task.
	 *
	 * @return  Number of workers (&ge; 1).
	 *
	 * @see  #workers(int)
	 */
	public int workers()
		{
		int rv = properties.workers();
		return rv == DEFAULT_THREADS ? 1 : rv;
		}

	/**
	 * Set this job's <TT>masterSchedule</TT> property. The
	 * <TT>masterSchedule</TT> property, along with the <TT>masterChunk</TT>
	 * property, specifies how the iterations of a master-worker cluster
	 * parallel for loop are partitioned among the worker tasks executing the
	 * parallel for loop. Refer to enum {@linkplain Schedule} for descriptions
	 * of the possible schedules.
	 *
	 * @param  schedule  Parallel for loop schedule, or {@link
	 *                   #DEFAULT_SCHEDULE}.
	 *
	 * @see  #masterSchedule()
	 * @see  #masterChunk(int)
	 * @see  #masterChunk()
	 */
	public void masterSchedule
		(Schedule schedule)
		{
		properties.schedule (schedule);
		}

	/**
	 * Get this job's <TT>masterSchedule</TT> property. The
	 * <TT>masterSchedule</TT> property, along with the <TT>masterChunk</TT>
	 * property, specifies how the iterations of a master-worker cluster
	 * parallel for loop are partitioned among the worker tasks executing the
	 * parallel for loop. The default is a fixed schedule.
	 *
	 * @return  Parallel for loop schedule.
	 *
	 * @see  #masterSchedule(Schedule)
	 * @see  #masterChunk()
	 */
	public Schedule masterSchedule()
		{
		return properties.schedule();
		}

	/**
	 * Set this job's <TT>masterChunk</TT> property. The <TT>masterChunk</TT>
	 * property, along with the <TT>masterSchedule</TT> property, specifies how
	 * the iterations of a master-worker cluster parallel for loop are
	 * partitioned among the worker tasks executing the parallel for loop. Refer
	 * to enum {@linkplain Schedule} for descriptions of the possible schedules.
	 *
	 * @param  chunk  Chunk size (&ge; 1), {@link #STANDARD_CHUNK}, or {@link
	 *                #DEFAULT_CHUNK}.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>chunk</TT> is illegal.
	 *
	 * @see  #masterSchedule(Schedule)
	 * @see  #masterSchedule()
	 * @see  #masterChunk()
	 */
	public void masterChunk
		(int chunk)
		{
		properties.chunk (chunk);
		}

	/**
	 * Get this job's <TT>masterChunk</TT> property. The <TT>masterChunk</TT>
	 * property, along with the <TT>masterSchedule</TT> property, specifies how
	 * the iterations of a master-worker cluster parallel for loop are
	 * partitioned among the worker tasks executing the parallel for loop.
	 * The default is the standard chunk size for the <TT>schedule</TT>
	 * property; see enum {@linkplain Schedule Schedule} for further
	 * information.
	 *
	 * @return  Chunk size (&ge; 1), or {@link #STANDARD_CHUNK}.
	 *
	 * @see  #masterChunk(int)
	 * @see  #masterSchedule()
	 */
	public int masterChunk()
		{
		return properties.chunk();
		}

	/**
	 * Set this job's <TT>trackerHost</TT> property. The <TT>trackerHost</TT>
	 * property specifies the host name at which the Tracker is listening for
	 * connections.
	 *
	 * @param  host  Host name.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>host</TT> is null.
	 *
	 * @see  #trackerPort(int)
	 */
	public void trackerHost
		(String host)
		{
		if (host == null)
			throw new NullPointerException
				("Job.trackerHost(): host is null");
		this.trackerHost = host;
		}

	/**
	 * Get this job's <TT>trackerHost</TT> property. The <TT>trackerHost</TT>
	 * property specifies the host name at which the Tracker is listening for
	 * connections. If set to <TT>"none"</TT>, the job will not connect to the
	 * Tracker. If not set, the job will try to connect to the Tracker on each
	 * of the local machine's network interfaces, stopping as soon as a
	 * connection is successfully established.
	 *
	 * @return  Host name.
	 *
	 * @see  #trackerPort()
	 */
	public String trackerHost()
		{
		return this.trackerHost;
		}

	/**
	 * Set this job's <TT>trackerPort</TT> property. The <TT>trackerPort</TT>
	 * property specifies the port number at which the Tracker is listening for
	 * connections.
	 *
	 * @param  port  Port number (0 .. 65535).
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>port</TT> is illegal.
	 *
	 * @see  #trackerHost(String)
	 */
	public void trackerPort
		(int port)
		{
		if (0 > port || port > 65535)
			throw new IllegalArgumentException (String.format
				("Job.trackerPort(): port = %d illegal", port));
		this.trackerPort = port;
		}

	/**
	 * Get this job's <TT>trackerPort</TT> property. The <TT>trackerPort</TT>
	 * property specifies the port number at which the Tracker is listening for
	 * connections. If not set, the default is 20618.
	 *
	 * @return  Port number (0 .. 65535).
	 *
	 * @see  #trackerHost()
	 */
	public int trackerPort()
		{
		return this.trackerPort;
		}

	/**
	 * Set this job's <TT>listenHost</TT> property. The <TT>listenHost</TT>
	 * property specifies the host name at which the job itself will listen for
	 * connections.
	 *
	 * @param  host  Host name, or null to use the default.
	 *
	 * @see  #listenPort(int)
	 */
	public void listenHost
		(String host)
		{
		this.listenHost = host;
		}

	/**
	 * Get this job's <TT>listenHost</TT> property. The <TT>listenHost</TT>
	 * property specifies the host name at which the job itself will listen for
	 * connections. If not set, the default is to use the same IP address as the
	 * job's connection to the Tracker.
	 *
	 * @return  Host name, or null if not set.
	 *
	 * @see  #listenPort()
	 */
	public String listenHost()
		{
		return this.listenHost;
		}

	/**
	 * Set this job's <TT>listenPort</TT> property. The <TT>listenPort</TT>
	 * property specifies the port number at which the job itself will listen
	 * for connections.
	 *
	 * @param  port  Port number (0 .. 65535).
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>port</TT> is illegal.
	 *
	 * @see  #listenHost(String)
	 */
	public void listenPort
		(int port)
		{
		if (0 > port || port > 65535)
			throw new IllegalArgumentException (String.format
				("Job.listenPort(): port = %d illegal", port));
		this.listenPort = port;
		}

	/**
	 * Get this job's <TT>listenPort</TT> property. The <TT>listenPort</TT>
	 * property specifies the port number at which the job itself will listen
	 * for connections. If not set, the default is 0, meaning the job will
	 * listen to an unused port.
	 *
	 * @return  Port number (0 .. 65535).
	 *
	 * @see  #listenHost()
	 */
	public int listenPort()
		{
		return this.listenPort;
		}

	/**
	 * Print the given debugging messages for this job.
	 *
	 * @param  debug  Debugging message(s) to print, or null to print none.
	 */
	public void debug
		(Debug... debug)
		{
		debugs.clear();
		if (debug != null)
			for (Debug d : debug)
				debugs.add (d);
		}

	/**
	 * Print the given debugging messages for this job.
	 *
	 * @param  debug  Set of debugging message(s) to print, or null to print
	 *                none.
	 */
	public void debug
		(Set<Debug> debug)
		{
		debugs.clear();
		if (debug != null)
			debugs.addAll (debug);
		}

	/**
	 * Determine if the given debugging message is activated for this job.
	 *
	 * @param  debug  Debugging message.
	 *
	 * @return  True if <TT>debug</TT> is activated, false if deactivated.
	 */
	public boolean debug
		(Debug debug)
		{
		return debugs.contains (debug);
		}

	/**
	 * Set this job's <TT>jvmFlags</TT> property. The JVM flags are included on
	 * the command line of the JVM that runs each task in this job. (The JVM
	 * flags are <I>not</I> included on the command line of the JVM that runs
	 * the job process itself.)
	 *
	 * @param  jvmFlags  JVM flags (zero or more).
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>jvmFlags</TT> is null or any
	 *     element of <TT>jvmFlags</TT> is null.
	 *
	 * @see  #jvmFlags()
	 */
	public void jvmFlags
		(String... jvmFlags)
		{
		if (jvmFlags == null)
			throw new NullPointerException
				("Job.jvmFlags(): jvmFlags is null");
		for (int i = 0; i < jvmFlags.length; ++ i)
			if (jvmFlags[i] == null)
				throw new NullPointerException (String.format
					("Job.jvmFlags(): jvmFlags[%d] is null", i));
		this.jvmFlags = jvmFlags;
		}

	/**
	 * Get this job's <TT>jvmFlags</TT> property. If not specified, the default
	 * is no JVM flags.
	 *
	 * @return  Array of JVM flags (zero or more).
	 *
	 * @see  #jvmFlags(String[])
	 */
	public String[] jvmFlags()
		{
		return this.jvmFlags;
		}

	/**
	 * Set the JAR file that contains the class files needed by this job and its
	 * tasks.
	 * <P>
	 * <I>Note:</I> The given JAR file <I>replaces</I> any previously specified
	 * JAR file; JAR files do not accumulate.
	 *
	 * @param  jarfile  JAR file.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>jarfile</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred while reading the JAR file.
	 */
	public void jar
		(File jarfile)
		throws IOException
		{
		if (jarfile == null)
			throw new NullPointerException
				("Job.jar(): jarfile is null");
		BufferedInputStream in = new BufferedInputStream
			(new FileInputStream (jarfile));
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		int b;
		while ((b = in.read()) != -1)
			out.write (b);
		in.close();
		this.jarfile = jarfile;
		this.jar = out.toByteArray();
		}

	/**
	 * Get the JAR file that contains the class files needed by this job and its
	 * tasks.
	 *
	 * @return  JAR file, or null if none.
	 */
	public File jar()
		{
		return this.jarfile;
		}

	/**
	 * Set the time limit for this job. If the job has not finished within the
	 * given number of seconds after the first task starts, the job will
	 * automatically terminate itself.
	 *
	 * @param  timeLimit  Time limit (seconds), or &le; 0 for no time limit.
	 */
	public void timeLimit
		(int timeLimit)
		{
		this.timeLimit = Math.max (timeLimit, 0);
		}

	/**
	 * Get the time limit for this job. If the time limit is not specified, the
	 * default is no time limit.
	 *
	 * @return  Time limit (seconds), or 0 if no time limit.
	 */
	public int timeLimit()
		{
		return this.timeLimit;
		}

// Hidden operations.

	/**
	 * Set up a connection to the Tracker.
	 *
	 * @return  Socket connected to the Tracker.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred; i.e., could not connect to the
	 *     Tracker.
	 */
	private Socket connectToTracker()
		throws IOException
		{
		// If tracker port was not specified, use default.
		if (trackerPort == -1) trackerPort = 20618;

		// If tracker host was not specified, try to connect to the Tracker on
		// every network interface in the machine.
		if (trackerHost == null)
			{
			// Check all network interfaces.
			try
				{
				Enumeration<NetworkInterface> netIntfs =
					NetworkInterface.getNetworkInterfaces();
				while (netIntfs.hasMoreElements())
					{
					NetworkInterface netIntf = netIntfs.nextElement();

					// Check all interface addresses on the current network
					// interface.
					List<InterfaceAddress> intfAddrs =
						netIntf.getInterfaceAddresses();
					for (InterfaceAddress intfAddr : intfAddrs)
						{
						// Get IP address. Omit IP addresses 224.0.0.0 and
						// higher.
						InetAddress addr = intfAddr.getAddress();
						byte[] ipaddr = addr.getAddress();
						int ipaddr_0 = ipaddr[0] & 255;
						int ipaddr_1 = ipaddr[1] & 255;
						int ipaddr_2 = ipaddr[2] & 255;
						int ipaddr_3 = ipaddr[3] & 255;
						if (ipaddr_0 >= 224) continue;
						trackerHost = String.format ("%d.%d.%d.%d",
							ipaddr_0, ipaddr_1, ipaddr_2, ipaddr_3);
//System.out.printf ("Try Tracker at %s:%d...%n", trackerHost, trackerPort);
//System.out.flush();

						// Try to set up connection. If it works, return socket,
						// otherwise go on to the next interface address.
						try
							{
							return new Socket
								(InetAddress.getByName (trackerHost),
								 trackerPort);
							}
						catch (IOException exc)
							{
							// Socket() failed
							}
						}
					}
				}
			catch (SocketException exc)
				{
				// getNetworkInterfaces() failed
				}

			// If the above was not successful, use default tracker host.
			trackerHost = "localhost";
			}

		// If tracker host was specified or the above was not successful, try to
		// connect to the Tracker at the tracker host.
		return new Socket (InetAddress.getByName (trackerHost), trackerPort);
		}

	/**
	 * Tell this job that the job launched.
	 *
	 * @param  jobID  Job ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private synchronized void jobLaunched
		(long jobID)
		throws IOException
		{
		this.jobID = jobID;
		this.taskMap = new TaskMap();

		if (debugs.contains (Debug.jobLaunch))
			{
			System.err.printf ("Job %d launched %s%n",
				jobID, jobET.startDate());
			System.err.flush();
			}
		}

	/**
	 * Tell this job that the job started.
	 *
	 * @param  jobID  Job ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private synchronized void jobStarted()
		throws IOException
		{
		// Fire rules.
		fireStartRules();
		fireOnDemandRules();
		fireFinishRules();
		}

	/**
	 * Fire all start rules unconditionally.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private synchronized void fireStartRules()
		throws IOException
		{
		AList<Tuple> noInputTuples = new AList<Tuple>();
		int nRules = startRules.size();
		for (int i = 0; i < nRules; ++ i)
			{
			Rule rule = startRules.get (i);
			fireRule (rule, noInputTuples);
			}
		startRules.clear();
		}

	/**
	 * Fire all on-demand rules that match tuples in tuple space.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private synchronized void fireOnDemandRules()
		throws IOException
		{
		int nRules = onDemandRules.size();
		for (int i = 0; i < nRules; ++ i)
			{
			Rule rule = onDemandRules.get (i);
			matchLoop : for (;;)
				{
				TupleSpace.Transaction trans = tupleSpace.getTransaction();
				rule.match (trans);
				AList<Tuple> inputTuples = trans.getTuples();
				if (inputTuples.isEmpty())
					break matchLoop;
				else
					fireRule (rule, inputTuples);
				}
			}
		}

	/**
	 * Fire all finish rules, if all previous tasks have finished.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private synchronized void fireFinishRules()
		throws IOException
		{
		// If there are no tasks, fire finish rules.
		if (taskMap.isEmpty())
			{
			jobIsFinished = true;
			AList<Tuple> noInputTuples = new AList<Tuple>();
			int nRules = finishRules.size();
			for (int i = 0; i < nRules; ++ i)
				{
				Rule rule = finishRules.get (i);
				fireRule (rule, noInputTuples);
				}
			onDemandRules.clear();
			finishRules.clear();

			// If there are still no tasks, the job is finished.
			if (taskMap.isEmpty())
				{
				jobET.stop();

				if (debugs.contains (Debug.jobFinish))
					{
					System.err.printf ("Job %d finished %s time %s%n",
						jobID, jobET.stopDate(), jobET);
					System.err.flush();
					}
				if (debugs.contains (Debug.makespan))
					{
					System.err.printf ("Job %d makespan %d msec%n",
						jobID, maxTaskFinishTime - minTaskStartTime);
					System.err.flush();
					}
				if (debugs.contains (Debug.remainingTuples))
					{
					AList<Tuple> allTuples = tupleSpace.allTuples();
					if (! allTuples.isEmpty())
						{
						System.err.printf ("   Remaining tuples:%n");
						allTuples.forEachItemDo (new PrintTuple());
						}
					System.err.flush();
					}

				// Wake up the PJ2 Launcher.
				stopJob (null);
				}
			}
		}

	/**
	 * Fire the given rule.
	 *
	 * @param  rule         Rule.
	 * @param  inputTuples  List of input tuples.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private synchronized void fireRule
		(final Rule rule,
		 final AList<Tuple> inputTuples)
		throws IOException
		{
		final AList<TaskSpecInfo> taskSpecList = new AList<TaskSpecInfo>();

		// Set up each task in the rule's task group.
		rule.taskGroup.forEachItemDo (new Action<TaskSpec>()
			{
			int size = rule.taskGroup.size();
			int rank = 0;
			public void run (TaskSpec taskSpec)
				{
				// Set up info for this job.
				TaskInfo info = taskMap.add();
				info.taskClassName = taskSpec.taskClassName;
				info.args = taskSpec.args;
				info.inputTuples = inputTuples;
				info.properties = new TaskProperties (taskSpec.properties);
				info.jar = jar;
				info.size = size;
				info.rank = rank ++;
				info.devnum = null;
				ExtraData extra = new ExtraData();
				extra.ET = new ElapsedTime();
				extra.debugs = taskSpec.debugs;
				info.moreData = extra;

				// Print debugging messages.
				if (taskDebug (Debug.taskLaunch, info))
					{
					System.err.printf ("Job %d task %d launched %s%n",
						jobID, info.taskID, extra.ET.startDate());
					System.err.flush();
					}
				if (taskDebug (Debug.taskArguments, info))
					{
					System.err.printf ("   Command: %s",
						info.taskClassName);
					for (int i = 0; i < info.args.length; ++ i)
						System.err.printf (" %s", info.args[i]);
					System.err.println();
					System.err.printf ("   %s%n", taskSpec.node());
					System.err.flush();
					}
				if (taskDebug (Debug.taskInputTuples, info) &&
						! info.inputTuples.isEmpty())
					{
					System.err.printf ("   Matching tuples that fired task:%n");
					info.inputTuples.forEachItemDo (new PrintTuple());
					System.err.flush();
					}

				// Decide where to run the task.
				if (taskSpec.runInJob || tracker == null)
					{
					// Run the task in the job's process.
					recordTaskStartTime (extra.ET);
					startJobTimeout();
					if (taskDebug (Debug.taskStart, info))
						{
						System.err.printf
							("Job %d task %d started in this process %s%n",
							jobID, info.taskID, extra.ET.startDate());
						System.err.flush();
						}
					extra.heartbeat = new BackendHeartbeat (null, info.taskID);
					// Don't enable heartbeat
					new TaskInJobThread (info) .start();
					}
				else
					{
					// Run the task via the Tracker.
					TaskSpecInfo taskSpecInfo = new TaskSpecInfo();
					taskSpecInfo.taskID = info.taskID;
					taskSpecInfo.node = taskSpec.node();
					taskSpecInfo.jvmFlags = taskSpec.jvmFlags;
					taskSpecList.addLast (taskSpecInfo);
					}
				}
			});

		// Launch tasks in the Tracker, if any.
		if (! taskSpecList.isEmpty())
			tracker.launchTaskGroup (jobID, taskSpecList);
		}

	/**
	 * Tell this job that the given task is launching.
	 *
	 * @param  taskID           Task ID.
	 * @param  devnum           Array of GPU device numbers the task is allowed
	 *                          to use (must be non-null).
	 * @param  runInJobProcess  True to run the task in the job's process, false
	 *                          otherwise.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private synchronized void taskLaunching
		(long taskID,
		 int[] devnum,
		 boolean runInJobProcess)
		{
		TaskInfo info = taskMap.get (taskID);
		if (info == null)
			throw new IllegalStateException (String.format
				("Job.taskLaunching(): Task %d nonexistent", taskID));

		info.devnum = devnum;

		ExtraData extra = (ExtraData) info.moreData;
		extra.informTracker = true;

		if (runInJobProcess)
			{
			// Run the task in the job's process.
			extra.ET.start();
			recordTaskStartTime (extra.ET);
			startJobTimeout();
			if (taskDebug (Debug.taskStart, info))
				{
				System.err.printf
					("Job %d task %d started in this process %s%n",
					jobID, info.taskID, extra.ET.startDate());
				if (info.devnum.length > 0)
					{
					System.err.printf ("   GPU device number(s):");
					for (int d : info.devnum)
						System.err.printf (" %d", d);
					System.err.println();
					}
				System.err.flush();
				}
			extra.heartbeat = new BackendHeartbeat (null, info.taskID);
			// Don't enable heartbeat
			new TaskInJobThread (info) .start();
			}
		else
			{
			// Task will run in a Backend.
//			if (extra.heartbeat == null)
//				{
//				extra.heartbeat = new TaskStartHeartbeat (taskID);
//				extra.heartbeat.enable (executor);
//				}
//					// Note: If the heartbeat object already exists, then the
//					// Backend already called taskLaunched() before the Tracker
//					// got around to calling taskLaunching(), so don't overwrite
//					// the existing heartbeat object.
			extra.heartbeat = new TaskStartHeartbeat (taskID);
			extra.heartbeat.enable (executor);
			}

		// Wake up any threads blocked in taskLaunched().
		notifyAll();
		}

	/**
	 * Tell this job that the given task was launched on the given backend.
	 *
	 * @param  backend  Backend.
	 * @param  taskID   Task ID.
	 * @param  name     Name of the node where the task is running.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private synchronized void taskLaunched
		(BackendRef backend,
		 long taskID,
		 String name)
		throws IOException
		{
		TaskInfo info = taskMap.get (taskID);
		if (info == null)
			throw new IllegalStateException (String.format
				("Job.taskLaunched(): Task %d nonexistent", taskID));

		// Wait until the Tracker has called taskLaunching() for this task ID
		// and set the allowed GPU device numbers.
		while (info.devnum == null)
			{
			try { wait(); } catch (InterruptedException exc) {}
			}

		ExtraData extra = (ExtraData) info.moreData;
//		if (extra.heartbeat != null)
//			extra.heartbeat.cancel();
//				// Note: If the heartbeat object does not exist, then the
//				// Backend called taskLaunched() before the Tracker got around
//				// to calling taskLaunching(), so don't cancel the nonexistent
//				// heartbeat object.
		extra.heartbeat.cancel();
		extra.heartbeat = new BackendHeartbeat (backend, taskID);
		extra.heartbeat.enable (executor);
		extra.backend = backend;
		extra.informTracker = true;

		extra.ET.start();
		recordTaskStartTime (extra.ET);
		startJobTimeout();
		if (taskDebug (Debug.taskStart, info))
			{
			System.err.printf ("Job %d task %d started on %s %s%n",
				jobID, info.taskID, name, extra.ET.startDate());
			if (info.devnum.length > 0)
				{
				System.err.printf ("   GPU device number(s):");
				for (int d : info.devnum)
					System.err.printf (" %d", d);
				System.err.println();
				}
			System.err.flush();
			}

		backend.startTask (info);
		}

	/**
	 * Tell this job to take a tuple that matches the given template out of
	 * tuple space.
	 *
	 * @param  taskID     Task ID.
	 * @param  requestID  Request ID.
	 * @param  template   Template.
	 * @param  blocking   True for a blocking take or read, false for a
	 *                    non-blocking take or read.
	 * @param  taking     True to take a tuple, false to read a tuple.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private synchronized void takeTuple
		(long taskID,
		 long requestID,
		 Tuple template,
		 boolean blocking,
		 boolean taking)
		throws IOException
		{
		TaskInfo info = taskMap.get (taskID);
		if (info == null)
			throw new IllegalStateException (String.format
				("Job.takeTuple(): Task %d nonexistent", taskID));

		if (taskDebug (Debug.taskInputTuples, info))
			{
			System.err.printf
				("Job %d task %d requests to %s%s tuple matching template%n",
				 jobID, taskID,
				 taking ? "take" : "read",
				 blocking ? "" : " (non-blocking)");
			System.err.printf ("   ");
			template.dump (System.err, 2);
			System.err.flush();
			}

		TupleTakeInfo tupleTakeInfo = new TupleTakeInfo();
		tupleTakeInfo.taskID = taskID;
		tupleTakeInfo.requestID = requestID;
		tupleTakeInfo.template = template;
		tupleTakeInfo.blocking = blocking;
		tupleTakeInfo.taking = taking;
		tupleTakeList.addLast (tupleTakeInfo);
		takeTuples();
		}

	/**
	 * Tell this job to write the given tuple into tuple space.
	 *
	 * @param  taskID  Task ID.
	 * @param  tuple   Tuple.
	 * @param  copies  Number of copies to write (1 or more).
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private synchronized void writeTuple
		(long taskID,
		 Tuple tuple,
		 int copies)
		throws IOException
		{
		TaskInfo info = taskMap.get (taskID);
		if (info == null)
			throw new IllegalStateException (String.format
				("Job.writeTuple(): Task %d nonexistent", taskID));

		if (! jobIsFinished)
			{
			if (taskDebug (Debug.taskOutputTuples, info))
				{
				System.err.printf ("Job %d task %d writes tuple%s%n",
					jobID, taskID, copies > 1 ? "("+copies+" copies)" : "");
				System.err.printf ("   ");
				tuple.dump (System.err, 2);
				System.err.flush();
				}

			for (int i = 0; i < copies; ++ i)
				tupleSpace.putTuple (tuple);
			takeTuples();
			fireOnDemandRules();
			fireFinishRules();
			}
		}

	/**
	 * Match any outstanding tuple take request templates with tuples in tuple
	 * space.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private synchronized void takeTuples()
		throws IOException
		{
		DListEntry<TupleTakeInfo> p, q;

		// Examine all tuple take requests.
		p = tupleTakeList.first();
		requestLoop : while (p != null)
			{
			q = p.succ();
			TupleTakeInfo tupleTakeInfo = p.item();

			// Get task info for current tuple take request.
			TaskInfo taskInfo = taskMap.get (tupleTakeInfo.taskID);
			if (taskInfo == null)
				throw new IllegalStateException (String.format
					("Job.takeTuples(): Task %d nonexistent",
					 tupleTakeInfo.taskID));

			// Set up a transaction for searching tuple space.
			TupleSpace.Transaction trans = tupleSpace.getTransaction();
			Iterator<TupleSpace.TupleRef> iter = trans.iterator();
			boolean match = false;
			TupleSpace.TupleRef ref = null;
			Tuple target = null;
			boolean reportTarget = false;

			// Search for a matching tuple.
			while (! match && iter.hasNext())
				{
				ref = iter.next();
				target = ref.tuple;
				if (tupleTakeInfo.template.match (target))
					match = true;
				else
					target = null;
				}

			// A matching tuple was not found.
			if (! match)
				{
				// Abort transaction.
				trans.abort();

				// If the tuple take request was non-blocking, report that there
				// was no match.
				if (! tupleTakeInfo.blocking)
					{
					reportTarget = true;
					if (taskDebug (Debug.taskInputTuples, taskInfo))
						{
						System.err.printf
							("Job %d task %d finds no matching tuple%n",
							 jobID, tupleTakeInfo.taskID);
						System.err.flush();
						}
					}
				}

			// A matching tuple was found for a take.
			else if (tupleTakeInfo.taking)
				{
				// Remove matching tuple from tuple space.
				trans.remove (ref);
				trans.commit();
				reportTarget = true;

				// Debug printout of matching tuple.
				if (taskDebug (Debug.taskInputTuples, taskInfo))
					{
					System.err.printf
						("Job %d task %d takes tuple%n",
						 jobID, tupleTakeInfo.taskID);
					System.err.printf ("   ");
					target.dump (System.err, 2);
					System.err.flush();
					}
				}

			// A matching tuple was found for a read.
			else
				{
				// Leave matching tuple in tuple space.
				trans.abort();
				reportTarget = true;

				// Debug printout of matching tuple.
				if (taskDebug (Debug.taskInputTuples, taskInfo))
					{
					System.err.printf
						("Job %d task %d reads tuple%n",
						 jobID, tupleTakeInfo.taskID);
					System.err.printf ("   ");
					target.dump (System.err, 2);
					System.err.flush();
					}
				}

			// Send result, if any, to taking task.
			if (reportTarget)
				{
				ExtraData extra = (ExtraData) taskInfo.moreData;
				if (extra.backend != null)
					extra.backend.tupleTaken
						(tupleTakeInfo.taskID,
						 tupleTakeInfo.requestID,
						 target);
				else
					tupleRequestMap.tupleTaken
						(tupleTakeInfo.taskID,
						 tupleTakeInfo.requestID,
						 target);

				// Remove tuple take request.
				p.remove();
				}

			// Go to next tuple take request.
			p = q;
			}
		}

	/**
	 * Tell this job that the given task finished.
	 *
	 * @param  taskID        Task ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private synchronized void taskFinished
		(long taskID)
		throws IOException
		{
		flushConsoleStreams();

		TaskInfo info = taskMap.remove (taskID);
		if (info == null)
			throw new IllegalStateException (String.format
				("Job.taskFinished(): Task %d nonexistent", taskID));

		ExtraData extra = (ExtraData) info.moreData;
		extra.ET.stop();
		extra.heartbeat.cancel();
		if (extra.backend != null)
			extra.backend.terminate();
		if (extra.informTracker)
			tracker.taskDone (jobID, taskID);
		removeTupleTakeRequestsFor (taskID);

		recordTaskFinishTime (extra.ET);
		if (taskDebug (Debug.taskFinish, info))
			{
			System.err.printf ("Job %d task %d finished %s time %s%n",
				jobID, info.taskID, extra.ET.stopDate(), extra.ET);
			System.err.flush();
			}

		fireFinishRules();
		}

	/**
	 * Tell this job that the given task failed.
	 *
	 * @param  taskID   Task ID.
	 * @param  exc      Exception that was thrown.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private synchronized void taskFailed
		(long taskID,
		 Throwable exc)
		throws IOException
		{
		flushConsoleStreams();

		TaskInfo info = taskMap.remove (taskID);
		if (info == null)
			throw new IllegalStateException (String.format
				("Job.taskFailed(): Task %d nonexistent", taskID), exc);

		ExtraData extra = (ExtraData) info.moreData;
		extra.ET.stop();
		recordTaskFinishTime (extra.ET);
		if (extra.heartbeat != null)
			extra.heartbeat.cancel();
		if (extra.backend != null)
			extra.backend.terminate();
		removeTupleTakeRequestsFor (taskID);

		//System.err.printf ("Job %d task %d failed%n", jobID, taskID);
		//System.err.flush();
		stopJob (exc);
		}

	/**
	 * Remove all tuple take requests for the given task ID.
	 *
	 * @param  taskID  Task ID.
	 */
	private synchronized void removeTupleTakeRequestsFor
		(final long taskID)
		{
		tupleTakeList.removeEachItemIf (new Predicate<TupleTakeInfo>()
			{
			public boolean test (TupleTakeInfo info)
				{
				return info.taskID == taskID;
				}
			});
		}

	/**
	 * Tell this job that the tracker is still alive.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private synchronized void heartbeatFromTracker()
		throws IOException
		{
		trackerHeartbeat.receiveHeartbeat();
		}

	/**
	 * Tell this job that the given task is still alive.
	 *
	 * @param  taskID  Task ID.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private synchronized void heartbeatFromTask
		(long taskID)
		throws IOException
		{
		TaskInfo info = taskMap.get (taskID);
		if (info == null)
			throw new IllegalStateException (String.format
				("Job.heartbeatFromTask(): Task %d nonexistent", taskID));

		ExtraData extra = (ExtraData) info.moreData;
		extra.heartbeat.receiveHeartbeat();
		}

	/**
	 * Write the given data to the standard output or standard error stream.
	 *
	 * @param  stream  0 to write to standard output, 1 to write to standard
	 *                 error.
	 * @param  len     Length of data.
	 * @param  data    Data buffer.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private synchronized void writeStandardStream
		(int stream,
		 int len,
		 byte[] data)
		throws IOException
		{
		switch (stream)
			{
			case 0:
				System.out.write (data, 0, len);
				System.out.flush();
				break;
			case 1:
				System.err.write (data, 0, len);
				System.err.flush();
				break;
			default:
				throw new IllegalArgumentException (String.format
					("Job.writeStandardStream(): stream = %d illegal",
					 stream));
			}
		}

	/**
	 * Tell this job that the whole job failed.
	 *
	 * @param  exc  Exception that was thrown.
	 */
	private synchronized void jobFailed
		(Throwable exc)
		{
		System.err.printf ("Job %d failed%n", jobID);
		System.err.flush();
		stopJob (exc);
		}

	/**
	 * Signal the PJ2 Launcher program to stop the job.
	 *
	 * @param  exc  Exception that was thrown, or null if none.
	 */
	private synchronized void stopJob
		(Throwable exc)
		{
		excThrown = exc;
		finishSema.release();
		}

	/**
	 * Determine if a certain debugging message is to be printed for the given
	 * task.
	 *
	 * @param  debug  Debugging message.
	 * @param  info   Task information object.
	 *
	 * @return  True to print the debugging message, false otherwise.
	 */
	private synchronized boolean taskDebug
		(Debug debug,
		 TaskInfo info)
		{
		ExtraData extra = (ExtraData) info.moreData;
		return extra.debugs == null ?
			debugs.contains (debug) : extra.debugs.contains (debug);
		}

	/**
	 * Start the job timeout thread.
	 */
	private synchronized void startJobTimeout()
		{
		if (timeLimit > 0 && timeoutThread == null)
			new TimeoutThread().start();
		}

	/**
	 * Record the start time of a task, for calculating the job's makespan.
	 *
	 * @param  ET  Task elapsed time object.
	 */
	private synchronized void recordTaskStartTime
		(ElapsedTime ET)
		{
		if (minTaskStartTime == Long.MAX_VALUE &&
			debugs.contains (Debug.jobStart))
				{
				System.err.printf ("Job %d started %s%n",
					jobID, ET.startDate());
				System.err.flush();
				}
		minTaskStartTime = Math.min (minTaskStartTime, ET.startTime());
		}

	/**
	 * Record the finish time of a task, for calculating the job's makespan.
	 *
	 * @param  ET  Task elapsed time object.
	 */
	private synchronized void recordTaskFinishTime
		(ElapsedTime ET)
		{
		maxTaskFinishTime = Math.max (maxTaskFinishTime, ET.stopTime());
		}

	/**
	 * Partition the given integer loop index range into chunks using a fixed
	 * schedule, and write the chunks into this job's tuple space.
	 *
	 * @param  workers      Number of worker tasks.
	 * @param  masterChunk  <TT>masterChunk</TT> parameter.
	 * @param  lb           Loop index lower bound.
	 * @param  ub           Loop index upper bound.
	 */
	void putChunksFixed
		(int workers,
		 int masterChunk,
		 int lb,
		 int ub)
		{
		long chunklb = lb;
		long chunkub = ub;
		long chunkSize = Math.max ((chunkub - chunklb + workers)/workers, 1L);
		long newlb;
		for (int r = 0; r < workers; ++ r)
			{
			newlb = chunklb + chunkSize;
			putTuple (new Chunk() .rank (r) .lb ((int)chunklb)
				.ub ((int)(Math.min (newlb - 1L, chunkub))));
			chunklb = newlb;
			}
		}

	/**
	 * Partition the given integer loop index range into chunks using a leapfrog
	 * schedule, and write the chunks into this job's tuple space.
	 *
	 * @param  workers      Number of worker tasks.
	 * @param  masterChunk  <TT>masterChunk</TT> parameter.
	 * @param  lb           Loop index lower bound.
	 * @param  ub           Loop index upper bound.
	 */
	void putChunksLeapfrog
		(int workers,
		 int masterChunk,
		 int lb,
		 int ub)
		{
		for (int r = 0; r < workers; ++ r)
			putTuple (new Chunk() .rank (r) .lb (lb + r) .ub (ub)
				.stride (workers));
		}

	/**
	 * Partition the given integer loop index range into chunks using a dynamic
	 * schedule, and write the chunks into this job's tuple space.
	 *
	 * @param  workers      Number of worker tasks.
	 * @param  masterChunk  <TT>masterChunk</TT> parameter.
	 * @param  lb           Loop index lower bound.
	 * @param  ub           Loop index upper bound.
	 */
	void putChunksDynamic
		(int workers,
		 int masterChunk,
		 int lb,
		 int ub)
		{
		long chunklb = lb;
		long chunkub = ub;
		long chunkSize = masterChunk;
		long newlb;
		while (chunklb <= chunkub)
			{
			newlb = chunklb + chunkSize;
			putTuple (new Chunk() .rank (Chunk.ANY) .lb ((int)chunklb)
				.ub ((int)(Math.min (newlb - 1L, chunkub))));
			chunklb = newlb;
			}
		}

	/**
	 * Partition the given integer loop index range into chunks using a
	 * proportional schedule, and write the chunks into this job's tuple space.
	 *
	 * @param  workers      Number of worker tasks.
	 * @param  masterChunk  <TT>masterChunk</TT> parameter.
	 * @param  lb           Loop index lower bound.
	 * @param  ub           Loop index upper bound.
	 */
	void putChunksProportional
		(int workers,
		 int masterChunk,
		 int lb,
		 int ub)
		{
		long chunklb = lb;
		long chunkub = ub;
		long n = masterChunk;
		n *= workers;
		long chunkSize = Math.max ((chunkub - chunklb + n)/n, 1L);
		long newlb;
		while (chunklb <= chunkub)
			{
			newlb = chunklb + chunkSize;
			putTuple (new Chunk() .rank (Chunk.ANY) .lb ((int)chunklb)
				.ub ((int)(Math.min (newlb - 1L, chunkub))));
			chunklb = newlb;
			}
		}

	/**
	 * Partition the given integer loop index range into chunks using a guided
	 * schedule, and write the chunks into this job's tuple space.
	 *
	 * @param  workers      Number of worker tasks.
	 * @param  masterChunk  <TT>masterChunk</TT> parameter.
	 * @param  lb           Loop index lower bound.
	 * @param  ub           Loop index upper bound.
	 */
	void putChunksGuided
		(int workers,
		 int masterChunk,
		 int lb,
		 int ub)
		{
		long chunklb = lb;
		long chunkub = ub;
		long minChunkSize = masterChunk;
		long chunkSize, newlb;
		while (chunklb <= chunkub)
			{
			chunkSize = Math.max ((chunkub - chunklb + 1L)/2/workers,
				minChunkSize);
			newlb = chunklb + chunkSize;
			putTuple (new Chunk() .rank (Chunk.ANY) .lb ((int)chunklb)
				.ub ((int)(Math.min (newlb - 1L, chunkub))));
			chunklb = newlb;
			}
		}

	/**
	 * Partition the given long integer loop index range into chunks using a
	 * fixed schedule, and write the chunks into this job's tuple space.
	 *
	 * @param  workers      Number of worker tasks.
	 * @param  masterChunk  <TT>masterChunk</TT> parameter.
	 * @param  lb           Loop index lower bound.
	 * @param  ub           Loop index upper bound.
	 */
	void putChunksFixed
		(int workers,
		 int masterChunk,
		 long lb,
		 long ub)
		{
		Int96 n = Int96.of(workers);
		Int96 chunklb = Int96.of(lb);
		Int96 chunkub = Int96.of(ub);
		Int96 chunkSize = Int96.of(chunkub) .sub(chunklb) .add(n) .div(n)
			.max(Int96.ONE);
		Int96 newlb = Int96.of(0);
		for (int r = 0; r < workers; ++ r)
			{
			newlb .assign(chunklb) .add(chunkSize);
			putTuple (new LongChunk() .rank (r)
				.lb (chunklb .longval())
				.ub (Int96.of(newlb) .sub(Int96.ONE) .min(chunkub) .longval()));
			chunklb .assign(newlb);
			}
		}

	/**
	 * Partition the given long integer loop index range into chunks using a
	 * leapfrog schedule, and write the chunks into this job's tuple space.
	 *
	 * @param  workers      Number of worker tasks.
	 * @param  masterChunk  <TT>masterChunk</TT> parameter.
	 * @param  lb           Loop index lower bound.
	 * @param  ub           Loop index upper bound.
	 */
	void putChunksLeapfrog
		(int workers,
		 int masterChunk,
		 long lb,
		 long ub)
		{
		for (int r = 0; r < workers; ++ r)
			putTuple (new LongChunk() .rank (r) .lb (lb + r) .ub (ub)
				.stride (workers));
		}

	/**
	 * Partition the given long integer loop index range into chunks using a
	 * dynamic schedule, and write the chunks into this job's tuple space.
	 *
	 * @param  workers      Number of worker tasks.
	 * @param  masterChunk  <TT>masterChunk</TT> parameter.
	 * @param  lb           Loop index lower bound.
	 * @param  ub           Loop index upper bound.
	 */
	void putChunksDynamic
		(int workers,
		 int masterChunk,
		 long lb,
		 long ub)
		{
		Int96 chunklb = Int96.of(lb);
		Int96 chunkub = Int96.of(ub);
		Int96 chunkSize = Int96.of(masterChunk);
		Int96 newlb = Int96.of(0);
		while (chunklb.compareTo (chunkub) <= 0)
			{
			newlb .assign(chunklb) .add(chunkSize);
			putTuple (new LongChunk() .rank (LongChunk.ANY)
				.lb (chunklb .longval())
				.ub (Int96.of(newlb) .sub(Int96.ONE) .min(chunkub) .longval()));
			chunklb .assign(newlb);
			}
		}

	/**
	 * Partition the given long integer loop index range into chunks using a
	 * proportional schedule, and write the chunks into this job's tuple space.
	 *
	 * @param  workers      Number of worker tasks.
	 * @param  masterChunk  <TT>masterChunk</TT> parameter.
	 * @param  lb           Loop index lower bound.
	 * @param  ub           Loop index upper bound.
	 */
	void putChunksProportional
		(int workers,
		 int masterChunk,
		 long lb,
		 long ub)
		{
		Int96 chunklb = Int96.of(lb);
		Int96 chunkub = Int96.of(ub);
		Int96 n = Int96.of(masterChunk) .mul(Int96.of(workers));
		Int96 chunkSize = Int96.of(chunkub) .sub(chunklb) .add(n) .div(n)
			.max(Int96.ONE);
		Int96 newlb = Int96.of(0);
		while (chunklb.compareTo (chunkub) <= 0)
			{
			newlb .assign(chunklb) .add(chunkSize);
			putTuple (new LongChunk() .rank (LongChunk.ANY)
				.lb (chunklb .longval())
				.ub (Int96.of(newlb) .sub(Int96.ONE) .min(chunkub) .longval()));
			chunklb .assign(newlb);
			}
		}

	/**
	 * Partition the given long integer loop index range into chunks using a
	 * guided schedule, and write the chunks into this job's tuple space.
	 *
	 * @param  workers      Number of worker tasks.
	 * @param  masterChunk  <TT>masterChunk</TT> parameter.
	 * @param  lb           Loop index lower bound.
	 * @param  ub           Loop index upper bound.
	 */
	void putChunksGuided
		(int workers,
		 int masterChunk,
		 long lb,
		 long ub)
		{
		Int96 chunklb = Int96.of(lb);
		Int96 chunkub = Int96.of(ub);
		Int96 W = Int96.of(workers);
		Int96 minChunkSize = Int96.of(masterChunk);
		Int96 chunkSize = Int96.of(0);
		Int96 newlb = Int96.of(0);
		while (chunklb.compareTo (chunkub) <= 0)
			{
			chunkSize .assign(chunkub) .sub(chunklb) .add(Int96.ONE)
				.div(Int96.TWO) .div(W) .max(minChunkSize);
			newlb .assign(chunklb) .add(chunkSize);
			putTuple (new LongChunk() .rank (LongChunk.ANY)
				.lb (chunklb .longval())
				.ub (Int96.of(newlb) .sub(Int96.ONE) .min(chunkub) .longval()));
			chunklb .assign(newlb);
			}
		}

	/**
	 * Flush the console streams.
	 */
	private void flushConsoleStreams()
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

// Hidden helper threads.

	/**
	 * Class ListenThread listens for connections from backend nodes.
	 */
	private class ListenThread
		extends Thread
		{
		public ListenThread()
			{
			setContextClassLoader
				(Thread.currentThread().getContextClassLoader());
			}

		public void run()
			{
			try
				{
				for (;;)
					{
					Socket socket = serverSocket.accept();
					new Proxy
						(socket,
						 new JobReceiver
							(new ReceiverListener()
								{
								public void receiverFailed
									(Receiver receiver,
									 Throwable exc)
									{
									System.err.printf
									 	("Failure while receiving a Backend message%n");
									exc.printStackTrace (System.err);
									System.err.flush();
									stopJob (exc);
									}
								},
							 jobRef));
					}
				}
			catch (IOException exc)
				{
				System.err.printf ("Failure while accepting a connection%n");
				System.err.flush();
				stopJob (exc);
				}
			}
		}

	/**
	 * Class TaskInJobThread runs a task in this job's process.
	 */
	private class TaskInJobThread
		extends Thread
		{
		private TaskInfo info;

		/**
		 * Construct a new thread for running the given task.
		 *
		 * @param  info  Task information record.
		 */
		public TaskInJobThread
			(TaskInfo info)
			{
			this.info = info;
			setContextClassLoader
				(Thread.currentThread().getContextClassLoader());
			}

		/**
		 * Run the task.
		 */
		public void run()
			{
			try
				{
				// Create instance of task subclass.
				Task task = (Task)
					Instance.newDefaultInstance (info.taskClassName, true);

				// Specify GPU device numbers the task is allowed to use.
				Gpu.setDeviceNumbers (info.devnum);

				// Run task.
				task.job = jobRef;
				task.taskID = info.taskID;
				task.inputTuples = info.inputTuples;
				task.tupleRequestMap = tupleRequestMap;
				task.properties = new TaskProperties (info.properties);
				task.groupSize = info.size;
				task.taskRank = info.rank;
				task.main (info.args);
				taskFinished (info.taskID);
				}
			catch (Throwable exc)
				{
				try { taskFailed (info.taskID, exc); }
					catch (IOException exc2) {}
				}
			}
		}

	/**
	 * Class TimeoutThread terminates the job after the time limit has elapsed.
	 */
	private class TimeoutThread
		extends Thread
		{
		public void run()
			{
			try
				{
				Thread.sleep (timeLimit*1000L);
				stopJob (new TerminateException ("Time limit exceeded"));
				}
			catch (InterruptedException exc)
				{
				}
			}
		}

	}
