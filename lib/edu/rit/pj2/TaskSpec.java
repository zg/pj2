//******************************************************************************
//
// File:    TaskSpec.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.TaskSpec
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

import edu.rit.pj2.tracker.JobProperties;
import edu.rit.pj2.tracker.NodeProperties;
import edu.rit.pj2.tracker.TaskProperties;
import java.lang.reflect.Method;
import java.util.EnumSet;
import java.util.Set;

/**
 * Class TaskSpec provides a task specification that is part of a {@linkplain
 * Rule Rule} in a {@linkplain Job Job}.
 * <P>
 * <B>Programming task specifications.</B>
 * To create a task specification:
 * <UL>
 * <P><LI>
 * Obtain an instance of class TaskSpec by calling the {@link Rule#task(Class)
 * task(Class)} or {@link Rule#task(int,Class) task(int,Class)} method on a
 * {@linkplain Rule}, or by calling the {@link #task(Class) task(Class)} or
 * {@link #task(int,Class) task(int,Class)} method on another TaskSpec. The
 * method's argument is the task subclass.
 * <P>
 * The following TaskSpec method calls are all optional; if not specified, a
 * default is used. For further information, see the documentation for each
 * method.
 * <P><LI>
 * Specify the task's command line arguments by calling the {@link
 * #args(String[]) args()} method one or more times.
 * <P><LI>
 * Specify the task's properties by calling the {@link #threads(int)
 * threads()}, {@link #schedule(Schedule) schedule()}, and {@link #chunk(int)
 * chunk()} methods.
 * <P><LI>
 * Specify the node capabilities required to execute the task by calling the
 * {@link #nodeName(String) nodeName()}, {@link #cores(int) cores()}, and {@link
 * #gpus(int) gpus()} methods.
 * <P><LI>
 * Specify the JVM flags for the task by calling the {@link #jvmFlags(String[])
 * jvmFlags()} method. The JVM flags, if any, are included on the command line
 * of the JVM that runs the task. For example, to set the maximum size of the
 * JVM heap to 2 gigabytes, call <TT>jvmFlags("-Xmx2000m")</TT>.
 * <P><LI>
 * Optionally, specify that the task must run in the job's own process by
 * calling the {@link #runInJobProcess() runInJobProcess()} method. When a job
 * is run on a cluster, grid, or cloud, by default tasks run in separate
 * processes on nodes other than the node where the job process is running.
 * Specifying <TT>runInJobProcess()</TT> forces the task to run in the job's own
 * process. Situations in which this might be necessary include:
 * <UL>
 * <P><LI>
 * The task needs to read the job's standard input.
 * <P><LI>
 * The task needs to display a GUI on the job's node's screen.
 * <P><LI>
 * The task needs to run in the user's account&mdash;for example, to read or
 * write the user's files&mdash;but tasks running on other nodes do not run in
 * the user's account.
 * </UL>
 * <P><LI>
 * Specify task-specific debugging printouts for the task by calling the {@link
 * #debug(Debug[]) debug()} method. The specified debug settings will be used
 * for the task, rather than the overall job's debug settings. For example, you
 * can turn off debugging for the job, but turn on debugging just for a certain
 * task.
 * </UL>
 * <P>
 * For example, the code
 * <PRE>
 *     rule() .task (MineCoinSmp.class) .args ("0123456789abcdef", "16");</PRE>
 * <P>
 * specifies a task that will execute the code in class {@linkplain
 * edu.rit.pj2.example.MineCoinSmp MineCoinSmp}, with two command line arguments
 * <TT>0123456789abcdef</TT> and <TT>16</TT>. Be default, the task can run on
 * any node, the task will use all CPU cores of the node, and the task does not
 * need any GPU accelerators.
 * <P>
 * For further information about constructing rules, see class {@linkplain Rule
 * Rule}. For further information about execution of a job, see class
 * {@linkplain Job Job}.
 *
 * @author  Alan Kaminsky
 * @version 08-Jul-2014
 */
public class TaskSpec
	{

// Exported constants.

	/**
	 * Indicates that the <TT>threads</TT> property is defaulted.
	 */
	public static final int DEFAULT_THREADS = TaskProperties.DEFAULT_THREADS;

	/**
	 * Indicates that the <TT>schedule</TT> property is defaulted.
	 */
	public static final Schedule DEFAULT_SCHEDULE =
		TaskProperties.DEFAULT_SCHEDULE;

	/**
	 * Indicates that the <TT>chunk</TT> property is defaulted.
	 */
	public static final int DEFAULT_CHUNK = TaskProperties.DEFAULT_CHUNK;

	/**
	 * Indicates that a parallel for loop will be executed by as many threads as
	 * there are cores on the machine.
	 */
	public static final int THREADS_EQUALS_CORES =
		TaskProperties.THREADS_EQUALS_CORES;

	/**
	 * Indicates to use the standard chunk size for the <TT>schedule</TT>
	 * property.
	 */
	public static final int STANDARD_CHUNK = TaskProperties.STANDARD_CHUNK;

	/**
	 * Indicates that the <TT>nodeName</TT> property is defaulted.
	 */
	public static final String DEFAULT_NODE_NAME =
		TaskProperties.DEFAULT_NODE_NAME;

	/**
	 * Indicates that the <TT>cores</TT> property is defaulted.
	 */
	public static final int DEFAULT_CORES = TaskProperties.DEFAULT_CORES;

	/**
	 * Indicates that the <TT>gpus</TT> property is defaulted.
	 */
	public static final int DEFAULT_GPUS = TaskProperties.DEFAULT_GPUS;

	/**
	 * Indicates that the task can run on any node of the cluster.
	 */
	public static final String ANY_NODE_NAME = TaskProperties.ANY_NODE_NAME;

	/**
	 * Indicates that the task requires all the cores on the node.
	 */
	public static final int ALL_CORES = TaskProperties.ALL_CORES;

	/**
	 * Indicates that the task requires all the GPU accelerators on the node.
	 */
	public static final int ALL_GPUS = TaskProperties.ALL_GPUS;

// Hidden data members.

	Rule rule;
	String taskClassName;
	String[] args;
	TaskProperties properties;
	String[] jvmFlags;
	boolean runInJob;
	EnumSet<Debug> debugs;

// Hidden constructors.

	/**
	 * Construct a new task specification.
	 *
	 * @param  rule       Rule that contains this task specification.
	 * @param  props      Chained job properties for this task specification.
	 * @param  jvmFlags   Initial JVM flags for this task specification.
	 * @param  taskClass  Task subclass.
	 */
	TaskSpec
		(Rule rule,
		 JobProperties props,
		 String[] jvmFlags,
		 Class<?> taskClass)
		{
		this.rule = rule;
		this.taskClassName = taskClass.getName();
		this.args = new String [0];
		this.properties = new TaskProperties() .chain (props);
		this.jvmFlags = jvmFlags;
		this.runInJob = false;
		this.debugs = null;

		// Query task class for task properties.
		TaskProperties initProps = new TaskProperties();
		this.properties.chain (initProps);
		try
			{
			Method method;
			method = getMethod (taskClass, "initialThreads");
			if (method != null)
				initProps.threads ((Integer) method.invoke (null));
			method = getMethod (taskClass, "initialSchedule");
			if (method != null)
				initProps.schedule ((Schedule) method.invoke (null));
			method = getMethod (taskClass, "initialChunk");
			if (method != null)
				initProps.chunk ((Integer) method.invoke (null));
			method = getMethod (taskClass, "nodeNameRequired");
			if (method != null)
				initProps.nodeName ((String) method.invoke (null));
			method = getMethod (taskClass, "coresRequired");
			if (method != null)
				initProps.cores ((Integer) method.invoke (null));
			method = getMethod (taskClass, "gpusRequired");
			if (method != null)
				initProps.gpus ((Integer) method.invoke (null));
			}
		catch (Throwable exc)
			{
			throw new TerminateException ("TaskSpec(): Shouldn't happen", exc);
			}
		}

	/**
	 * Get the zero-argument method with the given name in the given class. If
	 * not found, look in the superclasses. If still not found, return null.
	 *
	 * @param  cls         Class.
	 * @param  methodName  Method name.
	 *
	 * @return  Method object, or null if not found.
	 */
	private static Method getMethod
		(Class<?> cls,
		 String methodName)
		{
		Method method = null;
		while (method == null && cls != null)
			{
			try
				{
				method = cls.getDeclaredMethod (methodName);
				}
			catch (NoSuchMethodException exc)
				{
				cls = cls.getSuperclass();
				}
			}
		if (method != null) method.setAccessible (true);
		return method;
		}

// Exported operations.

	/**
	 * Specify the task's command line arguments. The strings specified as the
	 * <TT>args()</TT> method's arguments are appended to the task's array of
	 * command line arguments. If not specified, by default the task has no
	 * command line arguments.
	 *
	 * @param  args  Command line arguments (zero or more).
	 *
	 * @return  This task specification.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>args</TT> is null or any element
	 *     of <TT>args</TT> is null.
	 */
	public TaskSpec args
		(String... args)
		{
		if (args == null)
			throw new NullPointerException
				("TaskSpec.args(): args is null");
		for (int i = 0; i < args.length; ++ i)
			if (args[i] == null)
				throw new NullPointerException (String.format
					("TaskSpec.args(): args[%d] is null", i));
		int oldlen = this.args.length;
		int addlen = args.length;
		if (addlen > 0)
			{
			String[] newargs = new String [oldlen + addlen];
			System.arraycopy (this.args, 0, newargs, 0, oldlen);
			System.arraycopy (args, 0, newargs, oldlen, addlen);
			this.args = newargs;
			}
		return this;
		}

	/**
	 * Specify the task's <TT>threads</TT> property. The <TT>threads</TT>
	 * property specifies the number of threads that will execute a parallel for
	 * loop. If not specified, the default is to use the <TT>threads</TT>
	 * property of the enclosing job.
	 *
	 * @param  threads  Number of threads (&ge; 1), {@link
	 *                  #THREADS_EQUALS_CORES}, or {@link #DEFAULT_THREADS}.
	 *
	 * @return  This task specification.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>threads</TT> is illegal.
	 *
	 * @see  Task#threads(int)
	 */
	public TaskSpec threads
		(int threads)
		{
		properties.threads (threads);
		return this;
		}

	/**
	 * Specify the task's <TT>schedule</TT> property. The <TT>schedule</TT>
	 * property, along with the <TT>chunk</TT> property, specifies how the
	 * iterations of a parallel for loop are partitioned among the threads
	 * executing the parallel for loop. If not specified, the default is to use
	 * the <TT>schedule</TT> property of the enclosing job. Refer to enum
	 * {@linkplain Schedule} for descriptions of the possible schedules.
	 *
	 * @param  schedule  Parallel for loop schedule, or {@link
	 *                   #DEFAULT_SCHEDULE}.
	 *
	 * @return  This task specification.
	 *
	 * @see  Task#schedule(Schedule)
	 * @see  #chunk(int)
	 */
	public TaskSpec schedule
		(Schedule schedule)
		{
		properties.schedule (schedule);
		return this;
		}

	/**
	 * Specify the task's <TT>chunk</TT> property. The <TT>chunk</TT> property,
	 * along with the <TT>schedule</TT> property, specifies how the iterations
	 * of a parallel for loop are partitioned among the threads executing the
	 * parallel for loop. If not specified, the default is to use the
	 * <TT>chunk</TT> property of the enclosing job. Refer to enum {@linkplain
	 * Schedule} for descriptions of the possible schedules.
	 *
	 * @param  chunk  Chunk size (&ge; 1), {@link #STANDARD_CHUNK}, or {@link
	 *                #DEFAULT_CHUNK}.
	 *
	 * @return  This task specification.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>chunk</TT> is illegal.
	 *
	 * @see  Task#chunk(int)
	 * @see  #schedule(Schedule)
	 */
	public TaskSpec chunk
		(int chunk)
		{
		properties.chunk (chunk);
		return this;
		}

	/**
	 * Specify the task's <TT>nodeName</TT> property. The <TT>nodeName</TT>
	 * property specifies the name of the cluster node on which the task must
	 * run. The default is that the task can run on any node of the cluster.
	 *
	 * @param  nodeName  Node name, {@link #ANY_NODE_NAME}, or {@link
	 *                   #DEFAULT_NODE_NAME}.
	 *
	 * @return  This task specification.
	 *
	 * @see  Task#nodeName(String)
	 */
	public TaskSpec nodeName
		(String nodeName)
		{
		properties.nodeName (nodeName);
		return this;
		}

	/**
	 * Specify the task's <TT>cores</TT> property. The <TT>cores</TT> property
	 * specifies the number of CPU cores the task requires. The default is that
	 * the task requires all the cores on the node.
	 *
	 * @param  cores  Number of cores (&ge; 1), {@link #ALL_CORES}, or {@link
	 *                #DEFAULT_CORES}.
	 *
	 * @return  This task specification.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>cores</TT> is illegal.
	 *
	 * @see  Task#cores(int)
	 */
	public TaskSpec cores
		(int cores)
		{
		properties.cores (cores);
		return this;
		}

	/**
	 * Specify the task's <TT>gpus</TT> property. The <TT>gpus</TT> property
	 * specifies the number of GPU accelerators the task requires. The default
	 * is none (0).
	 *
	 * @param  gpus  Number of GPUs (&ge; 0), {@link #ALL_GPUS}, or {@link
	 *               #DEFAULT_GPUS}.
	 *
	 * @return  This task specification.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>gpus</TT> is illegal.
	 *
	 * @see  Task#gpus(int)
	 */
	public TaskSpec gpus
		(int gpus)
		{
		properties.gpus (gpus);
		return this;
		}

	/**
	 * Specify the JVM flags for the task. If not specified, the default is to
	 * use the <TT>jvmFlags</TT> property of the enclosing job. The JVM flags
	 * are included on the command line of the JVM that runs the task.
	 *
	 * @param  jvmFlags  JVM flags (zero or more).
	 *
	 * @return  This task specification.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>jvmFlags</TT> is null or any
	 *     element of <TT>jvmFlags</TT> is null.
	 */
	public TaskSpec jvmFlags
		(String... jvmFlags)
		{
		if (jvmFlags == null)
			throw new NullPointerException
				("TaskSpec.jvmFlags(): jvmFlags is null");
		for (int i = 0; i < jvmFlags.length; ++ i)
			if (jvmFlags[i] == null)
				throw new NullPointerException (String.format
					("TaskSpec.jvmFlags(): jvmFlags[%d] is null", i));
		this.jvmFlags = jvmFlags;
		return this;
		}

	/**
	 * Specify that the task must run in the job's process. If not specified,
	 * the default is to run in another process, possibly on another node.
	 *
	 * @return  This task specification.
	 */
	public TaskSpec runInJobProcess()
		{
		this.runInJob = true;
		return this;
		}

	/**
	 * Print the given debugging messages for the task. These debug settings
	 * override the overall job's debug settings for the task.
	 *
	 * @param  debug  Debugging message(s) to print, or null to print none.
	 *
	 * @return  This task specification.
	 */
	public TaskSpec debug
		(Debug... debug)
		{
		debugs = EnumSet.noneOf (Debug.class);
		if (debug != null)
			for (Debug d : debug)
				debugs.add (d);
		return this;
		}

	/**
	 * Print the given debugging messages for the task. These debug settings
	 * override the overall job's debug settings for the task.
	 *
	 * @param  debug  Set of debugging message(s) to print, or null to print
	 *                none.
	 *
	 * @return  This task specification.
	 */
	public TaskSpec debug
		(Set<Debug> debug)
		{
		debugs = EnumSet.noneOf (Debug.class);
		if (debug != null)
			debugs.addAll (debug);
		return this;
		}

	/**
	 * Add the given task to this task specification's rule's task group. When
	 * this rule fires, one instance of the given task will be executed. Call
	 * methods on the returned {@linkplain TaskSpec} object to configure the
	 * task.
	 *
	 * @param  <T>        Task data type.
	 * @param  taskClass  Task subclass.
	 *
	 * @return  Task specification for the given task.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>taskClass</TT> is null.
	 */
	public <T extends Task> TaskSpec task
		(Class<T> taskClass)
		{
		return rule.task (1, taskClass);
		}

	/**
	 * Add the given number of copies of the given task to this task
	 * specification's rule's task group. When this rule fires, the given number
	 * of instances of the given task will be executed. Call methods on the
	 * returned {@linkplain TaskSpec} object to configure the tasks (all copies
	 * will be configured the same).
	 *
	 * @param  <T>        Task data type.
	 * @param  n          Number of copies (&ge; 1).
	 * @param  taskClass  Task subclass.
	 *
	 * @return  Task specification for the given task.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>n</TT> &lt; 1.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>taskClass</TT> is null.
	 */
	public <T extends Task> TaskSpec task
		(int n,
		 Class<T> taskClass)
		{
		return rule.task (n, taskClass);
		}

// Hidden operations.

	/**
	 * Get the <TT>nodeName</TT>, <TT>cores</TT>, and <TT>gpus</TT> properties.
	 *
	 * @return  Node properties object.
	 */
	NodeProperties node()
		{
		return new NodeProperties
			(properties.nodeName(),
			 properties.cores(),
			 properties.gpus());
		}

	}
