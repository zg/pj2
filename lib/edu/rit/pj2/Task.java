//******************************************************************************
//
// File:    Task.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.Task
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

import edu.rit.pj2.tracker.JobRef;
import edu.rit.pj2.tracker.TaskProperties;
import edu.rit.util.AList;
import java.io.IOException;
import java.util.Iterator;

/**
 * Class Task is the abstract base class for a PJ2 task. A PJ2 parallel program
 * is either a single task (a subclass of class Task) or is a number of tasks
 * grouped together into a job (a subclass of class {@linkplain Job Job}). A PJ2
 * program is executed by running the PJ2 launcher program, class {@link pj2
 * pj2}.
 * <P>
 * <B>Programming a task.</B>
 * To set up a task:
 * <UL>
 * <P><LI>
 * Write a subclass of class Task.
 * <P><LI>
 * Optionally, specify the task's initial properties by overriding the {@link
 * #initialThreads() initialThreads()}, {@link #initialSchedule()
 * initialSchedule()}, and/or {@link #initialChunk() initialChunk()} methods.
 * <P><LI>
 * Optionally, specify the node capabilities required to execute the task by
 * overriding the {@link #nodeNameRequired() nodeNameRequired()}, {@link
 * #coresRequired() coresRequired()}, and/or {@link #gpusRequired()
 * gpusRequired()} methods.
 * <P><LI>
 * In the Task subclass's {@link #main(String[]) main()} method, write the code
 * for the task's computation.
 * </UL>
 * <P>
 * The task's computation can contain sequential statements executed by a single
 * thread, interspersed with parallel statements executed by multiple threads in
 * parallel. The possible parallel statements are:
 * <UL>
 * <LI>
 * Work sharing parallel for loop, integer loop index (see {@link
 * #parallelFor(int,int) parallelFor(int,int)})
 * <LI>
 * Work sharing parallel for loop, long integer loop index (see {@link
 * #parallelFor(long,long) parallelFor(long,long)})
 * <LI>
 * Work sharing parallel for loop over a series of work items in a {@linkplain
 * WorkQueue} (see {@link #parallelFor(WorkQueue) parallelFor(WorkQueue)})
 * <LI>
 * Parallel section group, multiple different sections (see {@link
 * #parallelDo(Section[]) parallelDo(Section...)})
 * <LI>
 * Parallel section group, multiple copies of the same section (see {@link
 * #parallelDo(int,Section) parallelDo(int,Section)})
 * <LI>
 * Worker portion of a master-worker cluster parallel for loop (see {@link
 * #workerFor() workerFor()})
 * </UL>
 * <P>
 * The task's computation can query the task's properties by calling the {@link
 * #threads() threads()}, {@link #schedule() schedule()}, {@link #chunk()
 * chunk()}, {@link #nodeName() nodeName()}, {@link #cores() cores()}, and
 * {@link #gpus() gpus()} methods. The task's computation can alter the task's
 * initial properties by calling the {@link #threads(int) threads(int)}, {@link
 * #schedule(Schedule) schedule(Schedule)}, and {@link #chunk(int) chunk(int)}
 * methods. While the task's computation can alter the task's <TT>nodeName</TT>,
 * <TT>cores</TT>, and <TT>gpus</TT> properties, doing so has no effect once the
 * task has started.
 * <P>
 * Shared variables used by the thread executing the sequential statements and
 * by the threads executing the parallel statements are typically declared as
 * fields of the Task subclass. This puts the shared variables in scope both for
 * code in the {@link #main(String[]) main()} method and for code in the
 * parallel statements.
 * <P>
 * <B>Running a task.</B>
 * To run a task, type this command:
 * <P>
 * <TT>$ java pj2 MainTask <I>arg</I> ...</TT>
 * <P>
 * The above command runs class MainTask, a subclass of class Task. Replace
 * <TT>MainTask</TT> with the fully-qualified class name of the desired task
 * subclass. The {@link pj2 pj2} program creates an instance of the task
 * subclass, then calls the task's {@link #main(String[]) main()} method,
 * passing in an array of the command line argument strings (zero or more). The
 * {@link #main(String[]) main()} method carries out the computation.
 * <P>
 * <B>Tasks in a job.</B>
 * A PJ2 task may be executed as part of a {@linkplain Job Job}. The job
 * maintains a repository of {@linkplain Tuple Tuple}s known as <I>tuple
 * space.</I> The job's execution is driven by <I>firing</I> the job's
 * {@linkplain Rule Rule}s, based on tuples that are written into tuple space.
 * Each rule has an associated <I>task group</I> consisting of one or more
 * tasks. There are three kinds of rules:
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
 * When a rule fires, for each task in the rule's task group, an instance of the
 * task is constructed, and the task's {@link Task#main(String[]) main()} method
 * is called to carry out the task's computation. For further information about
 * the criteria for firing rules and the actions performed when firing rules,
 * see class {@linkplain Rule Rule}.
 * <P>
 * The task can determine the number of tasks in its task group by calling the
 * {@link #groupSize() groupSize()} method. The task can determine its own rank
 * in its task group by calling the {@link #taskRank() taskRank()} method. The
 * tasks in a rule's task group are assigned successive ranks, starting with 0,
 * in the order the tasks were added to the rule. The task's rank uniquely
 * distinguishes the task from all other tasks in the task group.
 * <P>
 * The task may get <I>input tuples</I> in several ways. The task may take
 * tuples out of tuple space by calling the {@link #takeTuple(Tuple)
 * takeTuple()} or {@link #tryToTakeTuple(Tuple) tryToTakeTuple()} methods,
 * specifying a template for matching the desired tuple. The task may read
 * tuples, without taking them out of tuple space, by calling the {@link
 * #readTuple(Tuple) readTuple()} or {@link #tryToReadTuple(Tuple)
 * tryToReadTuple()} methods, specifying a template for matching the desired
 * tuple. The task may add a {@linkplain TupleListener} by calling the {@link
 * #addTupleListener(TupleListener) addTupleListener()} method. A task launched
 * by an on-demand rule may obtain one of the rule's matching tuples by calling
 * the {@link #getMatchingTuple(int) getMatchingTuple()} method.
 * <P>
 * The task may put <I>output tuples</I> into tuple space by calling the {@link
 * #putTuple(Tuple) putTuple()} method. Such tuples may trigger further rules to
 * fire and further tasks to be executed. The task may also specify that
 * multiple copies of a tuple are to be put into tuple space.
 * <P>
 * The task <I>must not alter its input tuples.</I> If an output tuple needs to
 * be the same as an input tuple but with modifications, create a copy of the
 * input tuple and modify that.
 * <P>
 * <B>I/O in a task.</B>
 * If a task is not running as part of a {@linkplain Job Job}, code in the task
 * can print on <TT>System.out</TT> or <TT>System.err</TT>, and the printouts
 * appear on the task's process's console as usual. The printouts actually
 * appear on the console when the task terminates; to make the printouts appear
 * earlier, call <TT>System.out.flush()</TT> or <TT>System.err.flush()</TT>.
 * Code in such a task can also read and write files as usual.
 * <P>
 * If a task is launched by a {@linkplain Job Job}, code in the task can print
 * on <TT>System.out</TT> or <TT>System.err</TT>, and the printouts are
 * automatically redirected to the job's process's console. The printouts
 * actually appear on the console when the task terminates; to make the
 * printouts appear earlier, call <TT>System.out.flush()</TT> or
 * <TT>System.err.flush()</TT>.
 * <P>
 * For other I/O, such as reading from the job's standard input, reading files,
 * and writing files, the task might need to run in the job's process. This
 * would be necessary, for example, if tasks running on separate nodes do not
 * run in the user's account and so cannot access the user's files. For further
 * information, see class {@linkplain Job Job}.
 * <P>
 * <B>JVM heap size in a task.</B>
 * If a task is not running as part of a {@linkplain Job Job}, you can specify
 * the maximum JVM heap size with a flag on the <TT>java</TT> command. For
 * example, to get a 2-gigabyte heap, type this command:
 * <P>
 * <TT>java -Xmx2000m pj2 MainTask <I>arg</I> ...</TT>
 * <P>
 * If a task is launched by a {@linkplain Job Job}, you can specify the maximum
 * JVM heap size by calling the {@link TaskSpec#jvmFlags(String[]) jvmFlags()}
 * method in the {@linkplain TaskSpec TaskSpec} in the {@linkplain Rule Rule}
 * that launches the task. For further information, see class {@linkplain Job
 * Job}.
 * <P>
 * <B>Abnormal termination.</B>
 * If code in a task throws an exception that propagates out of the
 * <TT>main()</TT> method, the {@link pj2 pj2} program prints an exception stack
 * trace and terminates.
 * <P>
 * Code in a task must <I>not</I> call the <TT>System.exit()</TT> method. If a
 * task needs to terminate itself, the task should throw an exception that
 * propagates out of every method without being caught, such as a {@linkplain
 * edu.rit.pj2.TerminateException TerminateException} or any other runtime
 * exception. The exception's detail message can explain why the task
 * terminated. This exception propagates out of the <TT>main()</TT> method and
 * causes the <TT>pj2</TT> program to terminate.
 *
 * @author  Alan Kaminsky
 * @version 25-Mar-2015
 */
public abstract class Task
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

// Hidden data members.

	JobRef job;                      // null if not part of a job
	long taskID;                     // 0 if not part of a job
	AList<Tuple> inputTuples;        // null if not part of a job
	TupleRequestMap tupleRequestMap; // null if not part of a job
	int groupSize;                   // 0 if not part of a job
	int taskRank;                    // 0 if not part of a job

	// Task properties.
	TaskProperties properties = new TaskProperties();

// Exported constructors.

	/**
	 * Construct a new task.
	 */
	public Task()
		{
		}

// Exported operations.

	/**
	 * Perform this task's computation.
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
	 * Set this task's <TT>threads</TT> property. The <TT>threads</TT> property
	 * specifies the number of threads that will execute a parallel for loop.
	 *
	 * @param  threads  Number of threads (&ge; 1), {@link
	 *                  #THREADS_EQUALS_CORES}, or {@link #DEFAULT_THREADS}.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>threads</TT> is illegal.
	 *
	 * @see  #threads()
	 * @see  #actualThreads()
	 */
	public void threads
		(int threads)
		{
		properties.threads (threads);
		}

	/**
	 * Get this task's <TT>threads</TT> property. The <TT>threads</TT> property
	 * specifies the number of threads that will execute a parallel for loop.
	 * The default is {@link #THREADS_EQUALS_CORES}, indicating that a parallel
	 * for loop will be executed by as many threads as there are cores on the
	 * machine.
	 * <P>
	 * <I>Note:</I> The <TT>threads()</TT> method returns the <TT>threads</TT>
	 * property <I>setting,</I> which may be {@link #THREADS_EQUALS_CORES}. For
	 * the actual number of threads, see the {@link #actualThreads()
	 * actualThreads()} method.
	 *
	 * @return  Number of threads (&ge; 1), or {@link #THREADS_EQUALS_CORES}.
	 *
	 * @see  #threads(int)
	 * @see  #actualThreads()
	 */
	public int threads()
		{
		return properties.threads();
		}

	/**
	 * Get the number of threads that will execute a parallel for loop.
	 * <P>
	 * <I>Note:</I> If the <TT>threads</TT> property is {@link
	 * #THREADS_EQUALS_CORES}, the <TT>actualThreads()</TT> method returns the
	 * actual number of threads that will execute a parallel for loop, namely
	 * the number of cores on the machine. For the <TT>threads</TT> property
	 * setting, see the {@link #threads() threads()} method.
	 *
	 * @return  Number of threads (&ge; 1).
	 *
	 * @see  #threads(int)
	 * @see  #threads()
	 */
	public int actualThreads()
		{
		return properties.actualThreads();
		}

	/**
	 * Set this task's <TT>schedule</TT> property. The <TT>schedule</TT>
	 * property, along with the <TT>chunk</TT> property, specifies how the
	 * iterations of a parallel for loop are partitioned among the threads
	 * executing the parallel for loop. Refer to enum {@linkplain Schedule} for
	 * descriptions of the possible schedules.
	 *
	 * @param  schedule  Parallel for loop schedule.
	 *
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
	 * Get this task's <TT>schedule</TT> property. The <TT>schedule</TT>
	 * property, along with the <TT>chunk</TT> property, specifies how the
	 * iterations of a parallel for loop are partitioned among the threads
	 * executing the parallel for loop. The default is a {@link Schedule#fixed
	 * fixed} schedule.
	 *
	 * @return  Parallel for loop schedule.
	 *
	 * @see  #schedule(Schedule)
	 * @see  #chunk(int)
	 * @see  #chunk()
	 */
	public Schedule schedule()
		{
		return properties.schedule();
		}

	/**
	 * Set this task's <TT>chunk</TT> property. The <TT>chunk</TT> property,
	 * along with the <TT>schedule</TT> property, specifies how the iterations
	 * of a parallel for loop are partitioned among the threads executing the
	 * parallel for loop. Refer to enum {@linkplain Schedule} for descriptions
	 * of the possible schedules.
	 *
	 * @param  chunk  Chunk size (&ge; 1), {@link #STANDARD_CHUNK}, or {@link
	 *                #DEFAULT_CHUNK}.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>chunk</TT> is illegal.
	 *
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
	 * Get this task's <TT>chunk</TT> property. The <TT>chunk</TT> property,
	 * along with the <TT>schedule</TT> property, specifies how the iterations
	 * of a parallel for loop are partitioned among the threads executing the
	 * parallel for loop. The default is {@link #STANDARD_CHUNK}, indicating the
	 * standard chunk size for the <TT>schedule</TT> property; see enum
	 * {@linkplain Schedule Schedule} for further information.
	 *
	 * @return  Chunk size (&ge; 1), {@link #STANDARD_CHUNK}.
	 *
	 * @see  #schedule(Schedule)
	 * @see  #schedule()
	 * @see  #chunk(int)
	 */
	public int chunk()
		{
		return properties.chunk();
		}

	/**
	 * Set this task's <TT>nodeName</TT> property. The <TT>nodeName</TT>
	 * property specifies the name of the cluster node on which the task must
	 * run.
	 *
	 * @param  nodeName  Node name, {@link #ANY_NODE_NAME}, or {@link
	 *                   #DEFAULT_NODE_NAME}.
	 *
	 * @see  #nodeName()
	 */
	public void nodeName
		(String nodeName)
		{
		properties.nodeName (nodeName);
		}

	/**
	 * Get this task's <TT>nodeName</TT> property. The <TT>nodeName</TT>
	 * property specifies the name of the cluster node on which the task must
	 * run. The default is {@link #ANY_NODE_NAME}, indicating that the task can
	 * run on any node of the cluster.
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
	 * Set this task's <TT>cores</TT> property. The <TT>cores</TT> property
	 * specifies the number of CPU cores the task requires.
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
	 * specifies the number of CPU cores the task requires. The default is
	 * {@link #ALL_CORES}, indicating that the task requires all the cores on
	 * the node.
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
	 * Set this task's <TT>gpus</TT> property. The <TT>gpus</TT> property
	 * specifies the number of GPU accelerators the task requires.
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
	 * Get this task's <TT>gpus</TT> property. The <TT>gpus</TT> property
	 * specifies the number of GPU accelerators the task requires. The default
	 * is 0, indicating that the task requires no GPU accelerators.
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
	 * Execute a work sharing parallel for loop with a loop index of type
	 * <TT>int</TT>. The loop index goes from the given lower bound to the given
	 * upper bound. For further information, see classes {@linkplain
	 * ParallelForLoop} and {@linkplain IntParallelForLoop}.
	 *
	 * @param  lb  Loop index lower bound (inclusive).
	 * @param  ub  Loop index upper bound (inclusive).
	 */
	public IntParallelForLoop parallelFor
		(int lb,
		 int ub)
		{
		return new IntParallelForLoop (this, lb, ub);
		}

	/**
	 * Execute a work sharing parallel for loop with a loop index of type
	 * <TT>long</TT>. The loop index goes from the given lower bound to the
	 * given upper bound. For further information, see classes {@linkplain
	 * ParallelForLoop} and {@linkplain LongParallelForLoop}.
	 *
	 * @param  lb  Loop index lower bound (inclusive).
	 * @param  ub  Loop index upper bound (inclusive).
	 */
	public LongParallelForLoop parallelFor
		(long lb,
		 long ub)
		{
		return new LongParallelForLoop (this, lb, ub);
		}

	/**
	 * Execute a work sharing parallel for loop over the work items in a
	 * {@linkplain WorkQueue}. For further information, see classes {@linkplain
	 * ParallelForLoop} and {@linkplain ObjectParallelForLoop}.
	 *
	 * @param  <W>    Data type of the work items.
	 * @param  queue  Work queue.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>queue</TT> is null.
	 */
	public <W> ObjectParallelForLoop<W> parallelFor
		(WorkQueue<W> queue)
		{
		return new ObjectParallelForLoop<W> (this, queue);
		}

	/**
	 * Execute the given group of sections in parallel. The sections' {@link
	 * Section#run() run()} methods are called simultaneously, each by a
	 * different thread. For further information, see class {@linkplain
	 * Section}.
	 *
	 * @param  sections  Sections to be executed in parallel.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if no sections are specified.
	 */
	public void parallelDo
		(final Section... sections)
		{
		final int NT = sections.length;
		if (NT == 0)
			throw new IllegalArgumentException
				("Task.parallelDo(): No sections specified");
		Team.execute (NT, new ParallelStatement (this)
			{
			void run (int rank, ReductionMap reductionMap) throws Exception
				{
				Section section = sections[rank];
				section.task = Task.this;
				section.threads = NT;
				section.rank = rank;
				section.reductionMap = reductionMap;
				section.run();
				section.task = null;
				section.threads = -1;
				section.rank = -1;
				section.reductionMap = null;
				}
			});
		}

	/**
	 * Execute the given number of copies of the given section in parallel. The
	 * copies are created by cloning the given section object. The sections'
	 * {@link Section#run() run()} methods are called simultaneously, each by a
	 * different thread. For further information, see class {@linkplain
	 * Section}.
	 *
	 * @param  K        Number of sections (&ge; 1).
	 * @param  section  Section to be executed in parallel.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>K</TT> &lt; 1.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>section</TT> is null.
	 */
	public void parallelDo
		(final int K,
		 final Section section)
		{
		if (K < 1)
			throw new IllegalArgumentException (String.format
				("Task.parallelDo(): K = %d illegal", K));
		if (section == null)
			throw new NullPointerException
				("Task.parallelDo(): section is null");
		Team.execute (K, new ParallelStatement (this)
			{
			void run (int rank, ReductionMap reductionMap) throws Exception
				{
				Section sec = rank == 0 ? section : (Section) section.clone();
				sec.task = Task.this;
				sec.threads = K;
				sec.rank = rank;
				sec.reductionMap = reductionMap;
				sec.run();
				sec.task = null;
				sec.threads = -1;
				sec.rank = -1;
				sec.reductionMap = null;
				}
			});
		}

	/**
	 * Execute the worker portion of a master-worker cluster parallel for loop.
	 * To complete the master-worker cluster parallel for loop, the {@linkplain
	 * Job Job}'s {@link Job#main(String[]) main()} method must call the {@link
	 * Job#masterFor(int,int,Class) masterFor(int,int,Class)} or {@link
	 * Job#masterFor(long,long,Class) masterFor(long,long,Class)} method. For
	 * further information, see class {@linkplain WorkerParallelForLoop}.
	 *
	 * @return  Worker parallel for loop object.
	 */
	public WorkerParallelForLoop workerFor()
		{
		return new WorkerParallelForLoop (this);
		}

	/**
	 * Returns the number of tasks in this task's task group.
	 *
	 * @return  Task group size.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this task is not executing as part of
	 *     a job.
	 */
	public int groupSize()
		{
		if (job == null)
			throw new IllegalStateException
				("Task.groupSize(): Task is not part of a job");
		return groupSize;
		}

	/**
	 * Returns this task's rank within its task group.
	 *
	 * @return  Rank in the range 0 .. <TT>groupSize()</TT>&minus;1.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this task is not executing as part of
	 *     a job.
	 */
	public int taskRank()
		{
		if (job == null)
			throw new IllegalStateException
				("Task.taskRank(): Task is not part of a job");
		return taskRank;
		}

	/**
	 * Returns the number of matching tuples for this task. For a task launched
	 * by an on-demand {@linkplain Rule Rule}, the number of matching tuples is
	 * the number of templates specified in the rule. For a task launched by a
	 * start rule or a finish rule, there are no matching tuples.
	 *
	 * @return  Number of matching tuples.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this task is not executing as part of
	 *     a job.
	 */
	public int matchingTupleCount()
		{
		if (job == null)
			throw new IllegalStateException
				("Task.matchingTupleCount(): Task is not part of a job");
		return inputTuples.size();
		}

	/**
	 * Get the given matching tuple for this task. The tuple index denotes one
	 * of the templates in the on-demand {@linkplain Rule Rule} that launched
	 * this task. This method returns the tuple that matched the template at the
	 * given index.
	 * <P>
	 * The task <I>must not alter its input tuples.</I> If an output tuple needs
	 * to be the same as an input tuple but with modifications, create a copy of
	 * the input tuple and modify that.
	 *
	 * @param  i  Tuple index in the range 0 ..
	 *            <TT>matchingTupleCount()</TT>&minus;1.
	 *
	 * @return  Matching tuple at index <TT>i</TT>.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this task is not executing as part of
	 *     a job.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>i</TT> is out of bounds.
	 */
	public Tuple getMatchingTuple
		(int i)
		{
		if (job == null)
			throw new IllegalStateException
				("Task.getMatchingTuple(): Task is not part of a job");
		return inputTuples.get (i);
		}

	/**
	 * Take a tuple that matches the given template out of tuple space. If
	 * there are no matching tuples in tuple space, this method blocks until a
	 * matching tuple appears in tuple space. If there is more than one matching
	 * tuple in tuple space, one of the matching tuples is chosen in an
	 * unspecified manner.
	 * <P>
	 * The task <I>must not alter its input tuples.</I> If an output tuple needs
	 * to be the same as an input tuple but with modifications, create a copy of
	 * the input tuple and modify that.
	 *
	 * @param  <T>       Tuple data type.
	 * @param  template  Template.
	 *
	 * @return  Tuple that matches the <TT>template</TT>.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this task is not executing as part of
	 *     a job.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>template</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public <T extends Tuple> T takeTuple
		(T template)
		throws IOException
		{
		if (job == null)
			throw new IllegalStateException
				("Task.takeTuple(): Task is not part of a job");
		if (template == null)
			throw new NullPointerException
				("Task.takeTuple(): template is null");
		long id = tupleRequestMap.addRequest (taskID, null);
		job.takeTuple (taskID, id, template, true, true);
		T tuple = (T) tupleRequestMap.takeTuple (taskID, id);
		if (tuple == null)
			throw new IllegalStateException
				("Task.takeTuple(): Shouldn't happen");
		else
			return tuple;
		}

	/**
	 * Try to take a tuple that matches the given template out of tuple space.
	 * If there are no matching tuples in tuple space, this method returns null
	 * without blocking. If there is more than one matching tuple in tuple
	 * space, one of the matching tuples is chosen in an unspecified manner.
	 * <P>
	 * The task <I>must not alter its input tuples.</I> If an output tuple needs
	 * to be the same as an input tuple but with modifications, create a copy of
	 * the input tuple and modify that.
	 *
	 * @param  <T>       Tuple data type.
	 * @param  template  Template.
	 *
	 * @return  Tuple that matches the <TT>template</TT>, or null if none.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this task is not executing as part of
	 *     a job.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>template</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public <T extends Tuple> T tryToTakeTuple
		(T template)
		throws IOException
		{
		if (job == null)
			throw new IllegalStateException
				("Task.tryToTakeTuple(): Task is not part of a job");
		if (template == null)
			throw new NullPointerException
				("Task.tryToTakeTuple(): template is null");
		long id = tupleRequestMap.addRequest (taskID, null);
		job.takeTuple (taskID, id, template, false, true);
		return (T) tupleRequestMap.takeTuple (taskID, id);
		}

	/**
	 * Read a tuple that matches the given template. If there are no matching
	 * tuples in tuple space, this method blocks until a matching tuple appears
	 * in tuple space. If there is more than one matching tuple in tuple space,
	 * one of the matching tuples is chosen in an unspecified manner. The
	 * matching tuple remains in tuple space.
	 * <P>
	 * The task <I>must not alter its input tuples.</I> If an output tuple needs
	 * to be the same as an input tuple but with modifications, create a copy of
	 * the input tuple and modify that.
	 *
	 * @param  <T>       Tuple data type.
	 * @param  template  Template.
	 *
	 * @return  Tuple that matches the <TT>template</TT>.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this task is not executing as part of
	 *     a job.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>template</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public <T extends Tuple> T readTuple
		(T template)
		throws IOException
		{
		if (job == null)
			throw new IllegalStateException
				("Task.readTuple(): Task is not part of a job");
		if (template == null)
			throw new NullPointerException
				("Task.readTuple(): template is null");
		long id = tupleRequestMap.addRequest (taskID, null);
		job.takeTuple (taskID, id, template, true, false);
		T tuple = (T) tupleRequestMap.takeTuple (taskID, id);
		if (tuple == null)
			throw new IllegalStateException
				("Task.readTuple(): Shouldn't happen");
		else
			return tuple;
		}

	/**
	 * Try to read a tuple that matches the given template. If there are no
	 * matching tuples in tuple space, this method returns null without
	 * blocking. If there is more than one matching tuple in tuple space, one of
	 * the matching tuples is chosen in an unspecified manner. The matching
	 * tuple remains in tuple space.
	 * <P>
	 * The task <I>must not alter its input tuples.</I> If an output tuple needs
	 * to be the same as an input tuple but with modifications, create a copy of
	 * the input tuple and modify that.
	 *
	 * @param  <T>       Tuple data type.
	 * @param  template  Template.
	 *
	 * @return  Tuple that matches the <TT>template</TT>, or null if none.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this task is not executing as part of
	 *     a job.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>template</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public <T extends Tuple> T tryToReadTuple
		(T template)
		throws IOException
		{
		if (job == null)
			throw new IllegalStateException
				("Task.tryToReadTuple(): Task is not part of a job");
		if (template == null)
			throw new NullPointerException
				("Task.tryToReadTuple(): template is null");
		long id = tupleRequestMap.addRequest (taskID, null);
		job.takeTuple (taskID, id, template, false, false);
		return (T) tupleRequestMap.takeTuple (taskID, id);
		}

	/**
	 * Add the given tuple listener to this task. The tuple listener specifies a
	 * template and an operation, either read or take. When there is a tuple in
	 * tuple space that matches the template, the tuple listener is triggered,
	 * the matching tuple is read or taken, and the matching tuple is passed to
	 * the tuple listener's {@link TupleListener#run(Tuple) run()} method. The
	 * tuple listener will not be triggered again. For further information, see
	 * class {@linkplain TupleListener}.
	 * <P>
	 * The task <I>must not alter its input tuples.</I> If an output tuple needs
	 * to be the same as an input tuple but with modifications, create a copy of
	 * the input tuple and modify that.
	 *
	 * @param  listener  Tuple listener.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this task is not executing as part of
	 *     a job.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>listener</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void addTupleListener
		(TupleListener<?> listener)
		throws IOException
		{
		if (job == null)
			throw new IllegalStateException
				("Task.addTupleListener(): Task is not part of a job");
		if (listener == null)
			throw new NullPointerException
				("Task.addTupleListener(): listener is null");
		long id = tupleRequestMap.addRequest (taskID, listener);
		job.takeTuple (taskID, id, listener.template, true, listener.take);
		}

	/**
	 * Put the given output tuple into tuple space.
	 *
	 * @param  tuple  Output tuple.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this task is not executing as part of
	 *     a job.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>tuple</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void putTuple
		(Tuple tuple)
		throws IOException
		{
		putTuple (1, tuple);
		}

	/**
	 * Put multiple copies of the given output tuple into tuple space. If
	 * <TT>copies</TT> = 0, this method does nothing.
	 *
	 * @param  copies  Number of copies (&ge; 0).
	 * @param  tuple   Output tuple.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this task is not executing as part of
	 *     a job.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>tuple</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>copies</TT> &lt; 0.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void putTuple
		(int copies,
		 Tuple tuple)
		throws IOException
		{
		if (job == null)
			throw new IllegalStateException
				("Task.putTuple(): Task is not part of a job");
		if (tuple == null)
			throw new NullPointerException
				("Task.putTuple(): tuple is null");
		if (copies < 0)
			throw new NullPointerException (String.format
				("Task.putTuple(): copies = %d illegal", copies));
		else if (copies > 0)
			job.writeTuple (taskID, tuple, copies);
		}

// Hidden operations.

	/**
	 * Get this task class's initial <TT>threads</TT> property. When a
	 * {@linkplain TaskSpec} is created for this task class, the task spec's
	 * <TT>threads</TT> property is initialized to the value the
	 * <TT>initialThreads()</TT> method returns. The <TT>threads</TT> property
	 * specifies the number of threads that will execute a parallel for loop.
	 * <P>
	 * In the Task base class, this method returns {@link #DEFAULT_THREADS},
	 * which indicates that a parallel for loop will be executed by as many
	 * threads as there are cores on the machine. A subclass can override this
	 * method to return something else.
	 *
	 * @return  Number of threads (&ge; 1), {@link #THREADS_EQUALS_CORES}, or
	 *          {@link #DEFAULT_THREADS}.
	 */
	protected static int initialThreads()
		{
		return DEFAULT_THREADS;
		}

	/**
	 * Get this task class's initial <TT>schedule</TT> property. When a
	 * {@linkplain TaskSpec} is created for this task class, the task spec's
	 * <TT>schedule</TT> property is initialized to the value the
	 * <TT>initialSchedule()</TT> method returns. The <TT>schedule</TT>
	 * property, along with the <TT>chunk</TT> property, specifies how the
	 * iterations of a parallel for loop are partitioned among the threads
	 * executing the parallel for loop. Refer to enum {@linkplain Schedule} for
	 * descriptions of the possible schedules.
	 * <P>
	 * In the Task base class, this method returns {@link #DEFAULT_SCHEDULE},
	 * which indicates a {@link Schedule#fixed fixed} schedule. A subclass can
	 * override this method to return something else.
	 *
	 * @return  Parallel for loop schedule, or {@link #DEFAULT_SCHEDULE}.
	 */
	protected static Schedule initialSchedule()
		{
		return DEFAULT_SCHEDULE;
		}

	/**
	 * Get this task class's initial <TT>chunk</TT> property. When a {@linkplain
	 * TaskSpec} is created for this task class, the task spec's <TT>chunk</TT>
	 * property is initialized to the value the <TT>initialChunk()</TT> method
	 * returns. The <TT>chunk</TT> property, along with the <TT>schedule</TT>
	 * property, specifies how the iterations of a parallel for loop are
	 * partitioned among the threads executing the parallel for loop. Refer to
	 * enum {@linkplain Schedule} for descriptions of the possible schedules.
	 * <P>
	 * In the Task base class, this method returns {@link #DEFAULT_CHUNK}, which
	 * indicates the standard chunk size for the <TT>schedule</TT> property; see
	 * enum {@linkplain Schedule Schedule} for further information. A subclass
	 * can override this method to return something else.
	 *
	 * @return  Chunk size (&ge; 1), {@link #STANDARD_CHUNK}, or {@link
	 *          #DEFAULT_CHUNK}.
	 */
	protected static int initialChunk()
		{
		return DEFAULT_CHUNK;
		}

	/**
	 * Get this task class's required <TT>nodeName</TT> property. When a
	 * {@linkplain TaskSpec} is created for this task class, the task spec's
	 * <TT>nodeName</TT> property is set to the value the
	 * <TT>nodeNameRequired()</TT> method returns. The <TT>nodeName</TT>
	 * property specifies the name of the cluster node on which the task must
	 * run.
	 * <P>
	 * In the Task base class, this method returns {@link #DEFAULT_NODE_NAME},
	 * which indicates that the task can run on any node of the cluster. A
	 * subclass can override this method to return something else.
	 *
	 * @return  Node name, {@link #ANY_NODE_NAME}, or {@link
	 *          #DEFAULT_NODE_NAME}.
	 */
	protected static String nodeNameRequired()
		{
		return DEFAULT_NODE_NAME;
		}

	/**
	 * Get this task class's required <TT>cores</TT> property. When a
	 * {@linkplain TaskSpec} is created for this task class, the task spec's
	 * <TT>cores</TT> property is set to the value the <TT>coresRequired()</TT>
	 * method returns. The <TT>cores</TT> property specifies the number of CPU
	 * cores the task requires.
	 * <P>
	 * In the Task base class, this method returns {@link #DEFAULT_CORES}, which
	 * indicates that the task requires all the cores on the node. A subclass
	 * can override this method to return something else.
	 *
	 * @return  Number of cores (&ge; 1), {@link #ALL_CORES}, or {@link
	 *          #DEFAULT_CORES}.
	 */
	protected static int coresRequired()
		{
		return DEFAULT_CORES;
		}

	/**
	 * Get this task class's requred <TT>gpus</TT> property. When a {@linkplain
	 * TaskSpec} is created for this task class, the task spec's <TT>gpus</TT>
	 * property is set to the value the <TT>gpusRequired()</TT> method returns.
	 * The <TT>gpus</TT> property specifies the number of GPU accelerators the
	 * task requires.
	 * <P>
	 * In the Task base class, this method returns {@link #DEFAULT_GPUS}, which
	 * indicates that the task requires no GPU accelerators. A subclass can
	 * override this method to return something else.
	 *
	 * @return  Number of GPUs (&ge; 0), {@link #ALL_GPUS}, or {@link
	 *          #DEFAULT_GPUS}.
	 */
	protected static int gpusRequired()
		{
		return DEFAULT_GPUS;
		}

	}
