//******************************************************************************
//
// File:    WorkerParallelForLoop.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.WorkerParallelForLoop
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

import edu.rit.pj2.tracker.TaskProperties;
import java.io.IOException;

/**
 * Class WorkerParallelForLoop provides the worker portion of a master-worker
 * cluster parallel for loop.
 * <P>
 * <B>Programming.</B>
 * To program a master-worker cluster parallel for loop, you must program the
 * master portion in the {@linkplain Job Job}, and you must program the worker
 * portion in a worker {@linkplain Task Task}. To program the master portion, in
 * the {@linkplain Job Job} subclass's {@link Job#main(String[]) main()} method:
 * <UL>
 * <P><LI>
 * Optionally, define the job's <TT>workers</TT>, <TT>masterSchedule</TT>, and
 * <TT>masterChunk</TT> properties by calling the {@link Job#workers(int)
 * workers()}, {@link Job#masterSchedule(Schedule) masterSchedule()}, or {@link
 * Job#masterChunk(int) masterChunk()} methods. These properties can also be
 * specified on the {@link pj2 pj2} command line. If any property is not
 * specified, a default value is used.
 * <P><LI>
 * If the loop index is type <TT>int</TT>, call the {@link
 * Job#masterFor(int,int,Class) masterFor(int,int,Class)} method, specifying the
 * loop index <I>inclusive</I> lower and upper bounds along with the worker task
 * class.
 * <P><LI>
 * If the loop index is type <TT>long</TT>, call the {@link
 * Job#masterFor(long,long,Class) masterFor(long,long,Class)} method, specifying
 * the loop index <I>inclusive</I> lower and upper bounds along with the worker
 * task class.
 * <P><LI>
 * The <TT>masterFor()</TT> method returns a {@linkplain TaskSpec TaskSpec}
 * object. Optionally, call methods on the task specification object to
 * configure the worker tasks.
 * </UL>
 * <P>
 * To program the worker portion, in the worker {@linkplain Task Task}
 * subclass's {@link Task#main(String[]) main()} method:
 * <UL>
 * <P><LI>
 * Call the {@link Task#workerFor() workerFor()} method, which returns a
 * WorkerParallelForLoop object.
 * <P><LI>
 * Optionally, define the worker parallel for loop's <TT>threads</TT>,
 * <TT>schedule</TT>, and <TT>chunk</TT> properties by calling the {@link
 * #threads(int) threads()}, {@link #schedule(Schedule) schedule()}, or {@link
 * #chunk(int) chunk()} methods on the WorkerParallelForLoop object. These
 * properties can also be specified on the {@link pj2 pj2} command line. If any
 * property is not specified, the enclosing task's property is used.
 * <P><LI>
 * If the loop index is type <TT>int</TT>, call the {@link #exec(Loop)
 * exec(Loop)} method, specifying the parallel for loop body, which is an
 * instance of class {@linkplain Loop Loop}. See class {@linkplain Loop Loop}
 * for further information.
 * <P><LI>
 * If the loop index is type <TT>long</TT>, call the {@link #exec(LongLoop)
 * exec(LongLoop)} method, specifying the parallel for loop body, which is an
 * instance of class {@linkplain LongLoop LongLoop}. See class {@linkplain
 * LongLoop LongLoop} for further information.
 * </UL>
 * <P>
 * <B>Operation.</B>
 * Under the hood, the master-worker cluster parallel for loop operates as
 * follows. There is one master and there are <I>K</I> worker tasks, where
 * <I>K</I> is given by the job's <TT>workers</TT> property. (The default is one
 * worker task.) In the job, the master partitions the loop index range into a
 * series of {@linkplain Chunk Chunk} tuples (for an <TT>int</TT> loop index) or
 * {@linkplain LongChunk LongChunk} tuples (for a <TT>long</TT> loop index), and
 * puts these tuples into tuple space. The partitioning is specified by the
 * job's <TT>masterSchedule</TT> and <TT>masterChunk</TT> properties. (The
 * default is a fixed schedule. See enum {@linkplain Schedule Schedule} for
 * further information about the possible schedules.) The master also adds a
 * rule to the job with a task group of <I>K</I> worker tasks. The worker tasks
 * start executing when the job starts.
 * <P>
 * In each worker task, when the {@link #exec(Loop) exec()} method is called,
 * the worker sets up a team of <I>T</I> threads, where <I>T</I> is given by the
 * worker parallel for loop's <TT>threads</TT> property. (The default is one
 * thread for each core of the machine where the worker task is running.) Each
 * team thread operates on its own copy of the {@linkplain Loop Loop} object.
 * Each team thread calls its loop object's {@link Loop#start() start()} method.
 * The worker repeatedly takes a {@linkplain Chunk Chunk} or {@linkplain
 * LongChunk LongChunk} tuple from tuple space and performs the loop iterations
 * specified by that chunk. The chunk's iterations are partitioned among the
 * team threads as specified by the worker parallel for loop's <TT>schedule</TT>
 * and <TT>chunk</TT> properties. (The default is a fixed schedule. See enum
 * {@linkplain Schedule Schedule} for further information about the possible
 * schedules.) Each team thread performs a series of iterations by calling its
 * loop object's {@link Loop#run(int) run()} method repeatedly. When all the
 * team threads have finished all the iterations for all the chunks, each team
 * thread calls its loop object's {@link Loop#finish() finish()} method.
 * <P>
 * Thus, a master-worker cluster parallel for loop has two levels of loop
 * iteration partitioning: the job level specified by the
 * <TT>masterSchedule</TT> and <TT>masterChunk</TT> properties, and the worker
 * task level specified by the <TT>schedule</TT> and <TT>chunk</TT> properties.
 * These properties can be used to achieve load balancing in a cluster parallel
 * program. Generally, coarse-grained (large) chunks are used at the job level
 * and fine-grained (small) chunks are used at the task level.
 * <P>
 * <B>Early loop exit.</B>
 * To terminate a master-worker cluster parallel for loop before all the loop
 * iterations have been executed:
 * <UL>
 * <P><LI>
 * Tell all the worker tasks to terminate execution. This is typically done by
 * putting a special "stop tuple" into tuple space, and having each worker task
 * take that tuple in a separate thread.
 * <P><LI>
 * Tell the worker parallel for loop in each task to terminate execution. This
 * is done by calling the {@link Loop#stop() stop()} method inside the loop's
 * {@link Loop#run(int) run()} method once the stop tuple appears.
 * </UL>
 * <P>
 * <B><I>Warning.</I></B>
 * There can be only one master-worker cluster parallel for loop in a job.
 *
 * @see  Loop
 * @see  LongLoop
 *
 * @author  Alan Kaminsky
 * @version 09-Jan-2014
 */
public class WorkerParallelForLoop
	extends ParallelForLoop
	{

// Hidden constructors.

	/**
	 * Construct a new worker parallel for loop.
	 *
	 * @param  task  Task in which the worker parallel for loop is executing.
	 */
	WorkerParallelForLoop
		(Task task)
		{
		super (task);
		}

// Exported operations.

	/**
	 * Set this parallel for loop's <TT>threads</TT> property. The
	 * <TT>threads</TT> property specifies the number of threads that will
	 * execute this parallel for loop. The default is the <TT>threads</TT>
	 * property of the enclosing task. For further information, see the {@link
	 * Task#threads(int) threads()} method of class {@linkplain Task}.
	 *
	 * @param  threads  Number of threads (&ge; 1), or {@link
	 *                  Task#DEFAULT_THREADS}.
	 *
	 * @return  This parallel for loop object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>threads</TT> is illegal.
	 */
	public WorkerParallelForLoop threads
		(int threads)
		{
		properties.threads (threads);
		return this;
		}

	/**
	 * Set this parallel for loop's <TT>schedule</TT> property. The
	 * <TT>schedule</TT> property, along with the <TT>chunk</TT> property,
	 * specifies how the iterations of this parallel for loop are partitioned
	 * among the threads executing this parallel for loop. Refer to enum
	 * {@linkplain Schedule} for descriptions of the possible schedules. The
	 * default is the <TT>schedule</TT> property of the enclosing task. For
	 * further information, see the {@link Task#schedule(Schedule) schedule()}
	 * method of class {@linkplain Task}.
	 *
	 * @param  schedule  Parallel for loop schedule.
	 *
	 * @return  This parallel for loop object.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>schedule</TT> is null.
	 *
	 * @see  #chunk(int)
	 */
	public WorkerParallelForLoop schedule
		(Schedule schedule)
		{
		properties.schedule (schedule);
		return this;
		}

	/**
	 * Set this parallel for loop's <TT>chunk</TT> property. The <TT>chunk</TT>
	 * property, along with the <TT>schedule</TT> property, specifies how the
	 * iterations of this parallel for loop are partitioned among the threads
	 * executing this parallel for loop. Refer to enum {@linkplain Schedule} for
	 * descriptions of the possible schedules. The default is the <TT>chunk</TT>
	 * property of the enclosing task. For further information, see the {@link
	 * Task#chunk(int) chunk()} method of class {@linkplain Task}.
	 *
	 * @param  chunk  Chunk size (&ge; 1), or {@link Task#DEFAULT_CHUNK}.
	 *
	 * @return  This parallel for loop object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>chunk</TT> is illegal.
	 *
	 * @see  #schedule(Schedule)
	 */
	public WorkerParallelForLoop chunk
		(int chunk)
		{
		properties.chunk (chunk);
		return this;
		}

	/**
	 * Execute this parallel for loop with the loop body specified by the given
	 * {@linkplain Loop Loop} object.
	 *
	 * @param  loop  Loop object.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>loop</TT> is null.
	 */
	public void exec
		(Loop loop)
		{
		new WorkerIntParallelForLoop (task, properties) .exec (loop);
		}

	/**
	 * Execute this parallel for loop with the loop body specified by the given
	 * {@linkplain LongLoop LongLoop} object.
	 *
	 * @param  loop  Loop object.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>loop</TT> is null.
	 */
	public void exec
		(LongLoop loop)
		{
		new WorkerLongParallelForLoop (task, properties) .exec (loop);
		}

// Hidden operations.

	/**
	 * Execute this parallel statement.
	 *
	 * @param  rank          Rank of the team thread.
	 * @param  reductionMap  Reduction map of the team thread.
	 *
	 * @exception  Exception
	 *     The <TT>run()</TT> method may throw any exception.
	 */
	void run
		(int rank,
		 ReductionMap reductionMap)
		throws Exception
		{
		throw new UnsupportedOperationException
			("WorkerParallelForLoop.run(): Shouldn't happen");
		}

	}
