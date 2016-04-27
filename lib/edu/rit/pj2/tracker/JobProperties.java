//******************************************************************************
//
// File:    JobProperties.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.JobProperties
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

import edu.rit.pj2.Schedule;

/**
 * Class JobProperties encapsulates the properties of a {@linkplain
 * edu.rit.pj2.Job Job}. The following properties may be specified:
 * <UL>
 * <LI><TT>workers</TT> &mdash; The number of worker tasks in a master-worker
 * parallel for loop in the job.
 * <LI><TT>masterSchedule</TT> &mdash; The schedule for a master-worker parallel
 * loop in the job.
 * <LI><TT>masterChunk</TT> &mdash; The chunk size for a master-worker parallel
 * loop in the job.
 * <LI><TT>threads</TT> &mdash; The number of threads executing parallel loops
 * in a task in the job.
 * <LI><TT>schedule</TT> &mdash; The schedule for parallel loops in a task in
 * the job.
 * <LI><TT>chunk</TT> &mdash; The chunk size for parallel loops in a task in the
 * job.
 * <LI><TT>nodeName</TT> &mdash; The name of the node on which to run a task in
 * the job.
 * <LI><TT>cores</TT> &mdash; The number of CPU cores needed to run a task in
 * the job.
 * <LI><TT>gpus</TT> &mdash; The number of GPU accelerators needed to run a task
 * in the job.
 * </UL>
 *
 * @author  Alan Kaminsky
 * @version 30-May-2014
 */
public class JobProperties
	{

// Exported constants.

	/**
	 * Indicates that the <TT>workers</TT> property is defaulted.
	 */
	public static final int DEFAULT_WORKERS = LoopProperties.DEFAULT_THREADS;

	/**
	 * Indicates that the <TT>threads</TT> property is defaulted.
	 */
	public static final int DEFAULT_THREADS = LoopProperties.DEFAULT_THREADS;

	/**
	 * Indicates that the <TT>masterSchedule</TT> or <TT>schedule</TT> property
	 * is defaulted.
	 */
	public static final Schedule DEFAULT_SCHEDULE =
		LoopProperties.DEFAULT_SCHEDULE;

	/**
	 * Indicates that the <TT>masterChunk</TT> or <TT>chunk</TT> property is
	 * defaulted.
	 */
	public static final int DEFAULT_CHUNK = LoopProperties.DEFAULT_CHUNK;

	/**
	 * Indicates that a parallel for loop will be executed by as many threads as
	 * there are cores on the machine.
	 */
	public static final int THREADS_EQUALS_CORES =
		LoopProperties.THREADS_EQUALS_CORES;

	/**
	 * Indicates to use the standard chunk size for the <TT>schedule</TT>
	 * property.
	 */
	public static final int STANDARD_CHUNK = LoopProperties.STANDARD_CHUNK;

	/**
	 * Indicates that the <TT>nodeName</TT> property is defaulted.
	 */
	public static final String DEFAULT_NODE_NAME =
		NodeProperties.DEFAULT_NODE_NAME;

	/**
	 * Indicates that the <TT>cores</TT> property is defaulted.
	 */
	public static final int DEFAULT_CORES = NodeProperties.DEFAULT_CORES;

	/**
	 * Indicates that the <TT>gpus</TT> property is defaulted.
	 */
	public static final int DEFAULT_GPUS = NodeProperties.DEFAULT_GPUS;

	/**
	 * Indicates that the task can run on any node of the cluster.
	 */
	public static final String ANY_NODE_NAME = NodeProperties.ANY_NODE_NAME;

	/**
	 * Indicates that the task requires all the cores on the node.
	 */
	public static final int ALL_CORES = NodeProperties.ALL_CORES;

	/**
	 * Indicates that the task requires all the GPU accelerators on the node.
	 */
	public static final int ALL_GPUS = NodeProperties.ALL_GPUS;

// Hidden data members.

	LoopProperties masterLoopProps;
	LoopProperties loopProps;
	NodeProperties nodeProps;

// Exported constructors.

	/**
	 * Construct a new job properties object. All settings are defaulted.
	 */
	public JobProperties()
		{
		masterLoopProps = new LoopProperties();
		loopProps = new LoopProperties();
		nodeProps = new NodeProperties();
		}

	/**
	 * Construct a new job properties object with the given settings.
	 *
	 * @param  workers
	 *     Number of worker tasks for a master-worker parallel for loop, or
	 *     {@link #DEFAULT_WORKERS}.
	 * @param  masterSchedule
	 *     Schedule for a master-worker parallel for loop, or {@link
	 *     #DEFAULT_SCHEDULE}.
	 * @param  masterChunk
	 *     Chunk size for a master-worker parallel for loop, {@link
	 *     #STANDARD_CHUNK}, or {@link #DEFAULT_CHUNK}.
	 * @param  threads
	 *     Number of threads for a parallel for loop, {@link
	 *     #THREADS_EQUALS_CORES}, or {@link #DEFAULT_THREADS}.
	 * @param  schedule
	 *     Schedule for a parallel for loop, or {@link #DEFAULT_SCHEDULE}.
	 * @param  chunk
	 *     Chunk size for a parallel for loop, {@link #STANDARD_CHUNK}, or
	 *     {@link #DEFAULT_CHUNK}.
	 * @param  nodeName
	 *     Node name on which a task must execute, {@link #ANY_NODE_NAME}, or
	 *     {@link #DEFAULT_NODE_NAME}.
	 * @param  cores
	 *     Number of CPU cores a task requires, {@link #ALL_CORES}, or {@link
	 *     #DEFAULT_CORES}.
	 * @param  gpus
	 *     Number of GPU accelerators a task requires, {@link #ALL_GPUS}, or
	 *     {@link #DEFAULT_GPUS}.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>workers</TT>,
	 *     <TT>masterChunk</TT>, <TT>threads</TT>, <TT>chunk</TT>,
	 *     <TT>cores</TT>, or <TT>gpus</TT> is illegal.
	 */
	public JobProperties
		(int workers,
		 Schedule masterSchedule,
		 int masterChunk,
		 int threads,
		 Schedule schedule,
		 int chunk,
		 String nodeName,
		 int cores,
		 int gpus)
		{
		if (workers == THREADS_EQUALS_CORES)
			throw new IllegalArgumentException (String.format
				("JobProperties(): workers = %d illegal", workers));
		masterLoopProps = new LoopProperties
			(workers, masterSchedule, masterChunk);
		loopProps = new LoopProperties (threads, schedule, chunk);
		nodeProps = new NodeProperties (nodeName, cores, gpus);
		}

// Exported operations.

	/**
	 * Set the <TT>workers</TT> property. The <TT>workers</TT> property
	 * specifies the number of worker tasks that will execute a master-worker
	 * parallel for loop.
	 *
	 * @param  workers  Number of worker tasks, or {@link #DEFAULT_WORKERS}.
	 *
	 * @return  This job properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>threads</TT> is illegal.
	 *
	 * @see  #workers()
	 */
	public JobProperties workers
		(int workers)
		{
		if (workers == THREADS_EQUALS_CORES)
			throw new IllegalArgumentException (String.format
				("JobProperties.workers(): workers = %d illegal", workers));
		masterLoopProps.threads (workers);
		return this;
		}

	/**
	 * Set the <TT>workers</TT> property. The <TT>workers</TT> property
	 * specifies the number of worker tasks that will execute a master-worker
	 * parallel for loop. If the <TT>workers</TT> property is defaulted, 1 is
	 * returned, indicating that a master-worker parallel for loop will be
	 * executed by one worker task.
	 *
	 * @return  Number of worker tasks (&ge; 1).
	 *
	 * @see  #workers(int)
	 */
	public int workers()
		{
		int rv = masterLoopProps.threads();
		return rv <= THREADS_EQUALS_CORES ? 1 : rv;
		}

	/**
	 * Set the <TT>masterSchedule</TT> property. The <TT>masterSchedule</TT>
	 * property, along with the <TT>masterChunk</TT> property, specifies how the
	 * iterations of a master-worker parallel for loop are partitioned among the
	 * worker tasks executing the parallel for loop. Refer to enum {@linkplain
	 * Schedule} for descriptions of the possible schedules.
	 *
	 * @param  schedule  Parallel for loop schedule.
	 *
	 * @return  This job properties object.
	 *
	 * @see  #masterSchedule()
	 * @see  #masterChunk(int)
	 * @see  #masterChunk()
	 */
	public JobProperties masterSchedule
		(Schedule schedule)
		{
		masterLoopProps.schedule (schedule);
		return this;
		}

	/**
	 * Get the <TT>masterSchedule</TT> property. The <TT>masterSchedule</TT>
	 * property, along with the <TT>masterChunk</TT> property, specifies how the
	 * iterations of a master-worker parallel for loop are partitioned among the
	 * worker tasks executing the parallel for loop. If the
	 * <TT>masterSchedule</TT> property is defaulted, a {@link Schedule#fixed
	 * fixed} schedule is returned.
	 *
	 * @return  Parallel for loop schedule.
	 *
	 * @see  #masterSchedule(Schedule)
	 * @see  #masterChunk(int)
	 * @see  #masterChunk()
	 */
	public Schedule masterSchedule()
		{
		return masterLoopProps.schedule();
		}

	/**
	 * Set the <TT>masterChunk</TT> property. The <TT>masterChunk</TT> property,
	 * along with the <TT>masterSchedule</TT> property, specifies how the
	 * iterations of a master-worker parallel for loop are partitioned among the
	 * worker tasks executing the parallel for loop. Refer to enum {@linkplain
	 * Schedule} for descriptions of the possible schedules.
	 *
	 * @param  chunk  Chunk size (&ge; 1), {@link #STANDARD_CHUNK}, or {@link
	 *                #DEFAULT_CHUNK}.
	 *
	 * @return  This job properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>chunk</TT> is illegal.
	 *
	 * @see  #masterSchedule(Schedule)
	 * @see  #masterSchedule()
	 * @see  #masterChunk()
	 */
	public JobProperties masterChunk
		(int chunk)
		{
		masterLoopProps.chunk (chunk);
		return this;
		}

	/**
	 * Get the <TT>masterChunk</TT> property. The <TT>masterChunk</TT> property,
	 * along with the <TT>masterSchedule</TT> property, specifies how the
	 * iterations of a master-worker parallel for loop are partitioned among the
	 * worker tasks executing the parallel for loop. If the <TT>masterChunk</TT>
	 * property is defaulted, {@link #STANDARD_CHUNK} is returned, indicating
	 * the standard chunk size for the <TT>masterSchedule</TT> property; see
	 * enum {@linkplain Schedule Schedule} for further information.
	 *
	 * @return  Chunk size (&ge; 1), or {@link #STANDARD_CHUNK}.
	 *
	 * @see  #masterSchedule(Schedule)
	 * @see  #masterSchedule()
	 * @see  #masterChunk(int)
	 */
	public int masterChunk()
		{
		return masterLoopProps.chunk();
		}

	/**
	 * Set the <TT>threads</TT> property. The <TT>threads</TT> property
	 * specifies the number of threads that will execute a parallel for loop.
	 *
	 * @param  threads  Number of threads (&ge; 1), {@link
	 *                  #THREADS_EQUALS_CORES}, or {@link #DEFAULT_THREADS}.
	 *
	 * @return  This job properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>threads</TT> is illegal.
	 *
	 * @see  #threads()
	 */
	public JobProperties threads
		(int threads)
		{
		loopProps.threads (threads);
		return this;
		}

	/**
	 * Get the <TT>threads</TT> property. The <TT>threads</TT> property
	 * specifies the number of threads that will execute a parallel for loop. If
	 * the <TT>threads</TT> property is defaulted, {@link #THREADS_EQUALS_CORES}
	 * is returned, indicating that a parallel for loop will be executed by as
	 * many threads as there are cores on the machine.
	 *
	 * @return  Number of threads (&ge; 1), or {@link #THREADS_EQUALS_CORES}.
	 *
	 * @see  #threads(int)
	 */
	public int threads()
		{
		return loopProps.threads();
		}

	/**
	 * Set the <TT>schedule</TT> property. The <TT>schedule</TT> property, along
	 * with the <TT>chunk</TT> property, specifies how the iterations of a
	 * parallel for loop are partitioned among the threads executing the
	 * parallel for loop. Refer to enum {@linkplain Schedule} for descriptions
	 * of the possible schedules.
	 *
	 * @param  schedule  Parallel for loop schedule.
	 *
	 * @return  This job properties object.
	 *
	 * @see  #schedule()
	 * @see  #chunk(int)
	 * @see  #chunk()
	 */
	public JobProperties schedule
		(Schedule schedule)
		{
		loopProps.schedule (schedule);
		return this;
		}

	/**
	 * Get the <TT>schedule</TT> property. The <TT>schedule</TT> property, along
	 * with the <TT>chunk</TT> property, specifies how the iterations of a
	 * parallel for loop are partitioned among the threads executing the
	 * parallel for loop. If the <TT>schedule</TT> property is defaulted, a
	 * {@link Schedule#fixed fixed} schedule is returned.
	 *
	 * @return  Parallel for loop schedule.
	 *
	 * @see  #schedule(Schedule)
	 * @see  #chunk(int)
	 * @see  #chunk()
	 */
	public Schedule schedule()
		{
		return loopProps.schedule();
		}

	/**
	 * Set the <TT>chunk</TT> property. The <TT>chunk</TT> property, along with
	 * the <TT>schedule</TT> property, specifies how the iterations of a
	 * parallel for loop are partitioned among the threads executing the
	 * parallel for loop. Refer to enum {@linkplain Schedule} for descriptions
	 * of the possible schedules.
	 *
	 * @param  chunk  Chunk size (&ge; 1), {@link #STANDARD_CHUNK}, or {@link
	 *                #DEFAULT_CHUNK}.
	 *
	 * @return  This job properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>chunk</TT> is illegal.
	 *
	 * @see  #schedule(Schedule)
	 * @see  #schedule()
	 * @see  #chunk()
	 */
	public JobProperties chunk
		(int chunk)
		{
		loopProps.chunk (chunk);
		return this;
		}

	/**
	 * Get the <TT>chunk</TT> property. The <TT>chunk</TT> property, along with
	 * the <TT>schedule</TT> property, specifies how the iterations of a
	 * parallel for loop are partitioned among the threads executing the
	 * parallel for loop. If the <TT>chunk</TT> property is defaulted, {@link
	 * #STANDARD_CHUNK} is returned, indicating the standard chunk size for the
	 * <TT>schedule</TT> property; see enum {@linkplain Schedule Schedule} for
	 * further information.
	 *
	 * @return  Chunk size (&ge; 1), or {@link #STANDARD_CHUNK}.
	 *
	 * @see  #schedule(Schedule)
	 * @see  #schedule()
	 * @see  #chunk(int)
	 */
	public int chunk()
		{
		return loopProps.chunk();
		}

	/**
	 * Set the <TT>nodeName</TT> property. The <TT>nodeName</TT> property
	 * specifies the name of the cluster node on which a task must run.
	 *
	 * @param  nodeName  Node name, {@link #ANY_NODE_NAME}, or {@link
	 *                   #DEFAULT_NODE_NAME}.
	 *
	 * @return  This job properties object.
	 *
	 * @see  #nodeName()
	 */
	public JobProperties nodeName
		(String nodeName)
		{
		nodeProps.nodeName (nodeName);
		return this;
		}

	/**
	 * Get the <TT>nodeName</TT> property. The <TT>nodeName</TT> property
	 * specifies the name of the cluster node on which a task must run. If the
	 * <TT>nodeName</TT> property is defaulted, {@link #ANY_NODE_NAME} is
	 * returned, indicating that a task can run on any node of the cluster.
	 *
	 * @return  Node name, or {@link #ANY_NODE_NAME}.
	 *
	 * @see  #nodeName(String)
	 */
	public String nodeName()
		{
		return nodeProps.nodeName();
		}

	/**
	 * Set the <TT>cores</TT> property. The <TT>cores</TT> property specifies
	 * the number of CPU cores a task requires.
	 *
	 * @param  cores  Number of cores (&ge; 1), {@link #ALL_CORES}, or {@link
	 *                #DEFAULT_CORES}.
	 *
	 * @return  This job properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>cores</TT> is illegal.
	 *
	 * @see  #cores()
	 */
	public JobProperties cores
		(int cores)
		{
		nodeProps.cores (cores);
		return this;
		}

	/**
	 * Get the <TT>cores</TT> property. The <TT>cores</TT> property specifies
	 * the number of CPU cores a task requires. If the <TT>cores</TT> property
	 * is defaulted, {@link #ALL_CORES} is returned, indicating that a task
	 * requires all the cores on the node.
	 *
	 * @return  Number of cores (&ge; 1), or {@link #ALL_CORES}.
	 *
	 * @see  #cores(int)
	 */
	public int cores()
		{
		return nodeProps.cores();
		}

	/**
	 * Set the <TT>gpus</TT> property. The <TT>gpus</TT> property specifies the
	 * number of GPU accelerators a task requires.
	 *
	 * @param  gpus  Number of GPUs (&ge; 0), or {@link #DEFAULT_GPUS}.
	 *
	 * @return  This job properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>gpus</TT> is illegal.
	 *
	 * @see  #gpus()
	 */
	public JobProperties gpus
		(int gpus)
		{
		nodeProps.gpus (gpus);
		return this;
		}

	/**
	 * Get the <TT>gpus</TT> property. The <TT>gpus</TT> property specifies the
	 * number of GPU accelerators a task requires. If the <TT>gpus</TT> property
	 * is defaulted, 0 is returned, indicating that a task requires no GPU
	 * accelerators.
	 *
	 * @return  Number of GPUs (&ge; 0).
	 *
	 * @see  #gpus(int)
	 */
	public int gpus()
		{
		return nodeProps.gpus();
		}

	/**
	 * Returns a string version of this job properties object.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		return String.format
			("JobProperties(workers=%d,masterSchedule=%s,masterChunk=%d,threads=%d,schedule=%s,chunk=%d,nodeName=\"%s\",cores=%d,gpus=%d)",
			 workers(), masterSchedule(), masterChunk(),
			 threads(), schedule(), chunk(), nodeName(), cores(), gpus());
		}

	}
