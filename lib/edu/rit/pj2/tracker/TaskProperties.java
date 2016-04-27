//******************************************************************************
//
// File:    TaskProperties.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.TaskProperties
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
import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import java.io.IOException;

/**
 * Class TaskProperties encapsulates the properties of a {@linkplain
 * edu.rit.pj2.Task Task}. The following properties may be specified:
 * <UL>
 * <LI><TT>threads</TT> &mdash; The number of threads executing parallel loops
 * in the task.
 * <LI><TT>schedule</TT> &mdash; The schedule for parallel loops in the task.
 * <LI><TT>chunk</TT> &mdash; The chunk size for parallel loops in the task.
 * <LI><TT>nodeName</TT> &mdash; The name of the node on which to run the task.
 * <LI><TT>cores</TT> &mdash; The number of CPU cores needed to run the task.
 * <LI><TT>gpus</TT> &mdash; The number of GPU accelerators needed to run the
 * task.
 * </UL>
 * <P>
 * Class TaskProperties provides a <I>chaining</I> capability. If a certain
 * setting in a task properties object is defaulted, and the task properties
 * object has a chained task properties object, the setting's value will be that
 * of the chained task properties object. If the setting is defaulted in every
 * chained task properties object, the setting's value will be a predefined
 * default value. When a task properties object is serialized by the {@link
 * #writeOut(OutStream) writeOut()} method, the chained task properties objects
 * if any are not serialized; only the ultimate setting values are serialized.
 *
 * @author  Alan Kaminsky
 * @version 30-May-2014
 */
public class TaskProperties
	implements Streamable
	{

// Exported constants.

	/**
	 * Indicates that the <TT>threads</TT> property is defaulted.
	 */
	public static final int DEFAULT_THREADS = LoopProperties.DEFAULT_THREADS;

	/**
	 * Indicates that the <TT>schedule</TT> property is defaulted.
	 */
	public static final Schedule DEFAULT_SCHEDULE =
		LoopProperties.DEFAULT_SCHEDULE;

	/**
	 * Indicates that the <TT>chunk</TT> property is defaulted.
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

	LoopProperties loopProps;
	NodeProperties nodeProps;
	TaskProperties chained;

// Exported constructors.

	/**
	 * Construct a new task properties object. All settings are defaulted.
	 */
	public TaskProperties()
		{
		loopProps = new LoopProperties();
		nodeProps = new NodeProperties();
		}

	/**
	 * Construct a new task properties object with the given settings.
	 *
	 * @param  threads   Number of threads for a parallel for loop, {@link
	 *                   #THREADS_EQUALS_CORES}, or {@link #DEFAULT_THREADS}.
	 * @param  schedule  Schedule for a parallel for loop, or {@link
	 *                   #DEFAULT_SCHEDULE}.
	 * @param  chunk     Chunk size for a parallel for loop, {@link
	 *                   #STANDARD_CHUNK}, or {@link #DEFAULT_CHUNK}.
	 * @param  nodeName  Node name on which the task must execute, {@link
	 *                   #ANY_NODE_NAME}, or {@link #DEFAULT_NODE_NAME}.
	 * @param  cores     Number of CPU cores the task requires, {@link
	 *                   #ALL_CORES}, or {@link #DEFAULT_CORES}.
	 * @param  gpus      Number of GPU accelerators the task requires, {@link
	 *                   #ALL_GPUS}, or {@link #DEFAULT_GPUS}.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>threads</TT>, <TT>chunk</TT>,
	 *     <TT>cores</TT>, or <TT>gpus</TT> is illegal.
	 */
	public TaskProperties
		(int threads,
		 Schedule schedule,
		 int chunk,
		 String nodeName,
		 int cores,
		 int gpus)
		{
		loopProps = new LoopProperties (threads, schedule, chunk);
		nodeProps = new NodeProperties (nodeName, cores, gpus);
		}

	/**
	 * Construct a new task properties object that is a copy of the given task
	 * properties object. The chained task properties objects of <TT>props</TT>,
	 * if any, are not copied; only the ultimate property values are copied.
	 *
	 * @param  props  Task properties.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>props</TT> is null.
	 */
	public TaskProperties
		(TaskProperties props)
		{
		loopProps = new LoopProperties
			(props.threads(), props.schedule(), props.chunk());
		nodeProps = new NodeProperties
			(props.nodeName(), props.cores(), props.gpus());
		}

	/**
	 * Construct a new task properties object with the given loop and node
	 * properties objects.
	 *
	 * @param  loopProps  Loop properties object.
	 * @param  nodeProps  Node properties object.
	 */
	private TaskProperties
		(LoopProperties loopProps,
		 NodeProperties nodeProps)
		{
		this.loopProps = loopProps;
		this.nodeProps = nodeProps;
		}

// Exported operations.

	/**
	 * Add the given task properties object to the end of this task properties
	 * object's chain.
	 *
	 * @param  props  Task properties object to be chained.
	 *
	 * @return  This task properties object.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>props</TT> is null.
	 */
	public TaskProperties chain
		(TaskProperties props)
		{
		if (props == null)
			throw new NullPointerException
				("TaskProperties.chain(): props is null");
		TaskProperties p = this;
		TaskProperties q = this.chained;
		while (q != null)
			{
			p = q;
			q = q.chained;
			}
		p.chained = new TaskProperties (props.loopProps, props.nodeProps);
		return this;
		}

	/**
	 * Add the given job properties object to the end of this task properties
	 * object's chain.
	 *
	 * @param  props  Job properties object to be chained.
	 *
	 * @return  This task properties object.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>props</TT> is null.
	 */
	public TaskProperties chain
		(JobProperties props)
		{
		if (props == null)
			throw new NullPointerException
				("TaskProperties.chain(): props is null");
		TaskProperties p = this;
		TaskProperties q = this.chained;
		while (q != null)
			{
			p = q;
			q = q.chained;
			}
		p.chained = new TaskProperties (props.loopProps, props.nodeProps);
		return this;
		}

	/**
	 * Set the <TT>threads</TT> property. The <TT>threads</TT> property
	 * specifies the number of threads that will execute a parallel for loop.
	 *
	 * @param  threads  Number of threads (&ge; 1), {@link
	 *                  #THREADS_EQUALS_CORES}, or {@link #DEFAULT_THREADS}.
	 *
	 * @return  This task properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>threads</TT> is illegal.
	 *
	 * @see  #threads()
	 * @see  #actualThreads()
	 */
	public TaskProperties threads
		(int threads)
		{
		loopProps.threads (threads);
		return this;
		}

	/**
	 * Get the <TT>threads</TT> property. The <TT>threads</TT> property
	 * specifies the number of threads that will execute a parallel for loop. If
	 * the <TT>threads</TT> property is defaulted or is {@link
	 * #THREADS_EQUALS_CORES}, and the <TT>cores</TT> property is set to a
	 * specific number of cores (&ge; 1), then the <TT>cores</TT> property
	 * setting is returned. Otherwise, if the <TT>threads</TT> property is
	 * defaulted, {@link #THREADS_EQUALS_CORES} is returned, indicating that a
	 * parallel for loop will be executed by as many threads as there are cores
	 * on the machine. Otherwise, the <TT>threads</TT> property setting is
	 * returned.
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
		TaskProperties p = this;
		int nt = p.loopProps.threads;
		while (nt == DEFAULT_THREADS && p.chained != null)
			{
			p = p.chained;
			nt = p.loopProps.threads;
			}
		int nc = cores();
		return 
			nc >= 1 && nt <= THREADS_EQUALS_CORES ?
				nc :
			nt == DEFAULT_THREADS ?
				THREADS_EQUALS_CORES :
				nt;
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
		int rv = threads();
		return
			rv == THREADS_EQUALS_CORES ?
				Runtime.getRuntime().availableProcessors() :
				rv;
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
	 * @return  This task properties object.
	 *
	 * @see  #schedule()
	 * @see  #chunk(int)
	 * @see  #chunk()
	 */
	public TaskProperties schedule
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
		TaskProperties p = this;
		Schedule rv = p.loopProps.schedule;
		while (rv == DEFAULT_SCHEDULE && p.chained != null)
			{
			p = p.chained;
			rv = p.loopProps.schedule;
			}
		return rv == DEFAULT_SCHEDULE ? Schedule.fixed : rv;
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
	 * @return  This task properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>chunk</TT> is illegal.
	 *
	 * @see  #schedule(Schedule)
	 * @see  #schedule()
	 * @see  #chunk()
	 */
	public TaskProperties chunk
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
		TaskProperties p = this;
		int rv = p.loopProps.chunk;
		while (rv == DEFAULT_CHUNK && p.chained != null)
			{
			p = p.chained;
			rv = p.loopProps.chunk;
			}
		return rv == DEFAULT_CHUNK ? STANDARD_CHUNK : rv;
		}

	/**
	 * Set the <TT>nodeName</TT> property. The <TT>nodeName</TT> property
	 * specifies the name of the cluster node on which the task must run.
	 *
	 * @param  nodeName  Node name, {@link #ANY_NODE_NAME}, or {@link
	 *                   #DEFAULT_NODE_NAME}.
	 *
	 * @return  This task properties object.
	 *
	 * @see  #nodeName()
	 */
	public TaskProperties nodeName
		(String nodeName)
		{
		nodeProps.nodeName (nodeName);
		return this;
		}

	/**
	 * Get the <TT>nodeName</TT> property. The <TT>nodeName</TT> property
	 * specifies the name of the cluster node on which the task must run. If the
	 * <TT>nodeName</TT> property is defaulted, {@link #ANY_NODE_NAME} is
	 * returned, indicating that the task can run on any node of the cluster.
	 *
	 * @return  Node name, or {@link #ANY_NODE_NAME}.
	 *
	 * @see  #nodeName(String)
	 */
	public String nodeName()
		{
		TaskProperties p = this;
		String rv = p.nodeProps.nodeName;
		while (rv == DEFAULT_NODE_NAME && p.chained != null)
			{
			p = p.chained;
			rv = p.nodeProps.nodeName;
			}
		return rv == DEFAULT_NODE_NAME ? ANY_NODE_NAME : rv;
		}

	/**
	 * Set the <TT>cores</TT> property. The <TT>cores</TT> property specifies
	 * the number of CPU cores the task requires.
	 *
	 * @param  cores  Number of cores (&ge; 1), {@link #ALL_CORES}, or {@link
	 *                #DEFAULT_CORES}.
	 *
	 * @return  This task properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>cores</TT> is illegal.
	 *
	 * @see  #cores()
	 */
	public TaskProperties cores
		(int cores)
		{
		nodeProps.cores (cores);
		return this;
		}

	/**
	 * Get the <TT>cores</TT> property. The <TT>cores</TT> property specifies
	 * the number of CPU cores the task requires. If the <TT>cores</TT> property
	 * is defaulted, {@link #ALL_CORES} is returned, indicating that the task
	 * requires all the cores on the node.
	 *
	 * @return  Number of cores (&ge; 1), or {@link #ALL_CORES}.
	 *
	 * @see  #cores(int)
	 */
	public int cores()
		{
		TaskProperties p = this;
		int rv = p.nodeProps.cores;
		while (rv == DEFAULT_CORES && p.chained != null)
			{
			p = p.chained;
			rv = p.nodeProps.cores;
			}
		return rv == DEFAULT_CORES ? ALL_CORES : rv;
		}

	/**
	 * Set the <TT>gpus</TT> property. The <TT>gpus</TT> property specifies the
	 * number of GPU accelerators the task requires.
	 *
	 * @param  gpus  Number of GPUs (&ge; 0), or {@link #DEFAULT_GPUS}.
	 *
	 * @return  This task properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>gpus</TT> is illegal.
	 *
	 * @see  #gpus()
	 */
	public TaskProperties gpus
		(int gpus)
		{
		nodeProps.gpus (gpus);
		return this;
		}

	/**
	 * Get the <TT>gpus</TT> property. The <TT>gpus</TT> property specifies the
	 * number of GPU accelerators the task requires. If the <TT>gpus</TT>
	 * property is defaulted, 0 is returned, indicating that the task requires
	 * no GPU accelerators.
	 *
	 * @return  Number of GPUs (&ge; 0).
	 *
	 * @see  #gpus(int)
	 */
	public int gpus()
		{
		TaskProperties p = this;
		int rv = p.nodeProps.gpus;
		while (rv == DEFAULT_GPUS && p.chained != null)
			{
			p = p.chained;
			rv = p.nodeProps.gpus;
			}
		return rv == DEFAULT_GPUS ? 0 : rv;
		}

	/**
	 * Get the node properties from this task properties object.
	 *
	 * @return  Node properties object.
	 */
	public NodeProperties nodeProperties()
		{
		return new NodeProperties (nodeName(), cores(), gpus());
		}

	/**
	 * Write this object's fields to the given out stream.
	 *
	 * @param  out  Out stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeOut
		(OutStream out)
		throws IOException
		{
		out.writeInt (threads());
		Schedule.writeOut (schedule(), out);
		out.writeInt (chunk());
		out.writeString (nodeName());
		out.writeInt (cores());
		out.writeInt (gpus());
		}

	/**
	 * Read this object's fields from the given in stream.
	 *
	 * @param  in  In stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readIn
		(InStream in)
		throws IOException
		{
		in.readFields (loopProps);
		in.readFields (nodeProps);
		chained = null;
		}

	/**
	 * Returns a string version of this task properties object.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		return String.format
			("TaskProperties(threads=%d,schedule=%s,chunk=%d,nodeName=\"%s\",cores=%d,gpus=%d)",
			 threads(), schedule(), chunk(), nodeName(), cores(), gpus());
		}

	}
