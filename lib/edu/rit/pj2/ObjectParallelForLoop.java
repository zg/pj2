//******************************************************************************
//
// File:    ObjectParallelForLoop.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.ObjectParallelForLoop
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

/**
 * Class ObjectParallelForLoop provides a work sharing parallel for loop
 * executed by multiple threads, looping over the work items in a {@linkplain
 * WorkQueue}. An object parallel for loop is constructed by the {@link
 * Task#parallelFor(WorkQueue) parallelFor(WorkQueue)} method of class
 * {@linkplain Task}.
 * <P>
 * <B>Programming pattern.</B>
 * To execute a parallel for loop over a series of work items in the {@link
 * Task#main(String[]) main()} method of a {@linkplain Task}, follow this
 * pattern:
 * <PRE>
 * public class MyTask extends Task
 *     {
 *     public void main (String[] args)
 *         {
 *         WorkQueue&lt;W&gt; queue = new WorkQueue&lt;W&gt;();
 *         parallelFor (queue) .exec (new ObjectLoop&lt;W&gt;()
 *             {
 *             // <I>Thread-local variable declarations (optional)</I>
 *             public void start()
 *                 {
 *                 // <I>One-time thread-local initialization (optional method)</I>
 *                 }
 *             public void run (W workitem)
 *                 {
 *                 // <I>Loop body code for workitem (required method)</I>
 *                 }
 *             public void finish()
 *                 {
 *                 // <I>One-time thread-local finalization (optional method)</I>
 *                 }
 *             });
 *         }
 *     }</PRE>
 * <P>
 * <TT>W</TT> is the data type of the work items. <TT>queue</TT> is a
 * {@linkplain WorkQueue} of work items to be processed in parallel. Call the
 * work queue's {@link WorkQueue#add(Object) add()} method to add a work item to
 * the work queue. Work items may be added to the work queue at any time by any
 * thread; for example: by the task main thread before starting the parallel for
 * loop; by a separate thread running concurrently with the parallel for loop;
 * or in the parallel for loop body itself.
 * <P>
 * <B>Parallel thread team.</B>
 * The parallel for loop is executed by a team of threads. The number of threads
 * is given by the <TT>threads</TT> property of the enclosing task (see the
 * {@link Task#threads() threads()} method of class {@linkplain Task}). The
 * default is one thread for each core of the machine on which the program is
 * running. The default can be overridden as follows:
 * <PRE>
 *     parallelFor (queue) .threads (<I>threads</I>) .exec (new ObjectLoop&lt;W&gt;() ...</PRE>
 * <P>
 * <B>Parallel loop body.</B>
 * The threads execute the methods in the inner {@linkplain ObjectLoop} class.
 * For further information about how the parallel for loop executes, see class
 * {@linkplain ObjectLoop}.
 *
 * @param  <W>  Data type of the work items.
 *
 * @see  ParallelStatement
 * @see  ParallelForLoop
 * @see  ObjectLoop
 *
 * @author  Alan Kaminsky
 * @version 18-May-2015
 */
public class ObjectParallelForLoop<W>
	extends ParallelForLoop
	{

// Hidden data members.

	private WorkQueue<W> queue;
	private ObjectLoop<W> loop;

// Hidden constructors.

	/**
	 * Construct a new object parallel for loop.
	 *
	 * @param  task   Task in which the parallel for loop is executing.
	 * @param  queue  Work queue.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>queue</TT> is null.
	 */
	ObjectParallelForLoop
		(Task task,
		 WorkQueue<W> queue)
		{
		super (task);
		if (queue == null)
			throw new NullPointerException
				("ObjectParallelForLoop(): queue is null");
		this.queue = queue;
		}

// Exported operations.

	/**
	 * Set this parallel for loop's <TT>threads</TT> property. The
	 * <TT>threads</TT> property specifies the number of threads that will
	 * execute this parallel for loop. The default is the <TT>threads</TT>
	 * property of the enclosing task. For further information, see the {@link
	 * Task#threads(int) threads()} method of class {@linkplain Task}.
	 *
	 * @param  threads  Number of threads (&ge; 1), {@link
	 *                  Task#THREADS_EQUALS_CORES}, or {@link
	 *                  Task#DEFAULT_THREADS}.
	 *
	 * @return  This parallel for loop object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>threads</TT> is illegal.
	 */
	public ObjectParallelForLoop<W> threads
		(int threads)
		{
		properties.threads (threads);
		return this;
		}

	/**
	 * Execute this parallel for loop with the loop body specified by the given
	 * {@linkplain ObjectLoop} object.
	 *
	 * @param  loop  Loop object.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>loop</TT> is null.
	 */
	public void exec
		(ObjectLoop<W> loop)
		{
		if (loop == null)
			throw new NullPointerException
				("ObjectParallelForLoop.exec(): loop is null");
		threads = properties.actualThreads();
		this.loop = loop;
		stop = false;
		queue.threads (threads);
		Team.execute (threads, this);
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
		// Thread 0 operates on the original loop object, the other threads
		// operate on their own copies of the loop object.
		ObjectLoop<W> loop =
			rank == 0 ? this.loop : (ObjectLoop<W>)(this.loop.clone());
		loop.parallelForLoop = this;
		loop.rank = rank;
		loop.reductionMap = reductionMap;
		loop.queue = queue;

		// Perform one-time initialization.
		loop.start();

		// Execute iterations.
		W workitem = null;
		while (! stop && (workitem = queue.remove()) != null)
			loop.run (workitem);

		// Perform one-time finalization.
		loop.finish();

		loop.parallelForLoop = null;
		loop.rank = -1;
		loop.reductionMap = null;
		loop.queue = null;
		}

	}
