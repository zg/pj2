//******************************************************************************
//
// File:    ObjectLoop.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.ObjectLoop
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
 * Class ObjectLoop is the abstract base class for a loop body executed by a
 * parallel for loop over the work items in a {@linkplain WorkQueue}.
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
 * Each thread operates on its own separate copy of the given ObjectLoop object.
 * Each thread automatically creates its own copy by calling the {@link #clone()
 * clone()} method on the given ObjectLoop object. Thus, any fields declared in
 * the inner ObjectLoop subclass are thread-local variables, not shared
 * variables. However, <I>such fields must not be initialized as part of their
 * declarations.</I> Instead, such fields must be initialized in the {@link
 * #start() start()} method. If fields of the inner ObjectLoop subclass are
 * initialized as part of their declarations, the cloning will not work
 * properly.
 * <P>
 * Each thread automatically does the following with its own loop object:
 * <UL>
 * <P><LI>
 * Call the {@link LoopBody#start() start()} method, once only. The {@link
 * LoopBody#start() start()} method can contain one-time initialization code.
 * Fields of the inner ObjectLoop subclass must be initialized in the {@link
 * LoopBody#start() start()} method. If the {@link LoopBody#start() start()}
 * method is not declared, the default is to do nothing.
 * <P><LI>
 * Call the {@link #run(Object) run()} method zero or more times. On each call,
 * a different work item from the work queue is passed in. The {@link
 * #run(Object) run()} method must contain code to execute the loop body for the
 * given <TT>workitem</TT>.
 * <P>
 * For an object parallel for loop, the <TT>schedule</TT> and <TT>chunk</TT>
 * properties are not used. Each thread repeatedly gets the next work item from
 * the work queue and calls the {@link #run(Object) run()} method with that
 * work item. The order in which the threads process the work items is not
 * specified.
 * <P><LI>
 * Call the {@link LoopBody#finish() finish()} method, once only. The {@link
 * LoopBody#finish() finish()} method can contain one-time finalization code. If
 * the {@link LoopBody#finish() finish()} method is not declared, the default is
 * to do nothing.
 * </UL>
 * <P>
 * <B>Early loop exit.</B>
 * To terminate the parallel for loop before all the loop iterations have been
 * executed, call the {@link LoopBody#stop() stop()} method from inside the
 * {@link #run(Object) run()} method. This causes <I>all</I> the threads in the
 * team to stop calling the {@link #run(Object) run()} method (as soon as each
 * thread's current iteration completes). Each thread in the team then proceeds
 * to call the {@link LoopBody#finish() finish()} method as described above.
 * <P>
 * <B>End-of-loop barrier.</B>
 * There is an implicit barrier at the end of the parallel for loop. After
 * returning from the {@link LoopBody#finish() finish()} method, each team
 * thread arrives at the barrier. Once every team thread has arrived at the
 * barrier, the {@link ObjectParallelForLoop#exec(ObjectLoop) exec()} method
 * returns.
 * <P>
 * <B>Reduction.</B>
 * To execute a parallel for loop with reduction, follow this pattern:
 * <PRE>
 * public class MyTask extends Task
 *     {
 *     IntVbl counter = new IntVbl.Sum();
 *
 *     public void main (String[] args)
 *         {
 *         WorkQueue&lt;W&gt; queue = new WorkQueue&lt;W&gt;();
 *         parallelFor (queue) .exec (new ObjectLoop&lt;W&gt;()
 *             {
 *             IntVbl thrCounter;
 *             public void start()
 *                 {
 *                 thrCounter = threadLocal (counter);
 *                 }
 *             public void run (W obj)
 *                 {
 *                 if (. . .)
 *                     ++ thrCounter.item;
 *                 }
 *             });
 *         System.out.printf ("Counter = %d%n", counter.item);
 *         }
 *     }</PRE>
 * <P>
 * In this example, <TT>counter</TT> is declared as a field of the {@link Task}
 * subclass, and so is a global shared variable. It is an instance of class
 * {@link edu.rit.pj2.vbl.IntVbl IntVbl.Sum}. The <TT>counter</TT>'s {@link
 * edu.rit.pj2.vbl.IntVbl#item item} field is an integer. The counter's {@link
 * edu.rit.pj2.vbl.IntVbl.Sum#reduce(Vbl) reduce()} method adds the
 * <TT>item</TT> fields together.
 * <P>
 * <TT>thrCounter</TT> is declared as a field of the inner ObjectLoop subclass,
 * and each team thread executing the loop gets its own copy of the loop, so
 * each team thread gets its own separate <TT>thrCounter</TT> field. Note that
 * the <TT>thrCounter</TT> field is <I>not</I> initialized as part of its
 * declaration. When executing the loop, each team thread does the following:
 * <UL>
 * <P><LI>
 * Execute the {@link LoopBody#start() start()} method. Call the {@link
 * LoopBody#threadLocal(Vbl) threadLocal()} method, passing in the global
 * counter. This returns a thread-local copy of the global counter, which
 * becomes the thread's own <TT>thrCounter</TT>.
 * <P><LI>
 * Execute the {@link #run(Object) run()} method multiple times. Increment the
 * thread's own <TT>thrCounter</TT>'s {@link edu.rit.pj2.vbl.IntVbl#item item}
 * field some number of times. Note that no synchronization is required, as each
 * thread is updating its own <TT>thrCounter</TT>.
 * </UL>
 * <P>
 * When the loop finishes executing, all the threads' local variables are
 * automatically reduced together in a multiple thread safe fashion. The
 * reduction is performed by the {@link edu.rit.pj2.vbl.IntVbl.Sum#reduce(Vbl)
 * reduce()} method, so all the threads' local counters' {@link
 * edu.rit.pj2.vbl.IntVbl#item item} fields are added together. The result is
 * stored back in the global variable's {@link edu.rit.pj2.vbl.IntVbl#item item}
 * field, which is then printed.
 *
 * @see  ParallelStatement
 * @see  ParallelForLoop
 * @see  ObjectParallelForLoop
 * @see  Schedule
 *
 * @param  <W>  Data type of the work items.
 *
 * @author  Alan Kaminsky
 * @version 18-May-2015
 */
public abstract class ObjectLoop<W>
	extends LoopBody
	{

// Hidden data members.

	WorkQueue<W> queue;

// Exported constructors.

	/**
	 * Construct a new loop object.
	 */
	public ObjectLoop()
		{
		}

// Exported operations.

	/**
	 * Execute the loop body for the given work item.
	 * <P>
	 * The <TT>run()</TT> method must be overridden in a subclass.
	 *
	 * @param  workitem  Work item.
	 *
	 * @exception  Exception
	 *     The <TT>run()</TT> method may throw any exception.
	 */
	public abstract void run
		(W workitem)
		throws Exception;

	/**
	 * Get the work queue from which this loop is obtaining work items.
	 *
	 * @return  Work queue.
	 */
	public WorkQueue<W> queue()
		{
		return queue;
		}

	}
