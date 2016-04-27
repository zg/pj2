//******************************************************************************
//
// File:    Loop.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.Loop
//
// This Java source file is copyright (C) 2013 by Alan Kaminsky. All rights
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
 * Class Loop is the abstract base class for a loop body executed by a parallel
 * for loop with an integer index.
 * <P>
 * To execute a parallel for loop with an integer index in the {@link
 * Task#main(String[]) main()} method of a {@linkplain Task}, follow this
 * pattern:
 * <PRE>
 * public class MyTask extends Task
 *     {
 *     public void main (String[] args)
 *         {
 *         parallelFor (<I>lb, ub</I>) .exec (new Loop()
 *             {
 *             // <I>Thread-local variable declarations (optional)</I>
 *             public void start()
 *                 {
 *                 // <I>One-time thread-local initialization (optional method)</I>
 *                 }
 *             public void run (int i)
 *                 {
 *                 // <I>Loop body code for iteration i (required method)</I>
 *                 }
 *             public void finish()
 *                 {
 *                 // <I>One-time thread-local finalization (optional method)</I>
 *                 }
 *             });
 *         }
 *     }</PRE>
 * <P>
 * <TT><I>lb</I></TT> is the lower bound of the loop index (inclusive).
 * <TT><I>ub</I></TT> is the upper bound of the loop index (inclusive).
 * These are both of type <TT>int</TT>.
 * <P>
 * <B>Parallel thread team.</B>
 * The parallel for loop is executed by a team of threads. Each thread operates
 * on its own separate copy of the given Loop object. Each thread automatically
 * creates its own copy by calling the {@link #clone() clone()} method on the
 * given Loop object. Thus, any fields declared in the inner Loop subclass are
 * thread-local variables, not shared variables. However, <I>such fields must
 * not be initialized as part of their declarations.</I> Instead, such fields
 * must be initialized in the {@link #start() start()} method. If fields of the
 * inner Loop subclass are initialized as part of their declarations, the
 * cloning will not work properly.
 * <P>
 * Each thread automatically does the following with its own loop object:
 * <UL>
 * <P><LI>
 * Call the {@link LoopBody#start() start()} method, once only. The {@link
 * LoopBody#start() start()} method can contain one-time initialization code.
 * Fields of the inner Loop subclass must be initialized in the {@link
 * LoopBody#start() start()} method. If the {@link LoopBody#start() start()}
 * method is not declared, the default is to do nothing.
 * <P><LI>
 * Call the {@link #run(int) run()} method zero or more times. On each call, a
 * different loop index is passed in. The {@link #run(int) run()} method must
 * contain code to execute the loop body for index <TT>i</TT>.
 * <P>
 * The parallel for loop's index range is partitioned into chunks as determined
 * by the parallel for loop's <TT>schedule</TT> and <TT>chunk</TT> properties.
 * Each thread calls the {@link #run(int) run()} method for the index or indexes
 * in the chunk or chunks assigned to that thread. See enum {@linkplain
 * Schedule} for further information.
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
 * {@link #run(int) run()} method. This causes <I>all</I> the threads in the
 * team to stop calling the {@link #run(int) run()} method (as soon as each
 * thread's current iteration completes). Each thread in the team then proceeds
 * to call the {@link LoopBody#finish() finish()} method as described above.
 * <P>
 * <B>End-of-loop barrier.</B>
 * There is an implicit barrier at the end of the parallel for loop. After
 * returning from the {@link LoopBody#finish() finish()} method, each team
 * thread arrives at the barrier. Once every team thread has arrived at the
 * barrier, the {@link IntParallelForLoop#exec(Loop) exec()} method returns.
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
 *         parallelFor (<I>lb, ub</I>) .exec (new Loop()
 *             {
 *             IntVbl thrCounter;
 *             public void start()
 *                 {
 *                 thrCounter = threadLocal (counter);
 *                 }
 *             public void run (int i)
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
 * {@link edu.rit.pj2.vbl.IntVbl.Sum IntVbl.Sum}. The <TT>counter</TT>'s {@link
 * edu.rit.pj2.vbl.IntVbl#item item} field is an integer. The counter's {@link
 * edu.rit.pj2.vbl.IntVbl.Sum#reduce(Vbl) reduce()} method adds the
 * <TT>item</TT> fields together.
 * <P>
 * <TT>thrCounter</TT> is declared as a field of the inner Loop subclass, and
 * each team thread executing the loop gets its own copy of the loop, so each
 * team thread gets its own separate <TT>thrCounter</TT> field. Note that the
 * <TT>thrCounter</TT> field is <I>not</I> initialized as part of its
 * declaration. When executing the loop, each team thread does the following:
 * <UL>
 * <P><LI>
 * Execute the {@link LoopBody#start() start()} method. Call the {@link
 * LoopBody#threadLocal(Vbl) threadLocal()} method, passing in the global
 * counter. This returns a thread-local copy of the global counter, which
 * becomes the thread's own <TT>thrCounter</TT>.
 * <P><LI>
 * Execute the {@link #run(int) run()} method multiple times. Increment the
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
 * @see  IntParallelForLoop
 * @see  Schedule
 *
 * @author  Alan Kaminsky
 * @version 19-Nov-2013
 */
public abstract class Loop
	extends LoopBody
	{

// Exported constructors.

	/**
	 * Construct a new loop object.
	 */
	public Loop()
		{
		}

// Exported operations.

	/**
	 * Execute the loop body for the given integer index.
	 * <P>
	 * The <TT>run()</TT> method must be overridden in a subclass.
	 *
	 * @param  i  Loop index.
	 *
	 * @exception  Exception
	 *     The <TT>run()</TT> method may throw any exception.
	 */
	public abstract void run
		(int i)
		throws Exception;

	}
