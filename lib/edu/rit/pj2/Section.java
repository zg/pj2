//******************************************************************************
//
// File:    Section.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.Section
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

import java.util.Iterator;

/**
 * Class Section is the abstract base class for a section of code executed in
 * parallel with other sections.
 * <P>
 * <B>Programming pattern.</B>
 * To execute a group of different parallel sections in the {@link
 * Task#main(String[]) main()} method of a {@linkplain Task}, follow this
 * pattern:
 * <PRE>
 * public class MyTask extends Task
 *     {
 *     public void main (String[] args)
 *         {
 *         parallelDo (new Section()
 *             {
 *             public void run()
 *                 {
 *                 // <I>Code for first section</I>
 *                 }
 *             },
 *         new Section()
 *             {
 *             public void run()
 *                 {
 *                 // <I>Code for second section</I>
 *                 }
 *             });
 *         }
 *     }</PRE>
 * <P>
 * There can be any number of sections in the {@link Task#parallelDo(Section[])
 * parallelDo()} method call. There must be at least one section.
 * <P>
 * To execute a number of copies of the same parallel section in the {@link
 * Task#main(String[]) main()} method of a {@linkplain Task}, follow this
 * pattern:
 * <PRE>
 * public class MyTask extends Task
 *     {
 *     public void main (String[] args)
 *         {
 *         parallelDo (3, new Section()
 *             {
 *             public void run()
 *                 {
 *                 // <I>Code for section</I>
 *                 }
 *             });
 *         }
 *     }</PRE>
 * <P>
 * In the above example, three copies of the section are executed in parallel.
 * <P>
 * <B>Parallel thread team.</B>
 * The parallel section group is executed by a team of threads. There are as
 * many threads as there are sections. Each thread calls the {@link #run()
 * run()} method on a different Section object.
 * <P>
 * <B>End-of-sections barrier.</B>
 * There is an implicit barrier at the end of the parallel section group. After
 * returning from the {@link #run() run()} method, each team thread arrives at
 * the barrier. Once every team thread has arrived at the barrier, the {@link
 * Task#parallelDo(Section[]) parallelDo()} method returns.
 * <P>
 * <B>Reduction.</B>
 * To execute a parallel section group with reduction, follow this pattern:
 * <PRE>
 * public class MyTask extends Task
 *     {
 *     IntVbl counter = new IntVbl.Sum();
 *
 *     public void main (String[] args)
 *         {
 *         parallelDo (new Section()
 *             {
 *             public void run()
 *                 {
 *                 IntVbl thrCounter = threadLocal (counter);
 *                 thrCounter.item = 42; // or whatever
 *                 }
 *             },
 *         new Section()
 *             {
 *             public void run()
 *                 {
 *                 IntVbl thrCounter = threadLocal (counter);
 *                 thrCounter.item = 24; // or whatever
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
 * Each thread, upon calling its section's {@link #run() run()} method, calls
 * the {@link #threadLocal(Vbl) threadLocal()} method, passing in the global
 * counter. This returns a thread-local copy of the global counter, which
 * becomes the thread's own <TT>thrCounter</TT>. Each thread sets its own
 * <TT>thrCounter</TT>'s item to a certain value. Note that no synchronization
 * is required, as each thread is updating its own <TT>thrCounter</TT>.
 * <P>
 * When the parallel section group finishes executing, all the threads' local
 * variables are automatically reduced together in a multiple thread safe
 * fashion. The reduction is performed by the {@link
 * edu.rit.pj2.vbl.IntVbl.Sum#reduce(Vbl) reduce()} method, so all the threads'
 * local counters' {@link edu.rit.pj2.vbl.IntVbl#item item} fields are added
 * together. The result (66 in this example) is stored back in the global
 * variable's {@link edu.rit.pj2.vbl.IntVbl#item item} field, which is then
 * printed.
 *
 * @see  ParallelStatement
 *
 * @author  Alan Kaminsky
 * @version 18-May-2015
 */
public abstract class Section
	implements Cloneable
	{

// Hidden data members.

	Task task;
	int threads;
	int rank;
	ReductionMap reductionMap;

// Exported constructors.

	/**
	 * Construct a new section object.
	 */
	public Section()
		{
		}

// Exported operations.

	/**
	 * Execute this section.
	 *
	 * @exception  Exception
	 *     The <TT>run()</TT> method may throw any exception.
	 */
	public abstract void run()
		throws Exception;

	/**
	 * Execute a work sharing parallel for loop with a loop index of type
	 * <TT>int</TT>. The loop index goes from the given lower bound to the given
	 * upper bound. For further information, see classes {@linkplain
	 * ParallelForLoop} and {@linkplain IntParallelForLoop}.
	 * <P>
	 * The <TT>parallelFor()</TT> method lets a parallel for loop be nested
	 * inside a parallel section. The <TT>parallelFor()</TT> method may only be
	 * called from the section's <TT>run()</TT> method.
	 *
	 * @param  lb  Loop index lower bound (inclusive).
	 * @param  ub  Loop index upper bound (inclusive).
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if the <TT>parallelFor()</TT> method is
	 *     not called from the <TT>run()</TT> method.
	 */
	public IntParallelForLoop parallelFor
		(int lb,
		 int ub)
		{
		if (task == null)
			throw new IllegalStateException();
		return new IntParallelForLoop (task, lb, ub);
		}

	/**
	 * Execute a work sharing parallel for loop with a loop index of type
	 * <TT>long</TT>. The loop index goes from the given lower bound to the
	 * given upper bound. For further information, see classes {@linkplain
	 * ParallelForLoop} and {@linkplain LongParallelForLoop}.
	 * <P>
	 * The <TT>parallelFor()</TT> method lets a parallel for loop be nested
	 * inside a parallel section. The <TT>parallelFor()</TT> method may only be
	 * called from the section's <TT>run()</TT> method.
	 *
	 * @param  lb  Loop index lower bound (inclusive).
	 * @param  ub  Loop index upper bound (inclusive).
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if the <TT>parallelFor()</TT> method is
	 *     not called from the <TT>run()</TT> method.
	 */
	public LongParallelForLoop parallelFor
		(long lb,
		 long ub)
		{
		if (task == null)
			throw new IllegalStateException();
		return new LongParallelForLoop (task, lb, ub);
		}

	/**
	 * Execute a work sharing parallel for loop over the work items in a
	 * {@linkplain WorkQueue}. For further information, see classes {@linkplain
	 * ParallelForLoop} and {@linkplain ObjectParallelForLoop}.
	 *
	 * @param  <W>    Data type of the work items.
	 * @param  queue  Work queue.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if the <TT>parallelFor()</TT> method is
	 *     not called from the <TT>run()</TT> method.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>queue</TT> is null.
	 */
	public <W> ObjectParallelForLoop<W> parallelFor
		(WorkQueue<W> queue)
		{
		if (task == null)
			throw new IllegalStateException();
		return new ObjectParallelForLoop<W> (task, queue);
		}

	/**
	 * Get a thread-local copy of the given global shared variable. The global
	 * variable and the local variable are instances of a subclass of class
	 * {@linkplain Vbl}. The section can update the local variable without
	 * needing to synchronize with other threads. When the parallel section
	 * group finishes execution, all the local copies are automatically reduced
	 * together using the {@linkplain Vbl} subclass's {@link Vbl#reduce(Vbl)
	 * reduce()} method, and the result is stored into the global variable
	 * (overwriting whatever was there).
	 * <P>
	 * The <TT>threadLocal()</TT> method may only be called from the
	 * <TT>run()</TT> method.
	 *
	 * @param  global  Global shared variable.
	 *
	 * @return  Thread-local copy of <TT>global</TT>. Typically, it is cast to
	 *          the data type of <TT>global</TT>.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if the <TT>threadLocal()</TT> method is
	 *     not called from the <TT>run()</TT> method.
	 */
	public <V extends Vbl> V threadLocal
		(V global)
		{
		if (reductionMap == null)
			throw new IllegalStateException();
		return (V) reductionMap.add (global);
		}

	/**
	 * Returns the number of threads in the thread team executing this parallel
	 * section group.
	 * <P>
	 * The <TT>threads()</TT> method may only be called from the <TT>run()</TT>
	 * method.
	 *
	 * @return  Number of threads.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if the <TT>threads()</TT> method is not
	 *     called from the <TT>run()</TT> method.
	 */
	public int threads()
		{
		if (threads == -1)
			throw new IllegalStateException();
		return threads;
		}

	/**
	 * Returns the rank of the thread executing this section.
	 * <P>
	 * The <TT>rank()</TT> method may only be called from the <TT>run()</TT>
	 * method.
	 *
	 * @return  Rank in the range 0 .. {@link #threads() threads()}&minus;1.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if the <TT>rank()</TT> method is not
	 *     called from the <TT>run()</TT> method.
	 */
	public int rank()
		{
		if (rank == -1)
			throw new IllegalStateException();
		return rank;
		}

	/**
	 * Create a clone of this section object.
	 *
	 * @return  The cloned object.
	 */
	public Object clone()
		{
		try
			{
			return super.clone();
			}
		catch (CloneNotSupportedException exc)
			{
			throw new TerminateException ("Shouldn't happen", exc);
			}
		}

	}
