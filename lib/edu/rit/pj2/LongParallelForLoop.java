//******************************************************************************
//
// File:    LongParallelForLoop.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.LongParallelForLoop
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

import edu.rit.numeric.Int96;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Class LongParallelForLoop provides a work sharing parallel for loop executed
 * by multiple threads, with a loop index of type <TT>long</TT>. A long integer
 * parallel for loop is constructed by the {@link Task#parallelFor(long,long)
 * parallelFor(long,long)} method of class {@linkplain Task}.
 * <P>
 * <B>Programming pattern.</B>
 * To execute a parallel for loop in the {@link Task#main(String[]) main()}
 * method of a {@linkplain Task}, follow this pattern:
 * <PRE>
 * public class MyTask extends Task
 *     {
 *     public void main (String[] args)
 *         {
 *         parallelFor (<I>lb, ub</I>) .exec (new LongLoop()
 *             {
 *             // <I>Thread-local variable declarations (optional)</I>
 *             public void start()
 *                 {
 *                 // <I>One-time thread-local initialization (optional method)</I>
 *                 }
 *             public void run (long i)
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
 * <B>Loop index bounds.</B>
 * <TT><I>lb</I></TT> is the lower bound of the loop index (inclusive).
 * <TT><I>ub</I></TT> is the upper bound of the loop index (inclusive). These
 * are both of type <TT>long</TT>. (For a parallel for loop with a loop index of
 * type <TT>int</TT>, see class {@linkplain IntParallelForLoop}.) If <TT>lb</TT>
 * &gt; <TT>ub</TT>, it represents a loop with no iterations. If <TT>lb</TT>
 * &le; <TT>ub</TT>, it represents a loop with one or more iterations.
 * <P>
 * <B>Parallel thread team.</B>
 * The parallel for loop is executed by a team of threads. The number of threads
 * is given by the <TT>threads</TT> property of the enclosing task (see the
 * {@link Task#threads() threads()} method of class {@linkplain Task}). The
 * default is one thread for each core of the machine on which the program is
 * running. The default can be overridden as follows:
 * <PRE>
 *     parallelFor (<I>lb, ub</I>) .threads (<I>threads</I>) .exec (new LongLoop() ...</PRE>
 * <P>
 * <B>Parallel loop schedule.</B>
 * The parallel for loop's iterations are partitioned among the threads as
 * specified by the <TT>schedule</TT> and <TT>chunk</TT> properties of the
 * enclosing task (see the {@link Task#schedule() schedule()} and {@link
 * Task#chunk() chunk()} methods of class {@linkplain Task}). The default is a
 * fixed schedule. The default schedule can be overridden as follows:
 * <PRE>
 *     parallelFor (<I>lb, ub</I>) .schedule (<I>schedule</I>) .exec (new LongLoop() ...</PRE>
 * <P>
 * Both the default schedule and the default chunk size can be overridden as
 * follows:
 * <PRE>
 *     parallelFor (<I>lb, ub</I>) .schedule (<I>schedule</I>) .chunk (<I>chunk</I>) .exec (new LongLoop() ...</PRE>
 * <P>
 * <B>Parallel loop body.</B>
 * The threads execute the methods in the inner {@linkplain LongLoop} class. For
 * further information about how the parallel for loop executes, see class
 * {@linkplain LongLoop}.
 *
 * @see  ParallelStatement
 * @see  ParallelForLoop
 * @see  LongLoop
 *
 * @author  Alan Kaminsky
 * @version 22-Mar-2014
 */
public class LongParallelForLoop
	extends ParallelForLoop
	{

// Hidden data members.

	Int96 blb;
	Int96 bub;
	AtomicReference<Int96> nextlb;
	LongLoop loop;

// Hidden constructors.

	/**
	 * Construct a new long integer parallel for loop.
	 *
	 * @param  task  Task in which the parallel for loop is executing.
	 * @param  lb    Loop index lower bound.
	 * @param  ub    Loop index upper bound.
	 */
	LongParallelForLoop
		(Task task,
		 long lb,
		 long ub)
		{
		super (task);
		blb = Int96.of(lb);
		bub = Int96.of(ub);
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
	public LongParallelForLoop threads
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
	 * @param  schedule  Parallel for loop schedule, or {@link
	 *                   Task#DEFAULT_SCHEDULE}.
	 *
	 * @return  This parallel for loop object.
	 *
	 * @see  #chunk(int)
	 */
	public LongParallelForLoop schedule
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
	 * @param  chunk  Chunk size (&ge; 1), {@link Task#STANDARD_CHUNK}, or
	 *                {@link Task#DEFAULT_CHUNK}.
	 *
	 * @return  This parallel for loop object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>chunk</TT> is illegal.
	 *
	 * @see  #schedule(Schedule)
	 */
	public LongParallelForLoop chunk
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
		(LongLoop loop)
		{
		if (loop == null)
			throw new NullPointerException
				("LongParallelForLoop.exec(): loop is null");
		threads = properties.actualThreads();
		schedule = properties.schedule();
		chunk = schedule.actualChunk (properties.chunk());
		nextlb = new AtomicReference<Int96> (Int96.of(blb));
		this.loop = loop;
		stop = false;
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
		LongLoop loop = rank == 0 ? this.loop : (LongLoop)(this.loop.clone());
		loop.parallelForLoop = this;
		loop.rank = rank;
		loop.reductionMap = reductionMap;

		// Perform one-time initialization.
		loop.start();

		// Execute iterations.
		schedule.iterate (this, loop, rank);

		// Perform one-time finalization.
		loop.finish();

		loop.parallelForLoop = null;
		loop.rank = -1;
		loop.reductionMap = null;
		}

	/**
	 * Execute this parallel for loop's iterations with a fixed schedule.
	 *
	 * @param  loop  Loop body.
	 * @param  rank  Team thread rank.
	 */
	void iterateFixed
		(LongLoop loop,
		 int rank)
		throws Exception
		{
		Int96 thr = Int96.of(threads);
		Int96 chunkSize = Int96.of(bub) .sub(blb) .add(thr) .div(thr)
			.max(Int96.ONE);
		long chunklb = Int96.of(rank) .mul(chunkSize) .add(blb) .longval();
		long chunkub = Int96.of(chunklb) .add(chunkSize) .sub(Int96.ONE)
			.min(bub) .longval();
//System.out.printf ("rank=%d chunklb=%d chunkub=%d%n", rank, chunklb, chunkub);
//System.out.flush();
//long t1 = System.currentTimeMillis();
		for (long i = chunklb; ! stop && i <= chunkub; ++ i)
			loop.run (i);
//long t2 = System.currentTimeMillis();
//System.out.printf ("rank=%d t1=%d t2=%d t2-t1=%d%n", rank, t1, t2, t2-t1);
//System.out.flush();
		}

	/**
	 * Execute this parallel for loop's iterations with a leapfrog schedule.
	 *
	 * @param  loop  Loop body.
	 * @param  rank  Team thread rank.
	 */
	void iterateLeapfrog
		(LongLoop loop,
		 int rank)
		throws Exception
		{
		long chunklb = Int96.of(rank) .add(blb) .longval();
		long chunkub = bub.longval();
		for (long i = chunklb; ! stop && i <= chunkub; i += threads)
			loop.run (i);
		}

	/**
	 * Execute this parallel for loop's iterations with a dynamic schedule.
	 *
	 * @param  loop  Loop body.
	 * @param  rank  Team thread rank.
	 */
	void iterateDynamic
		(LongLoop loop,
		 int rank)
		throws Exception
		{
		Int96 chunkSize = Int96.of(chunk);
		Int96 oldlb, newlb;
		long chunklb, chunkub;
		chunkloop: while (! stop)
			{
			do
				{
				oldlb = nextlb.get();
				if (oldlb.compareTo (bub) > 0) break chunkloop;
				newlb = Int96.of(oldlb) .add(chunkSize);
				}
			while (! nextlb.compareAndSet (oldlb, newlb));
			chunklb = oldlb .longval();
			chunkub = Int96.of(newlb) .sub(Int96.ONE) .min(bub) .longval();
			for (long i = chunklb; ! stop && i <= chunkub; ++ i)
				loop.run (i);
			}
		}

	/**
	 * Execute this parallel for loop's iterations with a proportional schedule.
	 *
	 * @param  loop  Loop body.
	 * @param  rank  Team thread rank.
	 */
	void iterateProportional
		(LongLoop loop,
		 int rank)
		throws Exception
		{
		Int96 cfthr = Int96.of(chunk) .mul(Int96.of(threads));
		Int96 chunkSize = Int96.of(bub) .sub(blb) .add(cfthr) .div(cfthr)
			.max(Int96.ONE);
		Int96 oldlb, newlb;
		long chunklb, chunkub;
		chunkloop: while (! stop)
			{
			do
				{
				oldlb = nextlb.get();
				if (oldlb.compareTo (bub) > 0) break chunkloop;
				newlb = Int96.of(oldlb) .add(chunkSize);
				}
			while (! nextlb.compareAndSet (oldlb, newlb));
			chunklb = oldlb .longval();
			chunkub = Int96.of(newlb) .sub(Int96.ONE) .min(bub) .longval();
			for (long i = chunklb; ! stop && i <= chunkub; ++ i)
				loop.run (i);
			}
		}

	/**
	 * Execute this parallel for loop's iterations with a guided schedule.
	 *
	 * @param  loop  Loop body.
	 * @param  rank  Team thread rank.
	 */
	void iterateGuided
		(LongLoop loop,
		 int rank)
		throws Exception
		{
		Int96 thr = Int96.of(threads);
		Int96 chunkSize = Int96.of(chunk);
		Int96 oldlb, newlb;
		Int96 newSize = Int96.of(0);
		long chunklb, chunkub;
		chunkloop: while (! stop)
			{
			do
				{
				oldlb = nextlb.get();
				if (oldlb.compareTo (bub) > 0) break chunkloop;
				newSize .assign(bub) .sub(oldlb) .add(Int96.ONE)
					.div(Int96.TWO) .div(thr) .max(chunkSize);
				newlb = Int96.of(oldlb) .add(newSize);
				}
			while (! nextlb.compareAndSet (oldlb, newlb));
			chunklb = oldlb .longval();
			chunkub = Int96.of(newlb) .sub(Int96.ONE) .min(bub) .longval();
			for (long i = chunklb; ! stop && i <= chunkub; ++ i)
				loop.run (i);
			}
		}

	}
