//******************************************************************************
//
// File:    Schedule.java
// Package: edu.rit.pj2
// Unit:    Enum edu.rit.pj2.Schedule
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

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import java.io.IOException;
import java.io.InvalidObjectException;

/**
 * Enum Schedule specifies the schedule for partitioning the iterations of a
 * work sharing parallel for loop among multiple threads.
 *
 * @see  ParallelForLoop
 * @see  IntParallelForLoop
 * @see  LongParallelForLoop
 * @see  Loop
 * @see  LongLoop
 *
 * @author  Alan Kaminsky
 * @version 22-Mar-2014
 */
public enum Schedule
	{

// Exported enumerals.

	/**
	 * Fixed schedule. The iterations are partitioned into as many chunks as
	 * there are threads. Each chunk is the same size (except possibly the last
	 * chunk). Each thread performs a different chunk of iterations. The
	 * <TT>chunk</TT> property is not used.
	 * <P>
	 * A fixed schedule is appropriate when each loop iteration takes the same
	 * amount of time, so load balancing is not needed; and when each thread
	 * should do a contiguous range of loop indexes.
	 */
	fixed (new Scheduler()
		{
		public int actualChunk (int chunk)
			{
			return chunk == Task.STANDARD_CHUNK ? 1 : chunk;
			}
		public void iterate (IntParallelForLoop pfl, Loop loop, int rank)
			throws Exception
			{
			pfl.iterateFixed (loop, rank);
			}
		public void iterate (LongParallelForLoop pfl, LongLoop loop, int rank)
			throws Exception
			{
			pfl.iterateFixed (loop, rank);
			}
		public void putChunks (Job job, int workers, int masterChunk,
				int lb, int ub)
			{
			job.putChunksFixed (workers, masterChunk, lb, ub);
			}
		public void putChunks (Job job, int workers, int masterChunk,
				long lb, long ub)
			{
			job.putChunksFixed (workers, masterChunk, lb, ub);
			}
		}),

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
	 */
	leapfrog (new Scheduler()
		{
		public int actualChunk (int chunk)
			{
			return chunk == Task.STANDARD_CHUNK ? 1 : chunk;
			}
		public void iterate (IntParallelForLoop pfl, Loop loop, int rank)
			throws Exception
			{
			pfl.iterateLeapfrog (loop, rank);
			}
		public void iterate (LongParallelForLoop pfl, LongLoop loop, int rank)
			throws Exception
			{
			pfl.iterateLeapfrog (loop, rank);
			}
		public void putChunks (Job job, int workers, int masterChunk,
				int lb, int ub)
			{
			job.putChunksLeapfrog (workers, masterChunk, lb, ub);
			}
		public void putChunks (Job job, int workers, int masterChunk,
				long lb, long ub)
			{
			job.putChunksLeapfrog (workers, masterChunk, lb, ub);
			}
		}),

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
	 */
	dynamic (new Scheduler()
		{
		public int actualChunk (int chunk)
			{
			return chunk == Task.STANDARD_CHUNK ? 1 : chunk;
			}
		public void iterate (IntParallelForLoop pfl, Loop loop, int rank)
			throws Exception
			{
			pfl.iterateDynamic (loop, rank);
			}
		public void iterate (LongParallelForLoop pfl, LongLoop loop, int rank)
			throws Exception
			{
			pfl.iterateDynamic (loop, rank);
			}
		public void putChunks (Job job, int workers, int masterChunk,
				int lb, int ub)
			{
			job.putChunksDynamic (workers, masterChunk, lb, ub);
			}
		public void putChunks (Job job, int workers, int masterChunk,
				long lb, long ub)
			{
			job.putChunksDynamic (workers, masterChunk, lb, ub);
			}
		}),

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
	 */
	proportional (new Scheduler()
		{
		public int actualChunk (int chunk)
			{
			return chunk == Task.STANDARD_CHUNK ? 10 : chunk;
			}
		public void iterate (IntParallelForLoop pfl, Loop loop, int rank)
			throws Exception
			{
			pfl.iterateProportional (loop, rank);
			}
		public void iterate (LongParallelForLoop pfl, LongLoop loop, int rank)
			throws Exception
			{
			pfl.iterateProportional (loop, rank);
			}
		public void putChunks (Job job, int workers, int masterChunk,
				int lb, int ub)
			{
			job.putChunksProportional (workers, masterChunk, lb, ub);
			}
		public void putChunks (Job job, int workers, int masterChunk,
				long lb, long ub)
			{
			job.putChunksProportional (workers, masterChunk, lb, ub);
			}
		}),

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
	 */
	guided (new Scheduler()
		{
		public int actualChunk (int chunk)
			{
			return chunk == Task.STANDARD_CHUNK ? 1 : chunk;
			}
		public void iterate (IntParallelForLoop pfl, Loop loop, int rank)
			throws Exception
			{
			pfl.iterateGuided (loop, rank);
			}
		public void iterate (LongParallelForLoop pfl, LongLoop loop, int rank)
			throws Exception
			{
			pfl.iterateGuided (loop, rank);
			}
		public void putChunks (Job job, int workers, int masterChunk,
				int lb, int ub)
			{
			job.putChunksGuided (workers, masterChunk, lb, ub);
			}
		public void putChunks (Job job, int workers, int masterChunk,
				long lb, long ub)
			{
			job.putChunksGuided (workers, masterChunk, lb, ub);
			}
		});

// Hidden helper interface.

	/**
	 * Interface Scheduler specifies the interface for a parallel for loop
	 * scheduler.
	 *
	 * @author  Alan Kaminsky
	 * @version 09-Jan-2014
	 */
	private interface Scheduler
		{
		/**
		 * Returns the actual <TT>chunk</TT> parameter value for this schedule.
		 *
		 * @param  chunk  <TT>chunk</TT> parameter, or STANDARD_CHUNK.
		 *
		 * @return  Actual <TT>chunk</TT> parameter.
		 */
		public int actualChunk
			(int chunk);

		/**
		 * Perform the iterations of the given integer parallel for loop using
		 * this schedule's pattern.
		 *
		 * @param  pfl   Integer parallel for loop.
		 * @param  loop  Loop body.
		 * @param  rank  Team thread rank.
		 */
		public void iterate
			(IntParallelForLoop pfl,
			 Loop loop,
			 int rank)
			throws Exception;

		/**
		 * Perform the iterations of the given long integer parallel for loop
		 * using this schedule's pattern.
		 *
		 * @param  pfl   Long integer parallel for loop.
		 * @param  loop  Loop body.
		 * @param  rank  Team thread rank.
		 */
		public void iterate
			(LongParallelForLoop pfl,
			 LongLoop loop,
			 int rank)
			throws Exception;

		/**
		 * Partition the given integer loop index range into chunks using this
		 * schedule's pattern, and write the chunks into the given job's tuple
		 * space.
		 *
		 * @param  job          Job.
		 * @param  workers      Number of worker tasks.
		 * @param  masterChunk  <TT>masterChunk</TT> parameter.
		 * @param  lb           Loop index lower bound.
		 * @param  ub           Loop index upper bound.
		 */
		public void putChunks
			(Job job,
			 int workers,
			 int masterChunk,
			 int lb,
			 int ub);

		/**
		 * Partition the given long integer loop index range into chunks using
		 * this schedule's pattern, and write the chunks into the given job's
		 * tuple space.
		 *
		 * @param  job          Job.
		 * @param  workers      Number of worker tasks.
		 * @param  masterChunk  <TT>masterChunk</TT> parameter.
		 * @param  lb           Loop index lower bound.
		 * @param  ub           Loop index upper bound.
		 */
		public void putChunks
			(Job job,
			 int workers,
			 int masterChunk,
			 long lb,
			 long ub);
		}

// Hidden data members.

	private static Schedule[] enumerals = values();

	private Scheduler scheduler;

// Hidden constructors.

	/**
	 * Construct a new Schedule enumeral.
	 *
	 * @param  scheduler  Scheduler.
	 */
	private Schedule
		(Scheduler scheduler)
		{
		this.scheduler = scheduler;
		}

// Exported operations.

	/**
	 * Write a Schedule enumeral to the given out stream.
	 *
	 * @param  schedule  Schedule enumeral, or null.
	 * @param  out       Object output stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public static void writeOut
		(Schedule schedule,
		 OutStream out)
		throws IOException
		{
		out.writeInt (schedule == null ? -1 : schedule.ordinal());
		}

	/**
	 * Read a Schedule enumeral from the given in stream.
	 *
	 * @param  in  Object input stream.
	 *
	 * @return  Schedule enumeral.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public static Schedule readIn
		(InStream in)
		throws IOException
		{
		int ordinal = in.readInt();
		if (-1 > ordinal || ordinal >= enumerals.length)
			throw new InvalidObjectException (String.format
				("Schedule.read(): ordinal = %d invalid", ordinal));
		return ordinal == -1 ? null : enumerals[ordinal];
		}

// Hidden operations.

	/**
	 * Returns the actual <TT>chunk</TT> parameter value for this schedule.
	 *
	 * @param  chunk  <TT>chunk</TT> parameter, or STANDARD_CHUNK.
	 *
	 * @return  Actual <TT>chunk</TT> parameter.
	 */
	int actualChunk
		(int chunk)
		{
		return scheduler.actualChunk (chunk);
		}

	/**
	 * Perform the iterations of the given integer parallel for loop using this
	 * schedule's pattern.
	 *
	 * @param  pfl   Integer parallel for loop.
	 * @param  loop  Loop body.
	 * @param  rank  Team thread rank.
	 */
	void iterate
		(IntParallelForLoop pfl,
		 Loop loop,
		 int rank)
		throws Exception
		{
		scheduler.iterate (pfl, loop, rank);
		}

	/**
	 * Perform the iterations of the given long integer parallel for loop using
	 * this schedule's pattern.
	 *
	 * @param  pfl   Long integer parallel for loop.
	 * @param  loop  Loop body.
	 * @param  rank  Team thread rank.
	 */
	void iterate
		(LongParallelForLoop pfl,
		 LongLoop loop,
		 int rank)
		throws Exception
		{
		scheduler.iterate (pfl, loop, rank);
		}

	/**
	 * Partition the given integer loop index range into chunks using this
	 * schedule's pattern, and write the chunks into the given job's tuple
	 * space.
	 *
	 * @param  job          Job.
	 * @param  workers      Number of worker tasks.
	 * @param  masterChunk  <TT>masterChunk</TT> parameter.
	 * @param  lb           Loop index lower bound.
	 * @param  ub           Loop index upper bound.
	 */
	void putChunks
		(Job job,
		 int workers,
		 int masterChunk,
		 int lb,
		 int ub)
		{
		scheduler.putChunks
			(job, workers, scheduler.actualChunk (masterChunk), lb, ub);
		}

	/**
	 * Partition the given long integer loop index range into chunks using this
	 * schedule's pattern, and write the chunks into the given job's tuple
	 * space.
	 *
	 * @param  job          Job.
	 * @param  workers      Number of worker tasks.
	 * @param  masterChunk  <TT>masterChunk</TT> parameter.
	 * @param  lb           Loop index lower bound.
	 * @param  ub           Loop index upper bound.
	 */
	void putChunks
		(Job job,
		 int workers,
		 int masterChunk,
		 long lb,
		 long ub)
		{
		scheduler.putChunks
			(job, workers, scheduler.actualChunk (masterChunk), lb, ub);
		}

	}
