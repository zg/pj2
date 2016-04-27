//******************************************************************************
//
// File:    WorkerIntParallelForLoop.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.WorkerIntParallelForLoop
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
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Class WorkerIntParallelForLoop provides the worker portion of a master-worker
 * cluster parallel for loop with a loop index of type <TT>int</TT>.
 *
 * @author  Alan Kaminsky
 * @version 20-Jun-2014
 */
class WorkerIntParallelForLoop
	extends IntParallelForLoop
	{

// Hidden data members.

	private Chunk template;
	private Chunk masterChunk;
	private CyclicBarrier barrier;

// Hidden constructors.

	/**
	 * Construct a new worker integer parallel for loop.
	 *
	 * @param  task        Task in which the parallel for loop is executing.
	 * @param  properties  Parallel for loop properties.
	 */
	WorkerIntParallelForLoop
		(Task task,
		 TaskProperties properties)
		{
		super (task, 0, 0);
		this.properties = properties.chain (task.properties);
		}

// Exported operations.

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
		if (loop == null)
			throw new NullPointerException
				("WorkerIntParallelForLoop.exec(): loop is null");
		threads = properties.actualThreads();
		schedule = properties.schedule();
		chunk = schedule.actualChunk (properties.chunk());
		nextlb = new AtomicLong();
		this.loop = loop;
		stop = false;
		template = new Chunk() .rank (task.taskRank());
		barrier = new CyclicBarrier (threads, new Runnable()
			{
			public void run()
				{
				try
					{
					masterChunk = stop ? null : task.tryToTakeTuple (template);
					if (masterChunk != null)
						{
						blb = masterChunk.lb();
						bub = masterChunk.ub();
						nextlb.set (blb);
						}
					}
				catch (IOException exc)
					{
					throw new TerminateException (exc);
					}
				}
			});
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
		Loop loop = rank == 0 ? this.loop : (Loop)(this.loop.clone());
		loop.parallelForLoop = this;
		loop.rank = rank;
		loop.reductionMap = reductionMap;

		// Perform one-time initialization.
		loop.start();

		// Execute a series of chunks from the master.
		for (;;)
			{
			barrier.await();
			if (masterChunk == null) break;
			int stride = masterChunk.stride();
			if (stride == 1)
				schedule.iterate (this, loop, rank);
			else
				iterateStridedLeapfrog (loop, rank, stride);
			}

		// Perform one-time finalization.
		loop.finish();

		loop.parallelForLoop = null;
		loop.rank = -1;
		loop.reductionMap = null;
		}

	/**
	 * Execute this parallel for loop's iterations with a strided leapfrog
	 * schedule.
	 *
	 * @param  loop    Loop body.
	 * @param  rank    Team thread rank.
	 * @param  stride  Stride.
	 */
	void iterateStridedLeapfrog
		(Loop loop,
		 int rank,
		 int stride)
		throws Exception
		{
		int rankStride = rank*stride;
		int thrStride = threads*stride;
		int chunklb = (int)(blb + rankStride);
		int chunkub = (int)(bub);
		for (int i = chunklb; ! stop && i <= chunkub; i += thrStride)
			loop.run (i);
		}

	}
