//******************************************************************************
//
// File:    WorkQueue.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.WorkQueue
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

import edu.rit.util.DList;

/**
 * Class WorkQueue provides a queue of work items to be performed by an
 * {@linkplain ObjectParallelForLoop}.
 * <P>
 * <B>Blocking behavior.</B>
 * The {@link #add(Object) add()} method does not block. The {@link #remove()
 * remove()} method's blocking behavior depends on the work queue's
 * <TT>threads</TT> property. If the queue is not empty, {@link #remove()
 * remove()} will not block, will remove a work item from the queue, and will
 * return the item. If the queue is empty, {@link #remove() remove()} will block
 * until either (1) an item is added to the queue, in which case the item is
 * removed and returned, or (2) the number of threads blocked calling {@link
 * #remove() remove()} is greater than or equal to the <TT>threads</TT>
 * property, in which case all the threads are unblocked and null is returned.
 * <P>
 * The {@link #threads() threads()} method queries the <TT>threads</TT>
 * property, and the {@link #threads(int) threads(int)} method sets the
 * <TT>threads</TT> property. Normally you do not need to set the
 * <TT>threads</TT> property yourself. When a parallel for loop iterates over a
 * work queue, the parallel for loop automatically sets the <TT>threads</TT>
 * property to the number of threads in the parallel team. The team threads
 * repeatedly remove work items from the queue and process them. When all the
 * work items have been processed, and all the team threads are calling {@link
 * #remove() remove()} trying to get the next work item, all the team threads
 * will get null instead; this signals the parallel for loop to terminate.
 * <P>
 * The work queue is designed this way so that parallel team threads can add
 * new work items to the queue while processing work items removed from the
 * queue. For example, say there are four team threads; the queue is empty;
 * three team threads are blocked calling {@link #remove() remove()}; the fourth
 * team thread is processing a work item. In this situation, the three team
 * threads must stay blocked in case the fourth team thread adds a work item or
 * items to the queue, whereupon one or more of the blocked team threads can
 * unblock, receive the work items, and process them. If on the other hand the
 * fourth team thread does not add a work item and calls {@link #remove()
 * remove()}, this signifies that there are no more work items, so all the team
 * threads unblock, receive null, and terminate.
 * <P>
 * You can affect the {@link #remove() remove()} method's blocking behavior by
 * setting the work queue's <TT>threads</TT> property. But be very sure you know
 * what you are doing.
 *
 * @param  <W>  Data type of the work items.
 *
 * @author  Alan Kaminsky
 * @version 20-May-2015
 */
public class WorkQueue<W>
	{

// Hidden data members.

	private DList<W> queue = new DList<W>();
	private int threads = Integer.MAX_VALUE;
	private int blocked = 0;

// Hidden constructors.

	/**
	 * Construct a new work queue.
	 */
	public WorkQueue()
		{
		}

// Exported operations.

	/**
	 * Clear this work queue. Any work items in the queue are deleted.
	 */
	public synchronized void clear()
		{
		queue.clear();
		notifyAll();
		}

	/**
	 * Add the given work item to this work queue.
	 *
	 * @param  workitem  Work item; must be non-null.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>workitem</TT> is null.
	 */
	public synchronized void add
		(W workitem)
		{
		if (workitem == null)
			throw new NullPointerException
				("WorkQueue.add(): workitem is null");
		queue.addLast (workitem);
		notifyAll();
		}

	/**
	 * Remove and return the next work item from this work queue. This method's
	 * blocking behavior is described above.
	 *
	 * @return  Work item, or null if there are no more work items.
	 *
	 * @exception  InterruptedException
	 *     Thrown if the calling thread is interrupted while blocked in this
	 *     method.
	 */
	public synchronized W remove()
		throws InterruptedException
		{
		++ blocked;
		notifyAll();
		while (queue.isEmpty() && blocked < threads)
			wait();
		if (! queue.isEmpty())
			{
			-- blocked;
			notifyAll();
			return queue.first().remove().item();
			}
		else
			return null;
		}

	/**
	 * Get this work queue's <TT>threads</TT> property.
	 *
	 * @return  <TT>threads</TT> property.
	 */
	public synchronized int threads()
		{
		return threads;
		}

	/**
	 * Set this work queue's <TT>threads</TT> property.
	 *
	 * @param  threads  <TT>threads</TT> property.
	 */
	public synchronized void threads
		(int threads)
		{
		this.threads = threads;
		notifyAll();
		}

	}
