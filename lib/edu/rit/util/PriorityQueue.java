//******************************************************************************
//
// File:    PriorityQueue.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.PriorityQueue
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

package edu.rit.util;

/**
 * Class PriorityQueue provides a priority queue. The items in the queue are
 * instances of a subclass of class {@linkplain PriorityQueue.Item
 * PriorityQueue.Item}. Items are removed from the queue in priority order, as
 * determined by the item's {@link
 * PriorityQueue.Item#comesBefore(PriorityQueue.Item) comesBefore()} method.
 *
 * @param  <T>  Item data type.
 *
 * @author  Alan Kaminsky
 * @version 12-Apr-2014
 */
public class PriorityQueue<T extends PriorityQueue.Item>
	{

// Hidden data members.

	// Chunk size for growing array.
	private static final int CHUNK = 8;

	// Array of queue items, organized as a heap. Index 0 is not used.
	private T[] items = (T[]) new Item [CHUNK];

	// Number of queue items.
	private int size = 0;

// Exported constructors.

	/**
	 * Construct a new priority queue.
	 */
	public PriorityQueue()
		{
		}

// Exported operations.

	/**
	 * Determine if this priority queue is empty.
	 *
	 * @return  True if this priority queue is empty, false if it isn't.
	 */
	public boolean isEmpty()
		{
		return size == 0;
		}

	/**
	 * Get the number of items in this priority queue.
	 *
	 * @return  Number of items.
	 */
	public int size()
		{
		return size;
		}

	/**
	 * Clear this priority queue.
	 */
	public void clear()
		{
		for (int i = 1; i <= size; ++ i)
			{
			items[i].queue = null;
			items[i] = null;
			}
		size = 0;
		}

	/**
	 * Add the given item to this priority queue.
	 *
	 * @param  item  Item.
	 *
	 * @return  Item that was added, namely <TT>item</TT>.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>item</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>item</TT> is already in a queue.
	 */
	public T add
		(T item)
		{
		if (item == null)
			throw new NullPointerException
				("PriorityQueue.add(): item is null");
		if (item.queue != null)
			throw new IllegalArgumentException
				("PriorityQueue.add(): item is already in a queue");
		if (size == items.length - 1)
			{
			T[] newitems = (T[]) new Item [items.length + CHUNK];
			System.arraycopy (items, 1, newitems, 1, size);
			items = newitems;
			}
		++ size;
		items[size] = item;
		item.queue = this;
		item.index = size;
		siftUp (size);
		return item;
		}

	/**
	 * Remove and return the highest-priority item from this priority queue. If
	 * this priority queue is empty, nothing happens and null is returned.
	 *
	 * @return  Item that was removed, or null.
	 */
	public T remove()
		{
		if (size == 0) return null;
		T item = items[1];
		items[1] = items[size];
		items[1].index = 1;
		items[size] = null;
		-- size;
		siftDown (1);
		item.queue = null;
		return item;
		}

// Hidden operations.

	/**
	 * Sift up the item at the given index.
	 *
	 * @param  index  Item index.
	 */
	void siftUp
		(int index)
		{
		T tmp;
		int C = index;
		int P = C/2;
		while (P >= 1 && items[C].comesBefore (items[P]))
			{
			tmp = items[P]; items[P] = items[C]; items[C] = tmp;
			items[P].index = P;
			items[C].index = C;
			C = P;
			P = C/2;
			}
		}

	/**
	 * Sift down the item at the given index.
	 *
	 * @param  index  Item index.
	 */
	void siftDown
		(int index)
		{
		T tmp;
		int P = index;
		int LC = P*2;
		int RC = P*2 + 1;
		int minC;
		while (LC <= size)
			{
			if (RC <= size && items[RC].comesBefore (items[LC]))
				minC = RC;
			else
				minC = LC;
			if (items[P].comesBefore (items[minC])) break;
			tmp = items[P]; items[P] = items[minC]; items[minC] = tmp;
			items[P].index = P;
			items[minC].index = minC;
			P = minC;
			LC = P*2;
			RC = P*2 + 1;
			}
		}

	/**
	 * Class PriorityQueue.Item is the abstract base class for an item contained
	 * in a {@linkplain PriorityQueue}.
	 *
	 * @author  Alan Kaminsky
	 * @version 12-Apr-2014
	 */
	public static abstract class Item
		{

	// Hidden data members.

		PriorityQueue queue; // Priority queue containing this item
		int index;           // Index of this item in the queue

	// Exported constructors.

		/**
		 * Construct a new priority queue item.
		 */
		public Item()
			{
			}

	// Exported operations.

		/**
		 * Determine if this item is contained in a queue.
		 *
		 * @return  True if this item is in a queue, false if it isn't.
		 */
		public boolean enqueued()
			{
			return queue != null;
			}

		/**
		 * Increase the priority of this item in the queue. If this item is not
		 * contained in a queue, nothing happens.
		 */
		public void increasePriority()
			{
			if (queue != null)
				queue.siftUp (index);
			}

		/**
		 * Decrease the priority of this item in the queue. If this item is not
		 * contained in a queue, nothing happens.
		 */
		public void decreasePriority()
			{
			if (queue != null)
				queue.siftDown (index);
			}

		/**
		 * Determine the priority of this item relative to the given item.
		 *
		 * @param  item  Priority queue item.
		 *
		 * @return  True if this item has a higher priority than <TT>item</TT>,
		 *          false otherwise.
		 */
		public abstract boolean comesBefore
			(Item item);

		}

	}
