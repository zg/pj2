//******************************************************************************
//
// File:    DoubleList.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.DoubleList
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

package edu.rit.util;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import java.io.IOException;
import java.util.Arrays;

/**
 * Class DoubleList provides a list of double precision floating point numbers
 * (type <TT>double</TT>) stored in a dynamically-sized array.
 * <P>
 * Operations take constant time unless otherwise specified. <I>n</I> is the
 * number of items stored in the list.
 *
 * @author  Alan Kaminsky
 * @version 12-Mar-2015
 */
public class DoubleList
	implements Streamable
	{

// Hidden data members.

	// Chunk size for growing array.
	private static final int CHUNK = 8;

	// Array of list items.
	double[] item;

	// Number of list items.
	int size;

// Exported constructors.

	/**
	 * Construct a new list.
	 */
	public DoubleList()
		{
		item = new double [CHUNK];
		}

	/**
	 * Construct a new list that is a copy of the given list. The new list's
	 * entries contain the same items in the same order as the given list. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  list  List to copy.
	 */
	public DoubleList
		(DoubleList list)
		{
		copy (list);
		}

// Exported operations.

	/**
	 * Determine if this list is empty.
	 *
	 * @return  True if this list is empty, false if it isn't.
	 */
	public boolean isEmpty()
		{
		return size == 0;
		}

	/**
	 * Clear this list.
	 */
	public void clear()
		{
		size = 0;
		}

	/**
	 * Set this list to a copy of the given list. Afterwards, this list contains
	 * the same items in the same order as the given list. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  list  List to copy.
	 */
	public void copy
		(DoubleList list)
		{
		int newlength = list.size + CHUNK - 1;
		newlength = Math.max (newlength/CHUNK, 1);
		newlength *= CHUNK;
		item = new double [newlength];
		System.arraycopy (list.item, 0, item, 0, list.size);
		size = list.size;
		}

	/**
	 * Returns the number of items in this list.
	 *
	 * @return  Number of items.
	 */
	public int size()
		{
		return size;
		}

	/**
	 * Get the item at the given position in this list.
	 *
	 * @param  p  Position, in the range 0 .. <TT>size()</TT>&minus;1.
	 *
	 * @return  Item stored at position <TT>p</TT>.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>p</TT> is out of bounds.
	 */
	public double get
		(int p)
		{
		if (0 > p || p >= size)
			{
			throw new IndexOutOfBoundsException
				("DoubleList.get(): p = "+p+" out of bounds");
			}
		return item[p];
		}

	/**
	 * Set the item at the given position in this list.
	 *
	 * @param  p  Position, in the range 0 .. <TT>size()</TT>&minus;1.
	 * @param  i  Item to store at position <TT>p</TT>.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>p</TT> is out of bounds.
	 */
	public void set
		(int p,
		 double i)
		{
		if (0 > p || p >= size)
			{
			throw new IndexOutOfBoundsException
				("DoubleList.set(): p = "+p+" out of bounds");
			}
		item[p] = i;
		}

	/**
	 * Swap the items at the given positions in this list.
	 *
	 * @param  p  First item position, in the range 0 ..
	 *            <TT>size()</TT>&minus;1.
	 * @param  q  Second item position, in the range 0 ..
	 *            <TT>size()</TT>&minus;1.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>p</TT> or <TT>q</TT> is out of
	 *     bounds.
	 */
	public void swap
		(int p,
		 int q)
		{
		if (0 > p || p >= size)
			{
			throw new IndexOutOfBoundsException
				("DoubleList.swap(): p = "+p+" out of bounds");
			}
		if (0 > q || q >= size)
			{
			throw new IndexOutOfBoundsException
				("DoubleList.swap(): q = "+q+" out of bounds");
			}
		double tmp = item[p];
		item[p] = item[q];
		item[q] = tmp;
		}

	/**
	 * Get the position of the given item in this list. This involves a linear
	 * scan of the list items starting from the beginning of the list; the
	 * position of the first item encountered that is equal to the given item is
	 * returned; if there is no such item, &minus;1 is returned. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  i  Item.
	 *
	 * @return  Position of <TT>i</TT> in the range 0 ..
	 *          <TT>size()</TT>&minus;1, or &minus;1 if none.
	 */
	public int position
		(double i)
		{
		for (int p = 0; p < size; ++ p)
			{
			if (item[p] == i) return p;
			}
		return -1;
		}

	/**
	 * Get the position of the first item in this list for which the given
	 * predicate is true. This involves a linear scan of the list items starting
	 * from the beginning of the list; the returned value is the position of the
	 * first item encountered for which the predicate's <TT>test()</TT> method
	 * returns true; if there is no such item, &minus;1 is returned. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  predicate  Predicate.
	 *
	 * @return  Position of matching item in the range 0 ..
	 *          <TT>size()</TT>&minus;1, or &minus;1 if none.
	 */
	public int position
		(DoublePredicate predicate)
		{
		for (int p = 0; p < size; ++ p)
			if (predicate.test (item[p]))
				return p;
		return -1;
		}

	/**
	 * Add the given item to the end of this list. This list's size increases by
	 * 1.
	 *
	 * @param  i  Item.
	 *
	 * @return  Newly added item.
	 */
	public double addLast
		(double i)
		{
		if (size == item.length)
			{
			double[] newitem = new double [item.length + CHUNK];
			System.arraycopy (item, 0, newitem, 0, size);
			item = newitem;
			}
		item[size] = i;
		++ size;
		return i;
		}

	/**
	 * Add the given item to the beginning of this list. Items at position 0 and
	 * beyond are moved forward one position. This list's size increases by 1.
	 * Time: <I>O</I>(<I>n</I>).
	 *
	 * @param  i  Item.
	 *
	 * @return  Newly added item.
	 */
	public double addFirst
		(double i)
		{
		if (size == item.length)
			{
			double[] newitem = new double [item.length + CHUNK];
			System.arraycopy (item, 0, newitem, 1, size);
			item = newitem;
			}
		else
			{
			System.arraycopy (item, 0, item, 1, size);
			}
		item[0] = i;
		++ size;
		return i;
		}

	/**
	 * Add the given item to this list at the given position. Items at position
	 * <TT>p</TT> and beyond are moved forward one position. This list's size
	 * increases by 1. Time: <I>O</I>(<I>n</I>).
	 *
	 * @param  p  Position, in the range 0 .. <TT>size()</TT>.
	 * @param  i  Item.
	 *
	 * @return  Newly added item.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>p</TT> is out of bounds.
	 */
	public double add
		(int p,
		 double i)
		{
		if (0 > p || p > size)
			{
			throw new IndexOutOfBoundsException
				("DoubleList.add(): p = "+p+" out of bounds");
			}
		if (size == item.length)
			{
			double[] newitem = new double [item.length + CHUNK];
			System.arraycopy (item, 0, newitem, 0, p);
			System.arraycopy (item, p, newitem, p + 1, size - p);
			item = newitem;
			}
		else
			{
			System.arraycopy (item, p, item, p + 1, size - p);
			}
		item[p] = i;
		++ size;
		return i;
		}

	/**
	 * Remove the item at the end of this list. The removed item is returned.
	 * This list's size decreases by 1. If this list is empty, nothing happens
	 * and 0 is returned.
	 *
	 * @return  Removed item, or 0 if this list is empty.
	 */
	public double removeLast()
		{
		if (size == 0) return 0;
		-- size;
		return item[size];
		}

	/**
	 * Remove the item at the beginning of this list. Items at position 1 and
	 * beyond are moved backward one position. The removed item is returned.
	 * This list's size decreases by 1. If this list is empty, nothing happens
	 * and 0 is returned. Time: <I>O</I>(<I>n</I>).
	 *
	 * @return  Removed item, or 0 if this list is empty.
	 */
	public double removeFirst()
		{
		if (size == 0) return 0;
		-- size;
		double tmp = item[0];
		System.arraycopy (item, 1, item, 0, size);
		return tmp;
		}

	/**
	 * Remove the item at the given position in this list. Items at position
	 * <TT>p</TT>+1 and beyond are moved backward one position. The removed item
	 * is returned. This list's size decreases by 1. If this list is empty,
	 * nothing happens and 0 is returned. Time: <I>O</I>(<I>n</I>).
	 *
	 * @param  p  Position, in the range 0 .. <TT>size()</TT>&minus;1.
	 *
	 * @return  Removed item, or 0 if this list is empty.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if this list is not empty and <TT>p</TT>
	 *     is out of bounds.
	 */
	public double remove
		(int p)
		{
		if (size == 0) return 0;
		if (0 > p || p >= size)
			{
			throw new IndexOutOfBoundsException
				("DoubleList.remove(): p = "+p+" out of bounds");
			}
		-- size;
		double tmp = item[p];
		System.arraycopy (item, p + 1, item, p, size - p);
		return tmp;
		}

	/**
	 * Swap the item at the given position with the item at the end of this
	 * list, then remove the item at the end of this list. The removed item,
	 * which used to be at the given position, is returned. This list's size
	 * decreases by 1.
	 *
	 * @param  p  Position, in the range 0 .. <TT>size()</TT>&minus;1.
	 *
	 * @return  Removed item.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>p</TT> is out of bounds.
	 */
	public double swapRemoveLast
		(int p)
		{
		if (0 > p || p >= size)
			{
			throw new IndexOutOfBoundsException
				("DoubleList.swapRemoveLast(): p = "+p+" out of bounds");
			}
		-- size;
		double tmp = item[p];
		item[p] = item[size];
		return tmp;
		}

	/**
	 * Perform the given action on each item in this list. For each item in this
	 * list, the given <TT>action</TT>'s <TT>run()</TT> method is called,
	 * passing in the item.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds items to or removes
	 * items from the list, the <TT>forEachItemDo()</TT> method's behavior is
	 * not specified.
	 *
	 * @param  action  Action.
	 */
	public void forEachItemDo
		(DoubleAction action)
		{
		for (int p = 0; p < size; ++ p) action.run (item[p]);
		}

	/**
	 * Perform the given action on each item in this list and return a result.
	 * For each item in this list, the given <TT>action</TT>'s <TT>run()</TT>
	 * method is called, passing in the item. After all the list items have been
	 * processed, the given <TT>action</TT>'s <TT>result()</TT> method is
	 * called, and its result is returned.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds items to or removes
	 * items from the list, the <TT>forEachItemDo()</TT> method's behavior is
	 * not specified.
	 *
	 * @param  <R>     Result data type.
	 * @param  action  Action.
	 *
	 * @return  Result of processing all the list items.
	 */
	public <R> R forEachItemDo
		(DoubleActionResult<R> action)
		{
		for (int p = 0; p < size; ++ p) action.run (item[p]);
		return action.result();
		}

	/**
	 * Evaluate the given predicate on, and possibly remove, each item in this
	 * list. For each item in this list from beginning to end, the given
	 * <TT>predicate</TT>'s <TT>test()</TT> method is called, passing in the
	 * item. If the <TT>test()</TT> method returns true, the item is removed
	 * from this list. Time: <I>O</I>(<I>n</I>).
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>predicate</TT> adds items to or removes
	 * items from the list, other than by returning true, the
	 * <TT>removeEachItemIf()</TT> method's behavior is not specified.
	 *
	 * @param  predicate  Predicate.
	 */
	public void removeEachItemIf
		(DoublePredicate predicate)
		{
		int p = 0;
		while (p < size)
			if (predicate.test (item[p]))
				remove (p);
			else
				++ p;
		}

	/**
	 * Store this list's items in the given array. The first item is stored at
	 * index 0, the second item at index 1, and so on. The number of array
	 * elements set is <TT>array.length</TT>. If this list contains fewer than
	 * <TT>array.length</TT> items, the remaining array elements are set to 0.
	 *
	 * @param  array  Array in which to store items.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 */
	public double[] toArray
		(double[] array)
		{
		return toArray (array, 0, array.length);
		}

	/**
	 * Store this list's items in the given array. The first item is stored at
	 * index <TT>off</TT>, the second item at index <TT>off</TT>+1, and so on.
	 * The number of array elements set is <TT>len</TT>. If this list contains
	 * fewer than <TT>len</TT> items, the remaining array elements are set to 0.
	 *
	 * @param  array  Array in which to store items.
	 * @param  off    Index at which to store first item.
	 * @param  len    Number of items to store.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>array.length</TT>.
	 */
	public double[] toArray
		(double[] array,
		 int off,
		 int len)
		{
		if (off < 0 || len < 0 || off+len > array.length)
			{
			throw new IndexOutOfBoundsException();
			}
		int n = Math.min (size, len);
		System.arraycopy (item, 0, array, off, n);
		if (n < len) Arrays.fill (array, off + n, off + len, 0);
		return array;
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
		out.writeInt (size);
		for (int i = 0; i < size; ++ i)
			out.writeDouble (item[i]);
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
		size = in.readInt();
		int newlength = size + CHUNK - 1;
		newlength = Math.max (newlength/CHUNK, 1);
		newlength *= CHUNK;
		item = new double [newlength];
		for (int i = 0; i < size; ++ i)
			item[i] = in.readDouble();
		}

	}
