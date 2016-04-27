//******************************************************************************
//
// File:    DList.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.DList
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

package edu.rit.util;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import java.io.IOException;

/**
 * Class DList provides a list of items stored in a doubly-linked list. Each
 * item is contained in a {@linkplain DListEntry DListEntry}.
 * <P>
 * Operations take constant time unless otherwise specified. <I>n</I> is the
 * number of items stored in the list.
 *
 * @param  <T>  List item data type.
 *
 * @author  Alan Kaminsky
 * @version 23-Dec-2014
 */
public class DList<T>
	implements Streamable
	{

// Hidden data members.

	// Sentinel entry.
	// - Its item field is unused (null).
	// - Its list field is null to indicate that it is a sentinel.
	// - Its predecessor field points to the last entry in the list.
	// - Its successor field points to the first entry in the list.
	private DListEntry<T> sentinel;

// Exported constructors.

	/**
	 * Construct a new list.
	 */
	public DList()
		{
		sentinel = new DListEntry<T> (null);
		sentinel.pred = sentinel;
		sentinel.succ = sentinel;
		}

	/**
	 * Construct a new list that is a copy of the given list. The new list's
	 * entries contain the same items in the same order as the given list. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  list  List to copy.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>list</TT> is null.
	 */
	public DList
		(DList<T> list)
		{
		this();
		copy (list);
		}

	/**
	 * Construct a new list that is a copy of the given list. The new list's
	 * entries contain the same items in the same order as the given list. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  list  List to copy.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>list</TT> is null.
	 */
	public DList
		(AList<T> list)
		{
		this();
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
		return sentinel.pred == sentinel;
		}

	/**
	 * Clear this list. Time: <I>O</I>(<I>n</I>).
	 */
	public void clear()
		{
		DListEntry<T> p, q;
		p = first();
		while (p != null)
			{
			q = p.succ();
			p.clear();
			p = q;
			}
		sentinel.pred = sentinel;
		sentinel.succ = sentinel;
		}

	/**
	 * Set this list to a copy of the given list. Afterwards, this list's
	 * entries contain the same items in the same order as the given list. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  list  List to copy.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>list</TT> is null.
	 */
	public void copy
		(DList<T> list)
		{
		if (list == null)
			throw new NullPointerException
				("DList.copy(): list is null");
		clear();
		DListEntry<T> p = list.first();
		while (p != null)
			{
			addLast (p.item);
			p = p.succ();
			}
		}

	/**
	 * Set this list to a copy of the given list. Afterwards, this list's
	 * entries contain the same items in the same order as the given list. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  list  List to copy.
	 */
	public void copy
		(AList<T> list)
		{
		clear();
		list.forEachItemDo (new Action<T>()
			{
			public void run (T item)
				{
				addLast (item);
				}
			});
		}

	/**
	 * Returns the number of items in this list. Time: <I>O</I>(<I>n</I>).
	 *
	 * @return  Number of items.
	 */
	public int size()
		{
		int size = 0;
		DListEntry<T> p = first();
		while (p != null)
			{
			++ size;
			p = p.succ();
			}
		return size;
		}

	/**
	 * Get the first entry in this list. If this list is empty, null is
	 * returned.
	 *
	 * @return  First entry, or null.
	 */
	public DListEntry<T> first()
		{
		return isEmpty() ? null : sentinel.succ;
		}

	/**
	 * Get the last entry in this list. If this list is empty, null is returned.
	 *
	 * @return  Last entry, or null.
	 */
	public DListEntry<T> last()
		{
		return isEmpty() ? null : sentinel.pred;
		}

	/**
	 * Find the first entry in this list that contains the given item. This
	 * involves a linear scan of the list entries from beginning to end; the
	 * returned value is the first entry encountered whose item is equal to the
	 * given item, as determined by the item's <TT>equals()</TT> method; if
	 * there is no such entry, null is returned. Time: <I>O</I>(<I>n</I>).
	 *
	 * @param  item  Item.
	 *
	 * @return  Entry containing matching item, or null if none.
	 */
	public DListEntry<T> find
		(T item)
		{
		DListEntry<T> p = first();
		while (p != null)
			if (p.item.equals (item))
				return p;
			else
				p = p.succ();
		return null;
		}

	/**
	 * Find the first entry in this list that contains an item for which the
	 * given predicate is true. This involves a linear scan of the list entries
	 * from beginning to end; the returned value is the first entry encountered
	 * for which the predicate's <TT>test()</TT> method returns true; if there
	 * is no such entry, null is returned. Time: <I>O</I>(<I>n</I>).
	 *
	 * @param  predicate  Predicate.
	 *
	 * @return  Entry containing matching item, or null if none.
	 */
	public DListEntry<T> find
		(Predicate<T> predicate)
		{
		DListEntry<T> p = first();
		while (p != null)
			if (predicate.test (p.item))
				return p;
			else
				p = p.succ();
		return null;
		}

	/**
	 * Determine if this list that contains the given item. This involves a
	 * linear scan of the list entries from beginning to end; if an entry is
	 * encountered whose item is equal to the given item, as determined by the
	 * item's <TT>equals()</TT> method, true is returned; otherwise false is
	 * returned. Time: <I>O</I>(<I>n</I>).
	 *
	 * @param  item  Item.
	 *
	 * @return  True if this list contains <TT>item</TT>, false otherwise.
	 */
	public boolean contains
		(T item)
		{
		return find (item) != null;
		}

	/**
	 * Determine if this list contains an item for which the given predicate is
	 * true. This involves a linear scan of the list entries from beginning to
	 * end; if an entry is encountered for which the predicate's <TT>test()</TT>
	 * method returns true, true is returned; otherwise false is returned. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  predicate  Predicate.
	 *
	 * @return  True if this list contains an item matching <TT>predicate</TT>,
	 *          false otherwise.
	 */
	public boolean contains
		(Predicate<T> predicate)
		{
		return find (predicate) != null;
		}

	/**
	 * Determine if this list contains the given entry. This involves a linear
	 * scan of the list entries from beginning to end; if the given entry is
	 * encountered, true is returned; otherwise false is returned. Time:
	 * <I>O</I>(<I>n</I>).
	 *
	 * @param  entry  Entry.
	 *
	 * @return  True if this list contains <TT>entry</TT>, false otherwise.
	 */
	public boolean contains
		(DListEntry<T> entry)
		{
		DListEntry<T> p = first();
		while (p != null)
			if (p == entry)
				return true;
			else
				p = p.succ();
		return false;
		}

	/**
	 * Add a new entry containing the given item at the beginning of this list.
	 * The item may be null.
	 *
	 * @param  item  Item.
	 *
	 * @return  Entry that was added.
	 */
	public DListEntry<T> addFirst
		(T item)
		{
		return addFirst (new DListEntry<T> (item));
		}

	/**
	 * Add the given entry at the beginning of this list.
	 *
	 * @param  entry  Entry.
	 *
	 * @return  Entry that was added (namely <TT>entry</TT>).
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>entry</TT> is null.
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if <TT>entry</TT> is in a list.
	 */
	public DListEntry<T> addFirst
		(DListEntry<T> entry)
		{
		if (entry == null)
			throw new NullPointerException
				("DList.addFirst(): entry is null");
		if (entry.list != null)
			throw new IllegalStateException
				("DList.addFirst(): entry is in a list");

		entry.list = this;
		entry.pred = sentinel;
		entry.succ = sentinel.succ;
		sentinel.succ.pred = entry;
		sentinel.succ = entry;

		return entry;
		}

	/**
	 * Add a new entry containing the given item at the end of this list. The
	 * item may be null.
	 *
	 * @param  item  Item.
	 *
	 * @return  Entry that was added.
	 */
	public DListEntry<T> addLast
		(T item)
		{
		return addLast (new DListEntry<T> (item));
		}

	/**
	 * Add the given entry at the end of this list.
	 *
	 * @param  entry  Entry.
	 *
	 * @return  Entry that was added (namely <TT>entry</TT>).
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>entry</TT> is null.
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if <TT>entry</TT> is in a list.
	 */
	public DListEntry<T> addLast
		(DListEntry<T> entry)
		{
		if (entry == null)
			throw new NullPointerException
				("DList.addLast(): entry is null");
		if (entry.list != null)
			throw new IllegalStateException
				("DList.addLast(): entry is in a list");

		entry.list = this;
		entry.pred = sentinel.pred;
		entry.succ = sentinel;
		sentinel.pred.succ = entry;
		sentinel.pred = entry;

		return entry;
		}

	/**
	 * Perform the given action on each item in this list. For each item in this
	 * list from beginning to end, the given <TT>action</TT>'s <TT>run()</TT>
	 * method is called, passing in the item. Time: <I>O</I>(<I>n</I>).
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds items to or removes
	 * items from the list, the <TT>forEachItemDo()</TT> method's behavior is
	 * not specified.
	 *
	 * @param  action  Action.
	 */
	public void forEachItemDo
		(Action<T> action)
		{
		DListEntry<T> p, q;
		p = first();
		while (p != null)
			{
			q = p.succ();
			action.run (p.item);
			p = q;
			}
		}

	/**
	 * Perform the given action on each item in this list and return a result.
	 * For each item in this list from beginning to end, the given
	 * <TT>action</TT>'s <TT>run()</TT> method is called, passing in the item.
	 * After all the items have been processed, the given <TT>action</TT>'s
	 * <TT>result()</TT> method is called, and its result is returned. Time:
	 * <I>O</I>(<I>n</I>).
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
		(ActionResult<T,R> action)
		{
		DListEntry<T> p, q;
		p = first();
		while (p != null)
			{
			q = p.succ();
			action.run (p.item);
			p = q;
			}
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
		(Predicate<T> predicate)
		{
		DListEntry<T> p, q;
		p = first();
		while (p != null)
			{
			q = p.succ();
			if (predicate.test (p.item))
				p.remove();
			p = q;
			}
		}

	/**
	 * Perform the given action on each entry in this list. For each entry in
	 * this list from beginning to end, the given <TT>action</TT>'s
	 * <TT>run()</TT> method is called, passing in the entry. Time:
	 * <I>O</I>(<I>n</I>).
	 * <P>
	 * <B><I>Warning:</I></B> The <TT>action</TT> is permitted to remove the
	 * given entry. If the <TT>action</TT> adds entries to or removes other
	 * entries from the list, the <TT>forEachEntryDo()</TT> method's behavior is
	 * not specified.
	 *
	 * @param  action  Action.
	 */
	public void forEachEntryDo
		(Action<DListEntry<T>> action)
		{
		DListEntry<T> p, q;
		p = first();
		while (p != null)
			{
			q = p.succ();
			action.run (p);
			p = q;
			}
		}

	/**
	 * Perform the given action on each entry in this list and return a result.
	 * For each entry in this list from beginning to end, the given
	 * <TT>action</TT>'s <TT>run()</TT> method is called, passing in the entry.
	 * After all the entries have been processed, the given <TT>action</TT>'s
	 * <TT>result()</TT> method is called, and its result is returned. Time:
	 * <I>O</I>(<I>n</I>).
	 * <P>
	 * <B><I>Warning:</I></B> The <TT>action</TT> is permitted to remove the
	 * given entry. If the <TT>action</TT> adds entries to or removes other
	 * entries from the list, the <TT>forEachEntryDo()</TT> method's behavior is
	 * not specified.
	 *
	 * @param  <R>     Result data type.
	 * @param  action  Action.
	 *
	 * @return  Result of processing all the list entries.
	 */
	public <R> R forEachEntryDo
		(ActionResult<DListEntry<T>,R> action)
		{
		DListEntry<T> p, q;
		p = first();
		while (p != null)
			{
			q = p.succ();
			action.run (p);
			p = q;
			}
		return action.result();
		}

	/**
	 * Evaluate the given predicate on, and possibly remove, each entry in this
	 * list. For each entry in this list from beginning to end, the given
	 * <TT>predicate</TT>'s <TT>test()</TT> method is called, passing in the
	 * entry. If the <TT>test()</TT> method returns true, the entry is removed
	 * from this list. Time: <I>O</I>(<I>n</I>).
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>predicate</TT> adds entries to or
	 * removes entries from the list, other than by returning true, the
	 * <TT>removeEachEntryIf()</TT> method's behavior is not specified.
	 *
	 * @param  predicate  Predicate.
	 */
	public void removeEachEntryIf
		(Predicate<DListEntry<T>> predicate)
		{
		DListEntry<T> p, q;
		p = first();
		while (p != null)
			{
			q = p.succ();
			if (predicate.test (p))
				p.remove();
			p = q;
			}
		}

	/**
	 * Store this list's items in the given array. The first item is stored at
	 * index 0, the second item at index 1, and so on. The number of array
	 * elements set is <TT>array.length</TT>. If this list contains fewer than
	 * <TT>array.length</TT> items, the remaining array elements are set to
	 * null. Time: <I>O</I>(<I>n</I>).
	 *
	 * @param  array  Array in which to store items.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 */
	public T[] toArray
		(T[] array)
		{
		return toArray (array, 0, array.length);
		}

	/**
	 * Store this list's items in the given array. The first item is stored at
	 * index <TT>off</TT>, the second item at index <TT>off</TT>+1, and so on.
	 * The number of array elements set is <TT>len</TT>. If this list contains
	 * fewer than <TT>len</TT> items, the remaining array elements are set to
	 * null. Time: <I>O</I>(<I>n</I>).
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
	public T[] toArray
		(T[] array,
		 int off,
		 int len)
		{
		if (off < 0 || len < 0 || off+len > array.length)
			throw new IndexOutOfBoundsException();
		DListEntry<T> p = first();
		while (p != null && len > 0)
			{
			array[off] = p.item;
			++ off;
			-- len;
			p = p.succ();
			}
		while (len > 0)
			{
			array[off] = null;
			++ off;
			-- len;
			}
		return array;
		}

	/**
	 * Write this object's fields to the given out stream. The list items are
	 * written using {@link OutStream#writeReference(Object) writeReference()}.
	 *
	 * @param  out  Out stream.
	 *
	 * @exception  ClassCastException
	 *     (unchecked exception) Thrown if an item in this list is not
	 *     streamable.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeOut
		(OutStream out)
		throws IOException
		{
		int n = size();
		DListEntry<T> p = first();
		out.writeInt (n);
		for (int i = 0; i < n; ++ i)
			{
			out.writeReference (p.item);
			p = p.succ();
			}
		}

	/**
	 * Read this object's fields from the given in stream. The list items are
	 * read using {@link InStream#readReference() readReference()}.
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
		clear();
		int n = in.readInt();
		for (int i = 0; i < n; ++ i)
			addLast ((T) in.readReference());
		}

	}
