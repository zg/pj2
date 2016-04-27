//******************************************************************************
//
// File:    DListEntry.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.DListEntry
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

/**
 * Class DListEntry contains one item stored in a {@linkplain DList DList}.
 * <P>
 * Operations take constant time unless otherwise specified. <I>n</I> is the
 * number of items stored in the list.
 *
 * @param  <T>  List item data type.
 *
 * @author  Alan Kaminsky
 * @version 06-Jun-2013
 */
public class DListEntry<T>
	{

// Hidden data members.

	T item;
	DList<T> list;
	DListEntry<T> pred;
	DListEntry<T> succ;

// Hidden constructors.

	/**
	 * Construct a new entry.
	 *
	 * @param  item  This entry's item.
	 */
	DListEntry
		(T item)
		{
		this.item = item;
		}

// Exported operations.

	/**
	 * Get the item contained in this entry. The item may be null.
	 *
	 * @return  Item.
	 */
	public T item()
		{
		return item;
		}

	/**
	 * Set the item contained in this entry. The item may be null.
	 *
	 * @param  item  Item.
	 */
	public void item
		(T item)
		{
		this.item = item;
		}

	/**
	 * Get the list that contains this entry. If this entry is not in a list,
	 * null is returned.
	 *
	 * @return  List, or null.
	 */
	public DList<T> list()
		{
		return list;
		}

	/**
	 * Get the entry that comes before this entry. If this entry is the first in
	 * its list, or if this entry is not part of a list, null is returned.
	 *
	 * @return  Predecessor entry, or null.
	 */
	public DListEntry<T> pred()
		{
		return pred == null || pred.list == null ? null : pred;
		}

	/**
	 * Get the entry that comes after this entry. If this entry is the last in
	 * its list, or if this entry is not part of a list, null is returned.
	 *
	 * @return  Successor entry, or null.
	 */
	public DListEntry<T> succ()
		{
		return succ == null || succ.list == null ? null : succ;
		}

	/**
	 * Add a new entry containing the given item before this entry in this
	 * entry's list. The item may be null.
	 *
	 * @param  item  Item.
	 *
	 * @return  Entry that was added.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this entry is not in a list.
	 */
	public DListEntry<T> addBefore
		(T item)
		{
		return addBefore (new DListEntry<T> (item));
		}

	/**
	 * Add the given entry before this entry in this entry's list.
	 *
	 * @param  entry  Entry.
	 *
	 * @return  Entry that was added (namely <TT>entry</TT>).
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>entry</TT> is null.
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if <TT>entry</TT> is in a list. Thrown
	 *     if this entry is not in a list.
	 */
	public DListEntry<T> addBefore
		(DListEntry<T> entry)
		{
		if (entry == null)
			throw new NullPointerException
				("DListEntry.addBefore(): entry is null");
		if (entry.list != null)
			throw new IllegalStateException
				("DListEntry.addBefore(): entry is in a list");
		if (this.list == null)
			throw new IllegalStateException
				("DListEntry.addBefore(): this is not in a list");

		entry.list = this.list;
		entry.pred = this.pred;
		entry.succ = this;
		this.pred.succ = entry;
		this.pred = entry;

		return entry;
		}

	/**
	 * Add a new entry containing the given item after this entry in this
	 * entry's list. The item may be null.
	 *
	 * @param  item  Item.
	 *
	 * @return  Entry that was added.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this entry is not in a list.
	 */
	public DListEntry<T> addAfter
		(T item)
		{
		return addAfter (new DListEntry<T> (item));
		}

	/**
	 * Add the given entry after this entry in this entry's list.
	 *
	 * @param  entry  Entry.
	 *
	 * @return  Entry that was added (namely <TT>entry</TT>).
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>entry</TT> is null.
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if <TT>entry</TT> is in a list. Thrown
	 *     if this entry is not in a list.
	 */
	public DListEntry<T> addAfter
		(DListEntry<T> entry)
		{
		if (entry == null)
			throw new NullPointerException
				("DListEntry.addAfter(): entry is null");
		if (entry.list != null)
			throw new IllegalStateException
				("DListEntry.addAfter(): entry is in a list");
		if (this.list == null)
			throw new IllegalStateException
				("DListEntry.addAfter(): this is not in a list");

		entry.list = this.list;
		entry.pred = this;
		entry.succ = this.succ;
		this.succ.pred = entry;
		this.succ = entry;

		return entry;
		}

	/**
	 * Remove this entry from this entry's list. Afterwards, this entry is not
	 * in a list, this entry has no predecessor, and this entry has no
	 * successor.
	 *
	 * @return  Entry that was removed (namely this entry).
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this entry is not in a list.
	 */
	public DListEntry<T> remove()
		{
		if (this.list == null)
			throw new IllegalStateException
				("DListEntry.remove(): this is not in a list");

		this.pred.succ = this.succ;
		this.succ.pred = this.pred;
		this.list = null;
		this.pred = null;
		this.succ = null;

		return this;
		}

// Hidden operations.

	/**
	 * Clear this entry.
	 */
	void clear()
		{
		this.item = null;
		this.list = null;
		this.pred = null;
		this.succ = null;
		}

	}
