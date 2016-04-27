//******************************************************************************
//
// File:    Set.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.Set
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

/**
 * Class Set provides a set of elements. The element must be an object suitable
 * for use in a hashed data structure; that is, the element class must define
 * the {@link Object#equals(Object) equals()} and {@link Object#hashCode()
 * hashCode()} methods properly. Elements must not be null. Once an element is
 * stored in a set, the state of the element must not be altered.
 * <P>
 * The set calls the protected {@link #copyElement(Object) copyElement()} method
 * to make a copy of an element being added to the set. The default is to copy
 * just the element reference. This behavior can be changed by overriding the
 * {@link #copyElement(Object) copyElement()} method in a subclass of class Set.
 * <P>
 * The set uses the protected {@link #same(Object,Object) same()} and {@link
 * #hashOf(Object) hashOf()} methods when searching for an element. The default
 * is to use the element class's {@link Object#equals(Object) equals()} and
 * {@link Object#hashCode() hashCode()} methods. This behavior can be changed by
 * overriding the {@link #same(Object,Object) same()} and {@link #hashOf(Object)
 * hashOf()} methods in a subclass of class Set.
 * <P>
 * Class Set is streamable. For object streaming to work properly, the element
 * class must be streamable or serializable.
 * <P>
 * <I>Note:</I> Class Set is not multiple thread safe.
 *
 * @param  <E>  Element data type.
 *
 * @author  Alan Kaminsky
 * @version 13-Jul-2015
 */
public class Set<E>
	implements Streamable, Cloneable
	{

// Hidden data members.

	private int nelem;   // Number of elements
	private int nmax;    // Maximum number of elements
	private int thresh;  // Threshold for growing table = nmax*3/4
	private E[] table;   // Table of elements

// Exported constructors.

	/**
	 * Construct a new empty set.
	 */
	public Set()
		{
		nelem = 0;
		nmax = 8;
		thresh = 6;
		table = (E[]) new Object [nmax];
		}

	/**
	 * Construct a new set that is a copy of the given set.
	 * <P>
	 * The elements of the given set are copied using the {@link
	 * #copyElement(Object) copyElement()} method.
	 *
	 * @param  set  Set to copy.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>set</TT> is null.
	 */
	public Set
		(Set<E> set)
		{
		copy (set);
		}

// Exported operations.

	/**
	 * Determine if this set is empty.
	 *
	 * @return  True if this set is empty, false otherwise.
	 */
	public boolean isEmpty()
		{
		return nelem == 0;
		}

	/**
	 * Clear this set.
	 */
	public void clear()
		{
		nelem = 0;
		for (int i = 0; i < nmax; ++ i)
			table[i] = null;
		}

	/**
	 * Change this set to be a copy of the given set.
	 * <P>
	 * The elements of the given set are copied using the {@link
	 * #copyElement(Object) copyElement()} method.
	 *
	 * @param  set  Set to copy.
	 *
	 * @return  This set.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>set</TT> is null.
	 */
	public Set<E> copy
		(Set<E> set)
		{
		clear();
		for (int i = 0; i < set.nmax; ++ i)
			if (set.table[i] != null)
				add (set.table[i]);
		return this;
		}

	/**
	 * Create a clone of this set. The clone is a new set that contains copies
	 * of the elements in this set.
	 * <P>
	 * The elements of this set are copied using the {@link #copyElement(Object)
	 * copyElement()} method.
	 *
	 * @return  New set.
	 */
	public Object clone()
		{
		try
			{
			Set<E> s = (Set<E>) super.clone();
			s.table = (E[]) new Object [s.nmax];
			s.copy (this);
			return s;
			}
		catch (CloneNotSupportedException exc)
			{
			throw new IllegalStateException ("Shouldn't happen", exc);
			}
		}

	/**
	 * Returns the number of elements in this set.
	 *
	 * @return  Number of elements.
	 */
	public int size()
		{
		return nelem;
		}

	/**
	 * Determine if this set contains the given element.
	 *
	 * @param  elem  Element.
	 *
	 * @return  True if this set contains <TT>elem</TT>, false otherwise.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>elem</TT> is null.
	 */
	public boolean contains
		(E elem)
		{
		int i = hash (elem);
		while (table[i] != null)
			if (same (table[i], elem))
				return true;
			else
				i = (i + 1) % nmax;
		return false;
		}

	/**
	 * Get the given element in this set. This method returns a reference to the
	 * actual element in this set that is equal to <TT>elem</TT>; this might or
	 * might not be the same object as <TT>elem</TT>. If this set does not
	 * contain such an element, null is returned.
	 *
	 * @param  elem  Element.
	 *
	 * @return  Element in this set that is equal to <TT>elem</TT>, or null if
	 *          this set does not contain such an element.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>elem</TT> is null.
	 */
	public E get
		(E elem)
		{
		int i = hash (elem);
		while (table[i] != null)
			if (same (table[i], elem))
				return table[i];
			else
				i = (i + 1) % nmax;
		return null;
		}

	/**
	 * Add the given element to this set.
	 * <P>
	 * If not already present, a copy of the given element, created using the
	 * {@link #copyElement(Object) copyElement()} method, is added to this set.
	 *
	 * @param  elem  Element.
	 *
	 * @return  True if this set did not previously contain <TT>elem</TT>, false
	 *          otherwise.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>elem</TT> is null.
	 */
	public boolean add
		(E elem)
		{
		int i = hash (elem);
		while (table[i] != null)
			if (same (table[i], elem))
				return false;
			else
				i = (i + 1) % nmax;
		table[i] = copyElement (elem);
		++ nelem;
		expandTable();
		return true;
		}

	/**
	 * Remove the given element from this set.
	 *
	 * @param  elem  Element.
	 *
	 * @return  True if this set previously contained <TT>elem</TT>, false
	 *          otherwise.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>elem</TT> is null.
	 */
	public boolean remove
		(E elem)
		{
		int i = hash (elem);
		while (table[i] != null)
			if (same (table[i], elem))
				{
				table[i] = null;
				-- nelem;
				rehash();
				return true;
				}
			else
				i = (i + 1) % nmax;
		return false;
		}

	/**
	 * Determine if this set is a subset of the given set. This is so if every
	 * element of this set is also an element of the given set.
	 *
	 * @param  set  Set.
	 *
	 * @return  True if this set is a subset of the given set, false otherwise.
	 */
	public boolean isSubsetOf
		(Set<E> set)
		{
		for (int i = 0; i < nmax; ++ i)
			if (table[i] != null && ! set.contains (table[i]))
				return false;
		return true;
		}

	/**
	 * Change this set to be the union of itself and the given set. The union
	 * consists of all elements that appear in this set or the given set or
	 * both.
	 *
	 * @param  set  Set.
	 */
	public void union
		(Set<E> set)
		{
		for (int i = 0; i < set.nmax; ++ i)
			if (set.table[i] != null)
				add (set.table[i]);
		}

	/**
	 * Change this set to be the intersection of itself and the given set. The
	 * intersection consists of all elements that appear in this set and the
	 * given set.
	 *
	 * @param  set  Set.
	 */
	public void intersection
		(Set<E> set)
		{
		for (int i = 0; i < nmax; ++ i)
			if (table[i] != null && ! set.contains (table[i]))
				{
				table[i] = null;
				-- nelem;
				}
		rehash();
		}

	/**
	 * Change this set to be the difference of itself and the given set. The
	 * difference consists of all elements that appear in this set and not in
	 * the given set.
	 *
	 * @param  set  Set.
	 */
	public void difference
		(Set<E> set)
		{
		for (int i = 0; i < nmax; ++ i)
			if (table[i] != null && set.contains (table[i]))
				{
				table[i] = null;
				-- nelem;
				}
		rehash();
		}

	/**
	 * Change this set to be the symmetric difference of itself and the given
	 * set. The symmetric difference consists of all elements that appear in
	 * this set or the given set, but not both.
	 *
	 * @param  set  Set.
	 */
	public void symmetricDifference
		(Set<E> set)
		{
		for (int i = 0; i < set.nmax; ++ i)
			if (set.table[i] == null)
				{
				}
			else if (this.contains (set.table[i]))
				this.remove (set.table[i]);
			else
				this.add (set.table[i]);
		}

	/**
	 * Perform the given action on each element in this set. For each element in
	 * this set in an unspecified order, the given <TT>action</TT>'s
	 * <TT>run()</TT> method is called, passing in the element. The state of the
	 * element must not be altered.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds elements to or removes
	 * elements from the set, the <TT>forEachItemDo()</TT> method's behavior is
	 * unspecified.
	 *
	 * @param  action  Action.
	 */
	public void forEachItemDo
		(Action<E> action)
		{
		for (int i = 0; i < nmax; ++ i)
			if (table[i] != null)
				action.run (table[i]);
		}

	/**
	 * Perform the given action on each element in this set and return a result.
	 * For each element in this set in an unspecified order, the given
	 * <TT>action</TT>'s <TT>run()</TT> method is called, passing in the
	 * element. The state of the element must not be altered. After all the
	 * elements have been processed, the given <TT>action</TT>'s
	 * <TT>result()</TT> method is called, and its result is returned.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds elements to or removes
	 * elements from the set, the <TT>forEachItemDo()</TT> method's behavior is
	 * unspecified.
	 *
	 * @param  <R>     Result data type.
	 * @param  action  Action.
	 *
	 * @return  Result of processing all the set elements.
	 */
	public <R> R forEachItemDo
		(ActionResult<E,R> action)
		{
		for (int i = 0; i < nmax; ++ i)
			if (table[i] != null)
				action.run (table[i]);
		return action.result();
		}

	/**
	 * Evaluate the given predicate on, and possibly remove, each element in
	 * this set. For each element in this set in an unspecified order, the given
	 * <TT>predicate</TT>'s <TT>test()</TT> method is called, passing in the
	 * element. The state of the element must not be altered. If the
	 * <TT>test()</TT> method returns true, the element is removed from this
	 * set.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>predicate</TT> adds elements to or
	 * removes elements from the set, other than by returning true, the
	 * <TT>removeEachItemIf()</TT> method's behavior is not specified.
	 *
	 * @param  predicate  Predicate.
	 */
	public void removeEachItemIf
		(Predicate<E> predicate)
		{
		for (int i = 0; i < nmax; ++ i)
			if (table[i] != null && predicate.test (table[i]))
				{
				table[i] = null;
				-- nelem;
				}
		rehash();
		}

	/**
	 * Store this set's elements in the given array. The elements are stored in
	 * an unspecified order. The first element is stored at index 0, the second
	 * element at index 1, and so on. The number of array elements stored is
	 * <TT>array.length</TT>. If this set contains fewer than
	 * <TT>array.length</TT> elements, the remaining array elements are set to
	 * null.
	 * <P>
	 * The <TT>toArray()</TT> method stores <I>references</I> to the set
	 * elements in the given array. The states of the set elements stored in the
	 * array must not be altered.
	 *
	 * @param  array  Array in which to store elements.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 */
	public E[] toArray
		(E[] array)
		{
		return toArray (array, 0, array.length);
		}

	/**
	 * Store this set's elements in a portion of the given array. The elements
	 * are stored in an unspecified order. The first element is stored at index
	 * <TT>off</TT>, the second element at index <TT>off</TT>+1, and so on. The
	 * number of array elements stored is <TT>len</TT>. If this set contains
	 * fewer than <TT>len</TT> elements, the remaining array elements are set to
	 * null.
	 * <P>
	 * The <TT>toArray()</TT> method stores <I>references</I> to the set
	 * elements in the given array. The states of the set elements stored in the
	 * array must not be altered.
	 *
	 * @param  array  Array in which to store elements.
	 * @param  off    Index at which to store first element.
	 * @param  len    Number of elements to store.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>array.length</TT>.
	 */
	public E[] toArray
		(E[] array,
		 int off,
		 int len)
		{
		if (off < 0 || len < 0 || off+len > array.length)
			throw new IndexOutOfBoundsException();
		for (int i = 0; i < nmax && len > 0; ++ i)
			if (table[i] != null)
				{
				array[off] = table[i];
				++ off;
				-- len;
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
	 * Write this object's fields to the given out stream. The set elements are
	 * written using {@link OutStream#writeObject(Object) writeObject()}.
	 *
	 * @param  out  Out stream.
	 *
	 * @exception  IOException
	 *     Thrown if an element in this set is not streamable or serializable.
	 *     Thrown if an I/O error occurred.
	 */
	public void writeOut
		(OutStream out)
		throws IOException
		{
		out.writeInt (nelem);
		for (int i = 0; i < nmax; ++ i)
			if (table[i] != null)
				out.writeObject (table[i]);
		}

	/**
	 * Read this object's fields from the given in stream. The set elements are
	 * read using {@link InStream#readObject() readObject()}.
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
			add ((E) in.readObject());
		}

// Hidden operations.

	/**
	 * Copy the given element. The <TT>copyElement()</TT> method in class Set
	 * merely returns the <TT>elem</TT> reference; this makes a <I>shallow
	 * copy.</I> A subclass of class Set can override the <TT>copyElement()</TT>
	 * method to return something else. Possibilities include:
	 * <UL>
	 * <P><LI>
	 * Return a clone of the element, which makes a <I>deep copy.</I>
	 * <P><LI>
	 * Return an instance of a subclass of the element's class, in which the
	 * {@link Streamable#writeOut(OutStream) writeOut()} and {@link
	 * Streamable#readIn(InStream) readIn()} methods have been overridden to
	 * implement different streaming behavior.
	 * </UL>
	 *
	 * @param  elem  Set element to copy.
	 *
	 * @return  Copy of <TT>elem</TT>.
	 */
	protected E copyElement
		(E elem)
		{
		return elem;
		}

	/**
	 * Determine if the given elements are the same. The <TT>same()</TT> method
	 * in class Set returns <TT>elem1.equals(elem2)</TT>. A subclass of class
	 * Set can override the <TT>same()</TT> method to use a different criterion
	 * for whether the given elements are the same.
	 * <P>
	 * If the <TT>same()</TT> method says that two elements are the same, then
	 * the {@link #hashOf(Object) hashOf()} method must return the same hash
	 * value for both elements.
	 *
	 * @param  elem1  First element; an element in this set.
	 * @param  elem2  Second element; an element that might or might not be in
	 *                this set.
	 */
	protected boolean same
		(E elem1,
		 E elem2)
		{
		return elem1.equals (elem2);
		}

	/**
	 * Returns a hash value for the given element. The <TT>hashOf()</TT> method
	 * in class Set returns <TT>elem.hashCode()</TT>. A subclass of class Set
	 * can override the <TT>hashOf()</TT> method to calculate the hash value
	 * differently.
	 * <P>
	 * If the {@link #same(Object,Object) same()} method says that two elements
	 * are the same, then the <TT>hashOf()</TT> method must return the same hash
	 * value for both elements.
	 *
	 * @param  elem  Element.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>elem</TT> is null.
	 */
	protected int hashOf
		(E elem)
		{
		return elem.hashCode();
		}

	/**
	 * Returns the hash table index of the given element.
	 */
	private int hash
		(E elem)
		{
		return hash (elem, nmax);
		}

	/**
	 * Returns the hash table index of the given element for the given table
	 * size.
	 */
	private int hash
		(E elem,
		 int n)
		{
		int h = hashOf (elem) % n;
		return h >= 0 ? h : h + n;
		}

	/**
	 * Expand the table when the number of elements reaches the threshold.
	 */
	private void expandTable()
		{
		// If threshold is not reached, do nothing.
		if (nelem < thresh) return;

		// Increase table size by 3/2.
		int nmaxNew = (nmax*3 + 1)/2;
		E[] tableNew = (E[]) new Object [nmaxNew];

		// Place all set elements into new table.
		for (int j = 0; j < nmax; ++ j)
			if (table[j] != null)
				{
				int i = hash (table[j], nmaxNew);
				while (tableNew[i] != null)
					i = (i + 1) % nmaxNew;
				tableNew[i] = table[j];
				}

		// Replace old table with new table.
		nmax = nmaxNew;
		thresh = (nmax*3 + 3)/4;
		table = tableNew;
		}

	/**
	 * Rehash the existing elements into the table.
	 */
	private void rehash()
		{
		E[] tableNew = (E[]) new Object [nmax];

		// Place all set elements into new table.
		for (int j = 0; j < nmax; ++ j)
			if (table[j] != null)
				{
				int i = hash (table[j]);
				while (tableNew[i] != null)
					i = (i + 1) % nmax;
				tableNew[i] = table[j];
				}

		// Replace old table with new table.
		table = tableNew;
		}

	}
