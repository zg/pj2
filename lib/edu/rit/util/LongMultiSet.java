//******************************************************************************
//
// File:    LongMultiSet.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.LongMultiSet
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
 * Class LongMultiSet provides a multiset of elements. A multiset is a set in
 * which each element may appear more than once. An element's
 * <I>multiplicity</I> (a long integer greater than or equal to 0) is the number
 * of times the element appears in the multiset. The element must be an object
 * suitable for use in a hashed data structure; that is, the element class must
 * define the {@link Object#equals(Object) equals()} and {@link
 * Object#hashCode() hashCode()} methods properly. Elements must not be null.
 * Once an element is stored in a multiset, the state of the element must not be
 * altered.
 * <P>
 * The multiset calls the protected {@link #getPair(Object,long) getPair()}
 * method to make an element-multiplicity pair being added to the multiset. The
 * multiset calls the protected {@link #copyElement(Object) copyElement()}
 * method to make a copy of an element being added to the multiset. The default
 * is to copy just the element reference. This behavior can be changed by
 * overriding the aforementioned methods in a subclass of class LongMultiSet.
 * <P>
 * Class LongMultiSet is streamable. For object streaming to work properly, the
 * element class must be streamable or serializable.
 * <P>
 * <I>Note:</I> Class LongMultiSet is not multiple thread safe.
 *
 * @param  <E>  Element data type.
 *
 * @author  Alan Kaminsky
 * @version 10-Jan-2015
 */
public class LongMultiSet<E>
	implements Streamable
	{

// Hidden data members.

	private Set<LongPair<E>> elements = new Set<LongPair<E>>()
		{
		protected LongPair<E> copyElement (LongPair<E> elem)
			{
			return getPair (elem.key(), elem.value());
			}
		};

// Exported constructors.

	/**
	 * Construct a new empty multiset.
	 */
	public LongMultiSet()
		{
		}

	/**
	 * Construct a new multiset that is a copy of the given multiset.
	 * <P>
	 * The element-multiplicity pairs of the given multiset are copied using the
	 * {@link #getPair(Object,long) getPair()} method.
	 *
	 * @param  mset  Multiset to copy.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>mset</TT> is null.
	 */
	public LongMultiSet
		(LongMultiSet<E> mset)
		{
		copy (mset);
		}

// Exported operations.

	/**
	 * Determine if this multiset is empty.
	 *
	 * @return  True if this multiset is empty, false otherwise.
	 */
	public boolean isEmpty()
		{
		return elements.isEmpty();
		}

	/**
	 * Clear this multiset.
	 */
	public void clear()
		{
		elements.clear();
		}

	/**
	 * Change this multiset to be a copy of the given multiset.
	 * <P>
	 * The element-multiplicity pairs of the given multiset are copied using the
	 * {@link #getPair(Object,long) getPair()} method.
	 *
	 * @param  mset  Multiset to copy.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>mset</TT> is null.
	 */
	public void copy
		(LongMultiSet<E> mset)
		{
		elements.copy (mset.elements);
		}

	/**
	 * Returns the number of unique elements in this multiset. This is the
	 * number of different elements, disregarding their multiplicities.
	 *
	 * @return  Number of unique elements.
	 */
	public int size()
		{
		return elements.size();
		}

	/**
	 * Returns the number of elements in this multiset. This is the sum of the
	 * elements' multiplicities.
	 *
	 * @return  Number of elements.
	 */
	public long countAll()
		{
		return elements.forEachItemDo (new ActionResult<LongPair<E>,Long>()
			{
			long total;
			public void run (LongPair<E> pair)
				{
				total += pair.value();
				}
			public Long result()
				{
				return total;
				}
			});
		}

	/**
	 * Determine if this multiset contains the given element.
	 *
	 * @param  elem  Element.
	 *
	 * @return  True if this multiset contains <TT>elem</TT> at least once,
	 *          false otherwise.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>elem</TT> is null.
	 */
	public boolean contains
		(E elem)
		{
		return elements.contains (new LongPair<E> (elem, 0));
		}

	/**
	 * Determine the multiplicity of the given element in this multiset.
	 *
	 * @param  elem  Element.
	 *
	 * @return  Number of occurrences of <TT>elem</TT> in this multiset (&ge;
	 *          0).
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>elem</TT> is null.
	 */
	public long count
		(E elem)
		{
		LongPair<E> pair = elements.get (new LongPair<E> (elem, 0));
		return pair == null ? 0 : pair.value();
		}

	/**
	 * Add one occurrence of the given element to this multiset.
	 * <P>
	 * If not already present, an element-multiplicity pair, created using the
	 * {@link #getPair(Object,long) getPair()} method, is added to this
	 * multiset.
	 *
	 * @param  elem  Element.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>elem</TT> is null.
	 */
	public void add
		(E elem)
		{
		add (elem, 1);
		}

	/**
	 * Add the given number of occurrences of the given element to this
	 * multiset.
	 * <P>
	 * If not already present, an element-multiplicity pair, created using the
	 * {@link #getPair(Object,long) getPair()} method, is added to this
	 * multiset.
	 *
	 * @param  elem   Element.
	 * @param  count  Number of occurrences to add (&ge; 0).
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>elem</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>count</TT> &lt; 0.
	 */
	public void add
		(E elem,
		 long count)
		{
		if (count < 0)
			throw new IllegalArgumentException (String.format
				("LongMultiSet.add(): count = %d illegal", count));
		LongPair<E> newPair = new LongPair<E> (elem, count);
		LongPair<E> oldPair = elements.get (newPair);
		if (oldPair == null)
			elements.add (newPair);
		else
			oldPair.value (oldPair.value() + count);
		}

	/**
	 * Remove one occurrence of the given element from this multiset. If this
	 * multiset does not contain any occurrences of <TT>elem</TT>, then
	 * afterwards this multiset is unchanged.
	 *
	 * @param  elem  Element.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>elem</TT> is null.
	 */
	public void remove
		(E elem)
		{
		remove (elem, 1);
		}

	/**
	 * Remove the given number of occurrences of the given element from this
	 * multiset. If this multiset contains <TT>count</TT> or fewer occurrences
	 * of <TT>elem</TT>, then afterwards this multiset does not contain
	 * <TT>elem</TT>.
	 *
	 * @param  elem   Element.
	 * @param  count  Number of occurrences to remove (&ge; 0).
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>elem</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>count</TT> &lt; 0.
	 */
	public void remove
		(E elem,
		 long count)
		{
		if (count < 0)
			throw new IllegalArgumentException (String.format
				("LongMultiSet.remove(): count = %d illegal", count));
		LongPair<E> pair = elements.get (new LongPair<E> (elem, 0));
		if (pair == null)
			{ }
		else if (pair.value() <= count)
			elements.remove (pair);
		else
			pair.value (pair.value() - count);
		}

	/**
	 * Remove all occurrences of the given element from this multiset.
	 *
	 * @param  elem   Element.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>elem</TT> is null.
	 */
	public void removeAll
		(E elem)
		{
		elements.remove (new LongPair<E> (elem, 0));
		}

	/**
	 * Determine if this multiset is a subset of the given multiset. This is so
	 * if every element of this multiset is also an element of the given
	 * multiset.
	 *
	 * @param  mset  Multiset.
	 *
	 * @return  True if this multiset is a subset of the given multiset, false
	 *          otherwise.
	 */
	public boolean isSubsetOf
		(final LongMultiSet<E> mset)
		{
		return elements.forEachItemDo (new ActionResult<LongPair<E>,Boolean>()
			{
			boolean isSubset = true;
			public void run (LongPair<E> pair)
				{
				isSubset &= (pair.value() <= mset.count (pair.key()));
				}
			public Boolean result()
				{
				return isSubset;
				}
			});
		}

	/**
	 * Change this multiset to be the union of itself and the given multiset.
	 * The union consists of all elements that appear in this set or the given
	 * set or both.
	 *
	 * @param  mset  Multiset.
	 */
	public void union
		(LongMultiSet<E> mset)
		{
		mset.elements.forEachItemDo (new Action<LongPair<E>>()
			{
			public void run (LongPair<E> msetpair)
				{
				add (msetpair.key(), msetpair.value());
				}
			});
		}

	/**
	 * Change this multiset to be the intersection of itself and the given
	 * multiset. The intersection consists of all elements that appear in this
	 * multiset and the given multiset.
	 *
	 * @param  mset  Multiset.
	 */
	public void intersection
		(final LongMultiSet<E> mset)
		{
		this.elements.removeEachItemIf (new Predicate<LongPair<E>>()
			{
			public boolean test (LongPair<E> thispair)
				{
				thispair.value
					(Math.min (thispair.value(), mset.count (thispair.key())));
				return thispair.value() <= 0;
				}
			});
		}

	/**
	 * Change this multiset to be the difference of itself and the given
	 * multiset. The difference consists of all elements that appear in this
	 * multiset and not in the given multiset.
	 *
	 * @param  mset  Multiset.
	 */
	public void difference
		(final LongMultiSet<E> mset)
		{
		this.elements.removeEachItemIf (new Predicate<LongPair<E>>()
			{
			public boolean test (LongPair<E> thispair)
				{
				thispair.value (thispair.value() - mset.count (thispair.key()));
				return thispair.value() <= 0;
				}
			});
		}

	/**
	 * Change this multiset to be the symmetric difference of itself and the
	 * given multiset. The symmetric difference consists of all elements that
	 * appear in this multiset or the given multiset, but not both.
	 *
	 * @param  mset  Multiset.
	 */
	public void symmetricDifference
		(LongMultiSet<E> mset)
		{
		mset.elements.forEachItemDo (new Action<LongPair<E>>()
			{
			public void run (LongPair<E> msetpair)
				{
				LongPair<E> thispair = elements.get (msetpair);
				if (thispair == null)
					add (msetpair.key(), msetpair.value());
				else if (thispair.value() > msetpair.value())
					thispair.value (thispair.value() - msetpair.value());
				else
					removeAll (msetpair.key());
				}
			});
		}

	/**
	 * Perform the given action on each element in this multiset. For each
	 * element in this multiset in an unspecified order, the given
	 * <TT>action</TT>'s <TT>run()</TT> method is called, passing in a pair
	 * consisting of the element (key) and the element's multiplicity (value).
	 * The state of the element must not be altered. The multiplicity may be
	 * altered. If the multiplicity is set to a value &le; 0, the element is
	 * removed from this multiset.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds elements to or removes
	 * elements from the multiset, other than by setting the multiplicity, the
	 * <TT>forEachItemDo()</TT> method's behavior is unspecified.
	 *
	 * @param  action  Action.
	 */
	public void forEachItemDo
		(final Action<LongPair<E>> action)
		{
		elements.removeEachItemIf (new Predicate<LongPair<E>>()
			{
			public boolean test (LongPair<E> pair)
				{
				action.run (pair);
				return pair.value() <= 0;
				}
			});
		}

	/**
	 * Perform the given action on each element in this multiset and return a
	 * result. For each element in this multiset in an unspecified order, the
	 * given <TT>action</TT>'s <TT>run()</TT> method is called, passing in a
	 * pair consisting of the element (key) and the element's multiplicity
	 * (value). The state of the element must not be altered. The multiplicity
	 * may be altered. If the multiplicity is set to a value &le; 0, the element
	 * is removed from this multiset. After all the elements have been
	 * processed, the given <TT>action</TT>'s <TT>result()</TT> method is
	 * called, and its result is returned.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds elements to or removes
	 * elements from the multiset, other than by setting the multiplicity, the
	 * <TT>forEachItemDo()</TT> method's behavior is unspecified.
	 *
	 * @param  <R>     Result data type.
	 * @param  action  Action.
	 *
	 * @return  Result of processing all the set elements.
	 */
	public <R> R forEachItemDo
		(final ActionResult<LongPair<E>,R> action)
		{
		elements.removeEachItemIf (new Predicate<LongPair<E>>()
			{
			public boolean test (LongPair<E> pair)
				{
				action.run (pair);
				return pair.value() <= 0;
				}
			});
		return action.result();
		}

	/**
	 * Store this multiset's elements in the given array. Each unique element is
	 * stored as a pair consisting of the element (key) and the element's
	 * multiplicity (value). The elements are stored in an unspecified order.
	 * The first element is stored at index 0, the second element at index 1,
	 * and so on. The number of array elements stored is <TT>array.length</TT>.
	 * If this multiset contains fewer than <TT>array.length</TT> unique
	 * elements, the remaining array elements are set to null.
	 * <P>
	 * The <TT>toArray()</TT> method stores <I>references</I> to this multiset's
	 * element-multiplicity pairs in the given array. The states of the elements
	 * stored in the array must not be altered. The multiplicities stored in the
	 * array must not be altered.
	 *
	 * @param  array  Array in which to store elements.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 */
	public LongPair<E>[] toArray
		(LongPair<E>[] array)
		{
		return toArray (array, 0, array.length);
		}

	/**
	 * Store this multiset's elements in the given array. Each unique element is
	 * stored as a pair consisting of the element (key) and the element's
	 * multiplicity (value). The elements are stored in an unspecified order.
	 * The first element is stored at index <TT>off</TT>, the second element at
	 * index <TT>off</TT>+1, and so on. The number of array elements stored is
	 * <TT>len</TT>. If this multiset contains fewer than <TT>len</TT> unique
	 * elements, the remaining array elements are set to null.
	 * <P>
	 * The <TT>toArray()</TT> method stores <I>references</I> to this multiset's
	 * element-multiplicity pairs in the given array. The states of the elements
	 * stored in the array must not be altered. The multiplicities stored in the
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
	public LongPair<E>[] toArray
		(LongPair<E>[] array,
		 int off,
		 int len)
		{
		return elements.toArray (array, off, len);
		}

	/**
	 * Write this object's fields to the given out stream. The multiset elements
	 * are written using {@link OutStream#writeObject(Object) writeObject()}.
	 * The elements' multiplicities are written using {@link
	 * OutStream#writeLong(long) writeLong()}.
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
		out.writeFields (elements);
		}

	/**
	 * Read this object's fields from the given in stream. The multiset elements
	 * are read using {@link InStream#readObject() readObject()}. The elements'
	 * multiplicities are written using {@link InStream#readLong() readLong()}.
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
		in.readFields (elements);
		}

// Hidden operations.

	/**
	 * Create a pair containing the given element and multiplicity. Whenever
	 * class LongMultiSet needs to create a pair, it does so by calling this
	 * method.
	 * <P>
	 * The <TT>getPair()</TT> method in class LongMultiSet returns an instance
	 * of class {@linkplain LongPair LongPair}&lt;E&gt;. The given element is
	 * copied using the {@link #copyElement(Object) copyElement()} method. A
	 * subclass of class LongMultiSet can override the <TT>getPair()</TT> method
	 * to return something else. Possibilities include:
	 * <UL>
	 * <P><LI>
	 * Return an instance of a subclass of class {@link LongPair
	 * LongPair}&lt;E&gt;, in which the {@link Object#equals(Object) equals()}
	 * and {@link Object#hashCode() hashCode()} methods have been overridden to
	 * implement a different pair equality criterion.
	 * <P><LI>
	 * Return an instance of a subclass of class {@link LongPair
	 * LongPair}&lt;E&gt;, in which the {@link Streamable#writeOut(OutStream)
	 * writeOut()} and {@link Streamable#readIn(InStream) readIn()} methods
	 * have been overridden to implement different streaming behavior.
	 * </UL>
	 *
	 * @param  key    Key (element).
	 * @param  value  Value (multiplicity).
	 *
	 * @return  Pair consisting of (key, value).
	 */
	protected LongPair<E> getPair
		(E key,
		 long value)
		{
		return new LongPair<E> (copyElement (key), value);
		}

	/**
	 * Copy the given element. The <TT>copyElement()</TT> method in class
	 * LongMultiSet merely returns the <TT>elem</TT> reference; this makes a
	 * <I>shallow copy.</I> A subclass of class LongMultiSet can override the
	 * <TT>copyElement()</TT> method to return something else. Possibilities
	 * include:
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

	}
