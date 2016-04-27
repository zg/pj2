//******************************************************************************
//
// File:    Map.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.Map
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
 * Class Map provides a mapping from keys to values. The key must be an object
 * suitable for use in a hashed data structure; that is, the key class must
 * define the {@link Object#equals(Object) equals()} and {@link
 * Object#hashCode() hashCode()} methods properly. Keys may be null. Once a
 * key is stored in a map, the state of the key must not be altered. The value
 * may be any object. Values may be null. Once a value is stored in a map, the
 * state of the value may be altered, and the value may be replaced by a
 * different object.
 * <P>
 * The map calls the protected {@link #getPair(Object,Object) getPair()} method
 * to make a key-value mapping (pair) being added to the map. The map calls the
 * protected {@link #copyKey(Object) copyKey()} method to make a copy of a key
 * being added to the map. The map calls the protected {@link #copyValue(Object)
 * copyValue()} method to make a copy of a value being added to the map. The
 * default is to copy just the key and value references. This behavior can be
 * changed by overriding the aforementioned methods in a subclass of class Map.
 * <P>
 * Class Map is streamable. For object streaming to work properly, the key and
 * the value classes must be streamable or serializable.
 * <P>
 * <I>Note:</I> Class Map is not multiple thread safe.
 *
 * @param  <K>  Key data type.
 * @param  <V>  Value data type.
 *
 * @author  Alan Kaminsky
 * @version 13-Jul-2015
 */
public class Map<K,V>
	implements Streamable, Cloneable
	{

// Hidden data members.

	private Set<Pair<K,V>> mapping = new Set<Pair<K,V>>()
		{
		protected Pair<K,V> copyElement (Pair<K,V> elem)
			{
			return getPair (elem.key(), elem.value());
			}
		};

// Exported constructors.

	/**
	 * Construct a new empty map.
	 */
	public Map()
		{
		}

	/**
	 * Construct a new map that is a copy of the given map.
	 * <P>
	 * The mappings in the given map are copied using the {@link
	 * #getPair(Object,Object) getPair()} method.
	 *
	 * @param  map  Map to copy.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>map</TT> is null.
	 */
	public Map
		(Map<K,V> map)
		{
		copy (map);
		}

// Exported operations.

	/**
	 * Determine if this map is empty.
	 *
	 * @return  True if this map is empty, false otherwise.
	 */
	public boolean isEmpty()
		{
		return mapping.isEmpty();
		}

	/**
	 * Clear this map.
	 */
	public void clear()
		{
		mapping.clear();
		}

	/**
	 * Change this map to be a copy of the given map.
	 * <P>
	 * The mappings in the given map are copied using the {@link
	 * #getPair(Object,Object) getPair()} method.
	 *
	 * @param  map  Map to copy.
	 *
	 * @return  This map.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>map</TT> is null.
	 */
	public Map<K,V> copy
		(Map<K,V> map)
		{
		this.mapping.copy (map.mapping);
		return this;
		}

	/**
	 * Create a clone of this map. The clone is a new map that contains copies
	 * of the mappings in this map.
	 * <P>
	 * The mappings in this map are copied using the {@link
	 * #getPair(Object,Object) getPair()} method.
	 *
	 * @return  New set.
	 */
	public Object clone()
		{
		try
			{
			Map<K,V> m = (Map<K,V>) super.clone();
			m.mapping = (Set<Pair<K,V>>) this.mapping.clone();
			return m;
			}
		catch (CloneNotSupportedException exc)
			{
			throw new IllegalStateException ("Shouldn't happen", exc);
			}
		}

	/**
	 * Returns the number of key-value mappings in this map.
	 *
	 * @return  Number of mappings.
	 */
	public int size()
		{
		return mapping.size();
		}

	/**
	 * Determine if this map contains the given key.
	 *
	 * @param  key  Key.
	 *
	 * @return  True if this map contains <TT>key</TT>, false otherwise.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>key</TT> is null.
	 */
	public boolean contains
		(K key)
		{
		return mapping.contains (new Pair<K,V> (key, null));
		}

	/**
	 * Get the value associated with the given key in this map. If this map does
	 * not contain <TT>key</TT>, null is returned.
	 *
	 * @param  key  Key.
	 *
	 * @return  Value associated with <TT>key</TT>, or null.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>key</TT> is null.
	 */
	public V get
		(K key)
		{
		Pair<K,V> pair = mapping.get (new Pair<K,V> (key, null));
		return pair == null ? null : pair.value();
		}

	/**
	 * Get the key associated with the given value in this map. If this map does
	 * not contain <TT>value</TT>, null is returned.
	 * <P>
	 * The <TT>inverseGet()</TT> method performs a linear search of the
	 * key-value mappings in this map to find a mapping whose value is equal to
	 * the given value, as determined by the value class's {@link
	 * Object#equals(Object) equals()} method. If more than one mapping has the
	 * given value, one such mapping is chosen in an unspecified manner, and the
	 * chosen mapping's key is returned.
	 *
	 * @param  value  Value, or null.
	 *
	 * @return  Key associated with <TT>value</TT>, or null.
	 */
	public K inverseGet
		(final V value)
		{
		return mapping.forEachItemDo (new ActionResult<Pair<K,V>,K>()
			{
			private K key = null;
			public void run (Pair<K,V> pair)
				{
				if ((value == null && pair.value() == null) ||
					(value != null && pair.value().equals (value)))
						key = pair.key();
				}
			public K result()
				{
				return key;
				}
			});
		}

	/**
	 * Map the given key to the given value in this map.
	 * <P>
	 * A pair containing the given key and value, created using the {@link
	 * #getPair(Object,Object) getPair()} method, is stored in this map.
	 *
	 * @param  key    Key.
	 * @param  value  Value.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>key</TT> is null.
	 */
	public void put
		(K key,
		 V value)
		{
		Pair<K,V> pair = new Pair<K,V> (key, value);
		mapping.remove (pair);
		mapping.add (pair);
		}

	/**
	 * Remove the given key and its associated value from this map. The value
	 * formerly associated with <TT>key</TT> is returned. If this map does not
	 * contain <TT>key</TT>, this map is not altered and null is returned.
	 *
	 * @param  key  Key.
	 *
	 * @return  Value associated with <TT>key</TT>, or null.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>key</TT> is null.
	 */
	public V remove
		(K key)
		{
		Pair<K,V> pair = mapping.get (new Pair<K,V> (key, null));
		if (pair == null)
			return null;
		else
			{
			mapping.remove (pair);
			return pair.value();
			}
		}

	/**
	 * Perform the given action on each key-value mapping in this map. For each
	 * mapping in this map in an unspecified order, the given <TT>action</TT>'s
	 * <TT>run()</TT> method is called, passing in a pair containing the key and
	 * the value. The state of the key must not be altered. The state of the
	 * value may be altered, and the value may be replaced by a different
	 * object; such changes are reflected in this map.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds mappings to or removes
	 * mappings from the map, the <TT>forEachItemDo()</TT> method's behavior is
	 * unspecified.
	 *
	 * @param  action  Action.
	 */
	public void forEachItemDo
		(Action<Pair<K,V>> action)
		{
		mapping.forEachItemDo (action);
		}

	/**
	 * Perform the given action on each key-value mapping in this map and return
	 * a result. For each mapping in this map in an unspecified order, the given
	 * <TT>action</TT>'s <TT>run()</TT> method is called, passing in a pair
	 * containing the key and the value. The state of the key must not be
	 * altered. The state of the value may be altered, and the value may be
	 * replaced by a different object; such changes are reflected in this map.
	 * After all the mappings have been processed, the given <TT>action</TT>'s
	 * <TT>result()</TT> method is called, and its result is returned.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds mappings to or removes
	 * mappings from the map, the <TT>forEachItemDo()</TT> method's behavior is
	 * unspecified.
	 *
	 * @param  <R>     Result data type.
	 * @param  action  Action.
	 *
	 * @return  Result of processing all the mappings.
	 */
	public <R> R forEachItemDo
		(ActionResult<Pair<K,V>,R> action)
		{
		return mapping.forEachItemDo (action);
		}

	/**
	 * Evaluate the given predicate on, and possibly remove, each key-value
	 * mapping in this map. For each mapping in this map in an unspecified
	 * order, the given <TT>predicate</TT>'s <TT>test()</TT> method is called,
	 * passing in a pair containing the key and the value. The state of the key
	 * must not be altered. The state of the value may be altered, and the value
	 * may be replaced by a different object; such changes are reflected in this
	 * map. If the <TT>test()</TT> method returns true, the mapping is removed
	 * from this map.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>predicate</TT> adds mappings to or
	 * removes mappings from the map, other than by returning true, the
	 * <TT>removeEachItemIf()</TT> method's behavior is unspecified.
	 *
	 * @param  predicate  Predicate.
	 */
	public void removeEachItemIf
		(Predicate<Pair<K,V>> predicate)
		{
		mapping.removeEachItemIf (predicate);
		}

	/**
	 * Perform the given action on each key in this map. For each key in this
	 * map in an unspecified order, the given <TT>action</TT>'s <TT>run()</TT>
	 * method is called, passing in the key. The state of the key must not be
	 * altered.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds mappings to or removes
	 * mappings from the map, the <TT>forEachKeyDo()</TT> method's behavior is
	 * unspecified.
	 *
	 * @param  action  Action.
	 */
	public void forEachKeyDo
		(final Action<K> action)
		{
		mapping.forEachItemDo (new Action<Pair<K,V>>()
			{
			public void run (Pair<K,V> pair)
				{
				action.run (pair.key());
				}
			});
		}

	/**
	 * Perform the given action on each key in this map and return a result. For
	 * each key in this map in an unspecified order, the given <TT>action</TT>'s
	 * <TT>run()</TT> method is called, passing in the key. The state of the key
	 * must not be altered. After all the keys have been processed, the given
	 * <TT>action</TT>'s <TT>result()</TT> method is called, and its result is
	 * returned.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds mappings to or removes
	 * mappings from the map, the <TT>forEachKeyDo()</TT> method's behavior is
	 * unspecified.
	 *
	 * @param  <R>     Result data type.
	 * @param  action  Action.
	 *
	 * @return  Result of processing all the mappings.
	 */
	public <R> R forEachKeyDo
		(final ActionResult<K,R> action)
		{
		mapping.forEachItemDo (new Action<Pair<K,V>>()
			{
			public void run (Pair<K,V> pair)
				{
				action.run (pair.key());
				}
			});
		return action.result();
		}

	/**
	 * Evaluate the given predicate on, and possibly remove, each key in this
	 * map. For each key in this map in an unspecified order, the given
	 * <TT>predicate</TT>'s <TT>test()</TT> method is called, passing in the
	 * key. The state of the key must not be altered. If the <TT>test()</TT>
	 * method returns true, the key and its associated value are removed from
	 * this map.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>predicate</TT> adds mappings to or
	 * removes mappings from the map, other than by returning true, the
	 * <TT>removeEachKeyIf()</TT> method's behavior is unspecified.
	 *
	 * @param  predicate  Predicate.
	 */
	public void removeEachKeyIf
		(final Predicate<K> predicate)
		{
		mapping.removeEachItemIf (new Predicate<Pair<K,V>>()
			{
			public boolean test (Pair<K,V> pair)
				{
				return predicate.test (pair.key());
				}
			});
		}

	/**
	 * Perform the given action on each value in this map. For each value in
	 * this map in an unspecified order, the given <TT>action</TT>'s
	 * <TT>run()</TT> method is called, passing in the value. The state of the
	 * value may be altered; such changes are reflected in this map.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds mappings to or removes
	 * mappings from the map, the <TT>forEachValue()</TT> method's behavior is
	 * unspecified.
	 *
	 * @param  action  Action.
	 */
	public void forEachValueDo
		(final Action<V> action)
		{
		mapping.forEachItemDo (new Action<Pair<K,V>>()
			{
			public void run (Pair<K,V> pair)
				{
				action.run (pair.value());
				}
			});
		}

	/**
	 * Perform the given action on each value in this map and return a result.
	 * For each value in this map in an unspecified order, the given
	 * <TT>action</TT>'s <TT>run()</TT> method is called, passing in the value.
	 * The state of the value may be altered; such changes are reflected in this
	 * map. After all the values have been processed, the given
	 * <TT>action</TT>'s <TT>result()</TT> method is called, and its result is
	 * returned.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds mappings to or removes
	 * mappings from the map, the <TT>forEachValueDo()</TT> method's behavior is
	 * unspecified.
	 *
	 * @param  <R>     Result data type.
	 * @param  action  Action.
	 *
	 * @return  Result of processing all the mappings.
	 */
	public <R> R forEachValueDo
		(final ActionResult<V,R> action)
		{
		mapping.forEachItemDo (new Action<Pair<K,V>>()
			{
			public void run (Pair<K,V> pair)
				{
				action.run (pair.value());
				}
			});
		return action.result();
		}

	/**
	 * Evaluate the given predicate on, and possibly remove, each value in this
	 * map. For each value in this map in an unspecified order, the given
	 * <TT>predicate</TT>'s <TT>test()</TT> method is called, passing in the
	 * value. The state of the value may be altered; such changes are reflected
	 * in this map. If the <TT>test()</TT> method returns true, the value and
	 * its associated key are removed from this map.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>predicate</TT> adds mappings to or
	 * removes mappings from the map, other than by returning true, the
	 * <TT>removeEachValueIf()</TT> method's behavior is unspecified.
	 *
	 * @param  predicate  Predicate.
	 */
	public void removeEachValueIf
		(final Predicate<V> predicate)
		{
		mapping.removeEachItemIf (new Predicate<Pair<K,V>>()
			{
			public boolean test (Pair<K,V> pair)
				{
				return predicate.test (pair.value());
				}
			});
		}

	/**
	 * Store this map's key-value mappings in the given array. The mappings are
	 * stored in an unspecified order. The first mapping is stored at index 0,
	 * the second mapping at index 1, and so on. The number of array elements
	 * stored is <TT>array.length</TT>. If this map contains fewer than
	 * <TT>array.length</TT> mappings, the remaining array elements are set to
	 * null.
	 * <P>
	 * The <TT>toArray()</TT> method stores <I>references</I> to this map's
	 * mappings in the given array. The states of the keys stored in the array
	 * must not be altered. The states of the values stored in the array may be
	 * altered, and the values stored in the array may be replaced by different
	 * objects; such changes are reflected in this map.
	 *
	 * @param  array  Array in which to store mappings.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 */
	public Pair<K,V>[] toArray
		(Pair<K,V>[] array)
		{
		return mapping.toArray (array, 0, array.length);
		}

	/**
	 * Store this map's key-value mappings in a portion of the given array. The
	 * mappings are stored in an unspecified order. The first mapping is stored
	 * at index <TT>off</TT>, the second mapping at index <TT>off</TT>+1, and so
	 * on. The number of array elements stored is <TT>len</TT>. If this map
	 * contains fewer than <TT>len</TT> mappings, the remaining array elements
	 * are set to null.
	 * <P>
	 * The <TT>toArray()</TT> method stores <I>references</I> to this map's
	 * mappings in the given array. The states of the keys stored in the array
	 * must not be altered. The states of the values stored in the array may be
	 * altered, and the values stored in the array may be replaced by different
	 * objects; such changes are reflected in this map.
	 *
	 * @param  array  Array in which to store mappings.
	 * @param  off    Index at which to store first mapping.
	 * @param  len    Number of mappings to store.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>array.length</TT>.
	 */
	public Pair<K,V>[] toArray
		(Pair<K,V>[] array,
		 int off,
		 int len)
		{
		return mapping.toArray (array, off, len);
		}

	/**
	 * Store this map's keys in the given array. The keys are stored in an
	 * unspecified order. The first key is stored at index 0, the second key at
	 * index 1, and so on. The number of array elements stored is
	 * <TT>array.length</TT>. If this map contains fewer than
	 * <TT>array.length</TT> mappings, the remaining array elements are set to
	 * null.
	 * <P>
	 * The <TT>keysToArray()</TT> method stores <I>references</I> to this map's
	 * keys in the given array. The states of the keys stored in the array must
	 * not be altered.
	 *
	 * @param  array  Array in which to store keys.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 */
	public K[] keysToArray
		(K[] array)
		{
		return keysToArray (array, 0, array.length);
		}

	/**
	 * Store this map's keys in a portion of the given array. The keys are
	 * stored in an unspecified order. The first key is stored at index
	 * <TT>off</TT>, the second key at index <TT>off</TT>+1, and so on. The
	 * number of array elements stored is <TT>len</TT>. If this map contains
	 * fewer than <TT>len</TT> mappings, the remaining array elements are set to
	 * null.
	 * <P>
	 * The <TT>keysToArray()</TT> method stores <I>references</I> to this map's
	 * keys in the given array. The states of the keys stored in the array must
	 * not be altered.
	 *
	 * @param  array  Array in which to store keys.
	 * @param  off    Index at which to store first key.
	 * @param  len    Number of keys to store.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>array.length</TT>.
	 */
	public K[] keysToArray
		(final K[] array,
		 final int off,
		 final int len)
		{
		mapping.forEachItemDo (new Action<Pair<K,V>>()
			{
			int aoff = off;
			int alen = len;
			public void run (Pair<K,V> pair)
				{
				if (alen > 0)
					{
					array[aoff] = pair.key();
					++ aoff;
					-- alen;
					}
				}
			});
		for (int i = size(); i < len; ++ i)
			array[off+i] = null;
		return array;
		}

	/**
	 * Store this map's values in the given array. The values are stored in an
	 * unspecified order. The first value is stored at index 0, the second value
	 * at index 1, and so on. The number of array elements stored is
	 * <TT>array.length</TT>. If this map contains fewer than
	 * <TT>array.length</TT> mappings, the remaining array elements are set to
	 * null.
	 * <P>
	 * The <TT>valuesToArray()</TT> method stores <I>references</I> to this
	 * map's values in the given array. The states of the values stored in the
	 * array may be altered; such changes are reflected in this map.
	 *
	 * @param  array  Array in which to store values.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 */
	public V[] valuesToArray
		(V[] array)
		{
		return valuesToArray (array, 0, array.length);
		}

	/**
	 * Store this map's values in a portion of the given array. The values are
	 * stored in an unspecified order. The first value is stored at index
	 * <TT>off</TT>, the second value at index <TT>off</TT>+1, and so on. The
	 * number of array elements stored is <TT>len</TT>. If this map contains
	 * fewer than <TT>len</TT> mappings, the remaining array elements are set to
	 * null.
	 * <P>
	 * The <TT>valuesToArray()</TT> method stores <I>references</I> to this
	 * map's values in the given array. The states of the values stored in the
	 * array may be altered; such changes are reflected in this map.
	 *
	 * @param  array  Array in which to store values.
	 * @param  off    Index at which to store first value.
	 * @param  len    Number of values to store.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>array.length</TT>.
	 */
	public V[] valuesToArray
		(final V[] array,
		 final int off,
		 final int len)
		{
		mapping.forEachItemDo (new Action<Pair<K,V>>()
			{
			int aoff = off;
			int alen = len;
			public void run (Pair<K,V> pair)
				{
				if (alen > 0)
					{
					array[aoff] = pair.value();
					++ aoff;
					-- alen;
					}
				}
			});
		for (int i = size(); i < len; ++ i)
			array[off+i] = null;
		return array;
		}

	/**
	 * Write this object's fields to the given out stream. Each key-value
	 * mapping, an instance of type {@linkplain Pair Pair}, is written using
	 * {@link OutStream#writeObject(Object) writeObject()}.
	 *
	 * @param  out  Out stream.
	 *
	 * @exception  IOException
	 *     Thrown if a key or value in this set is not streamable or
	 *     serializable. Thrown if an I/O error occurred.
	 */
	public void writeOut
		(OutStream out)
		throws IOException
		{
		out.writeFields (mapping);
		}

	/**
	 * Read this object's fields from the given in stream. Each key-value
	 * mapping, an instance of type {@linkplain Pair Pair}, is read using {@link
	 * InStream#readObject() readObject()}.
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
		in.readFields (mapping);
		}

// Hidden operations.

	/**
	 * Create a pair containing the given key and value. Whenever class Map
	 * needs to create a pair, it does so by calling this method.
	 * <P>
	 * The <TT>getPair()</TT> method in class Map returns an instance of class
	 * {@linkplain Pair Pair}&lt;K,V&gt;. The given key is copied using the
	 * {@link #copyKey(Object) copyKey()} method. The given value is copied
	 * using the {@link #copyValue(Object) copyValue()} method. A subclass of
	 * class Map can override the <TT>getPair()</TT> method to return something
	 * else. Possibilities include:
	 * <UL>
	 * <P><LI>
	 * Return an instance of a subclass of class {@link Pair Pair}&lt;K,V&gt;,
	 * in which the {@link Object#equals(Object) equals()} and {@link
	 * Object#hashCode() hashCode()} methods have been overridden to implement a
	 * different pair equality criterion.
	 * <P><LI>
	 * Return an instance of a subclass of class {@link Pair Pair}&lt;K,V&gt;,
	 * in which the {@link Streamable#writeOut(OutStream) writeOut()} and {@link
	 * Streamable#readIn(InStream) readIn()} methods have been overridden to
	 * implement different streaming behavior.
	 * </UL>
	 *
	 * @param  key    Key.
	 * @param  value  Value.
	 *
	 * @return  Pair consisting of (key, value).
	 */
	protected Pair<K,V> getPair
		(K key,
		 V value)
		{
		return new Pair<K,V> (copyKey (key), copyValue (value));
		}

	/**
	 * Copy the given key. The <TT>copyKey()</TT> method in class Map merely
	 * returns the <TT>key</TT> reference; this makes a <I>shallow copy.</I> A
	 * subclass of class Map can override the <TT>copyKey()</TT> method to
	 * return something else. Possibilities include:
	 * <UL>
	 * <P><LI>
	 * Return a clone of the key, which makes a <I>deep copy.</I>
	 * <P><LI>
	 * Return an instance of a subclass of the key's class, in which the
	 * {@link Object#equals(Object) equals()} and {@link Object#hashCode()
	 * hashCode()} methods have been overridden to implement a different key
	 * equality criterion.
	 * <P><LI>
	 * Return an instance of a subclass of the key's class, in which the
	 * {@link Streamable#writeOut(OutStream) writeOut()} and {@link
	 * Streamable#readIn(InStream) readIn()} methods have been overridden to
	 * implement different streaming behavior.
	 * </UL>
	 *
	 * @param  key  Key to copy.
	 *
	 * @return  Copy of <TT>key</TT>.
	 */
	protected K copyKey
		(K key)
		{
		return key;
		}

	/**
	 * Copy the given value. The <TT>copyValue()</TT> method in class Map merely
	 * returns the <TT>value</TT> reference; this makes a <I>shallow copy.</I> A
	 * subclass of class Map can override the <TT>copyValue()</TT> method to
	 * return something else. Possibilities include:
	 * <UL>
	 * <P><LI>
	 * Return a clone of the value, which makes a <I>deep copy.</I>
	 * <P><LI>
	 * Return an instance of a subclass of the value's class, in which the
	 * {@link Streamable#writeOut(OutStream) writeOut()} and {@link
	 * Streamable#readIn(InStream) readIn()} methods have been overridden to
	 * implement different streaming behavior.
	 * </UL>
	 *
	 * @param  value  Value to copy.
	 *
	 * @return  Copy of <TT>value</TT>.
	 */
	protected V copyValue
		(V value)
		{
		return value;
		}

	}
