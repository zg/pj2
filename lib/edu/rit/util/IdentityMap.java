//******************************************************************************
//
// File:    IdentityMap.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.IdentityMap
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

/**
 * Class IdentityMap provides a mapping from keys to values with reference
 * equality semantics. Two (key, value) mappings are equal if their keys refer
 * to the same object. That is, equality of keys is tested using the <TT>==</TT>
 * operator rather than the <TT>equals()</TT> method.
 * <P>
 * Keys must not be null. Once a key is stored in a map, the state of the key
 * must not be altered. The value may be any object. Values may be null. Once a
 * value is stored in a map, the state of the value may be altered, and the
 * value may be replaced by a different object.
 * <P>
 * The map calls the protected {@link #getPair(Object,Object) getPair()} method
 * to make (key, value) mappings (pairs). The default is to copy just the key
 * and value references. This behavior can be changed by overriding the {@link
 * #getPair(Object,Object) getPair()} method in a subclass.
 * <P>
 * Class IdentityMap is streamable. For object streaming to work properly, the
 * key and the value classes must be streamable or serializable.
 * <P>
 * <I>Note:</I> Class IdentityMap is not multiple thread safe.
 *
 * @param  <K>  Key data type.
 * @param  <V>  Value data type.
 *
 * @author  Alan Kaminsky
 * @version 09-Jan-2015
 */
public class IdentityMap<K,V>
	extends Map<K,V>
	{

// Exported constructors.

	/**
	 * Construct a new empty map.
	 */
	public IdentityMap()
		{
		}

	/**
	 * Construct a new map that is a copy of the given map.
	 *
	 * @param  map  Map to copy.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>map</TT> is null.
	 */
	public IdentityMap
		(Map<K,V> map)
		{
		copy (map);
		}

// Hidden operations.

	/**
	 * Create a pair containing the given key and value. Whenever class
	 * IdentityMap needs to create a pair, it does so by calling this method.
	 * <P>
	 * The <TT>getPair()</TT> method in class IdentityMap returns an instance of
	 * class {@linkplain IdentityPair IdentityPair}&lt;K,V&gt; consisting of the
	 * given <TT>key</TT> and <TT>value</TT> references.
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
		return new IdentityPair<K,V> (key, value);
		}

	}
