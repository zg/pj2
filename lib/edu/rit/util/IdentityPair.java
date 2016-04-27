//******************************************************************************
//
// File:    IdentityPair.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.IdentityPair
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
 * Class IdentityPair is a {@linkplain Pair Pair} with reference equality
 * semantics. Two IdentityPairs are equal if their keys refer to the same
 * object. That is, equality of keys is tested using the <TT>==</TT> operator
 * rather than the <TT>equals()</TT> method.
 * <P>
 * The key may be null. Once a key is stored in a pair, the state of the key
 * must not be altered. The value may be any object. The value may be null. Once
 * a value is stored in a pair, the state of the value may be altered, and the
 * value may be replaced by a different object.
 * <P>
 * Class IdentityPair is streamable. For object streaming to work properly, the
 * key class and the value class must be streamable or serializable.
 *
 * @param  <K>  Key data type.
 * @param  <V>  Value data type.
 *
 * @author  Alan Kaminsky
 * @version 18-Jan-2015
 */
public class IdentityPair<K,V>
	extends Pair<K,V>
	{

// Exported constructors.

	/**
	 * Construct a new pair. The key and the value are null.
	 */
	public IdentityPair()
		{
		super();
		}

	/**
	 * Construct a new pair with the given key and value.
	 *
	 * @param  key    Key.
	 * @param  value  Value.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>key</TT> is null.
	 */
	public IdentityPair
		(K key,
		 V value)
		{
		super (key, value);
		}

// Exported operations.

	/**
	 * Determine if this pair is equal to the given object. The pairs are equal
	 * if their keys refer to the same object, as determined by the <TT>==</TT>
	 * operator.
	 *
	 * @param  obj  Object to test.
	 *
	 * @return  True if this pair is equal to <TT>obj</TT>, false otherwise.
	 */
	public boolean equals
		(Object obj)
		{
		return
			(obj instanceof PairBase) &&
			(this.key == ((PairBase)obj).key);
		}

	/**
	 * Returns a hash code for this pair. The hash code is that of the key, as
	 * returned by the {@link System#identityHashCode(Object)
	 * System.identityHashCode()} method.
	 *
	 * @return  Hash code.
	 */
	public int hashCode()
		{
		return System.identityHashCode (key);
		}

	}
