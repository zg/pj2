//******************************************************************************
//
// File:    PairBase.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.PairBase
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

import edu.rit.io.Streamable;

/**
 * Class PairBase is the abstract base class for a pair consisting of a key and
 * its associated value. The key is defined in the base class; the value is
 * defined in a subclass. The key must be an object suitable for use in a hashed
 * data structure; that is, the key class must define the {@link
 * Object#equals(Object) equals()} and {@link Object#hashCode() hashCode()}
 * methods properly. The key may be null. Once a key is stored in a pair, the
 * state of the key must not be altered.
 * <P>
 * Class PairBase is streamable. For object streaming to work properly, the key
 * class must be streamable or serializable.
 *
 * @param  <K>  Key data type.
 *
 * @author  Alan Kaminsky
 * @version 18-Jan-2015
 */
public abstract class PairBase<K>
	implements Streamable
	{

// Hidden data members.

	/**
	 * The pair's key.
	 */
	protected K key;

// Exported constructors.

	/**
	 * Construct a new pair. The key is null.
	 */
	public PairBase()
		{
		}

	/**
	 * Construct a new pair with the given key.
	 *
	 * @param  key  Key.
	 */
	public PairBase
		(K key)
		{
		this.key = key;
		}

// Exported operations.

	/**
	 * Returns this pair's key.
	 *
	 * @return  Key.
	 */
	public K key()
		{
		return key;
		}

	/**
	 * Determine if this pair is equal to the given object. The pairs are equal
	 * if their keys are equal, as determined by the key class's {@link
	 * Object#equals(Object) equals()} method.
	 *
	 * @param  obj  Object to test.
	 *
	 * @return  True if this pair is equal to <TT>obj</TT>, false otherwise.
	 */
	public boolean equals
		(Object obj)
		{
		if (obj instanceof PairBase)
			{
			PairBase pair = (PairBase)obj;
			return
				(this.key == null && pair.key == null) ||
				(this.key != null && this.key.equals (pair.key));
			}
		else
			return false;
		}

	/**
	 * Returns a hash code for this pair. The hash code is that of the key, as
	 * returned by the key class's {@link Object#hashCode() hashCode()} method.
	 *
	 * @return  Hash code.
	 */
	public int hashCode()
		{
		return key == null ? 0 : key.hashCode();
		}

	}
