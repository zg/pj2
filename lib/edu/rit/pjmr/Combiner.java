//******************************************************************************
//
// File:    Combiner.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.Combiner
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

package edu.rit.pjmr;

import edu.rit.pj2.Vbl;
import edu.rit.pj2.TerminateException;
import edu.rit.util.Action;
import edu.rit.util.Instance;
import edu.rit.util.Map;
import edu.rit.util.Pair;

/**
 * Class Combiner provides a combiner in the Parallel Java Map Reduce Framework.
 * See class {@linkplain PjmrJob} for further information about configuring
 * combiners as part of a PJMR job.
 * <P>
 * A combiner is a mapping from keys to values. The key can be any object that
 * is suitable for use as a key in a hashed data structure. The value must be an
 * object that implements interface {@linkplain edu.rit.pj2.Vbl Vbl}, which is
 * the interface for an object that supports reduction. The basic operation is
 * {@link #add(Object,Vbl) add(key,value)}; this absorbs the given (key, value)
 * pair into the combiner by reducing the given value into the value associated
 * with the given key inside the combiner.
 * <P>
 * To be suitable for use as a key in a hashed data structure: (1) The key
 * object's class must define the <TT>equals()</TT> and <TT>hashCode()</TT>
 * methods in the proper manner. (2) When an object is used as a key, the state
 * of the object must not be altered.
 * <P>
 * A combiner is itself a {@linkplain edu.rit.pj2.Vbl Vbl}. The {@link
 * #reduce(Vbl) reduce()} method absorbs every (key, value) pair in a given
 * combiner into this combiner.
 * <P>
 * <I>Note:</I> Class Combiner is not multiple thread safe. Any necessary thread
 * synchronization is handled by the caller.
 *
 * @param  <K>  Key data type.
 * @param  <V>  Value data type; must implement interface {@linkplain
 *              edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 13-Jul-2015
 */
public class Combiner<K,V extends Vbl>
	extends Map<K,V>
	implements Vbl
	{

// Exported constructors.

	/**
	 * Construct a new combiner.
	 */
	public Combiner()
		{
		super();
		}

	/**
	 * Construct a new combiner that is a copy of the given combiner. Tthe new
	 * combiner's keys are set to the key references in the given combiner. The
	 * new combiner's values are set to clones of the values in the given
	 * combiner.
	 *
	 * @param  combiner  Combiner to copy.
	 */
	public Combiner
		(Combiner<K,V> combiner)
		{
		super (combiner);
		}

// Exported operations.

	/**
	 * Add the given (key, value) pair into this combiner. If the <TT>key</TT>
	 * does not yet exist in this combiner, a new pair (<TT>key</TT>,
	 * <TT>v</TT>) is first added, where <TT>v</TT> is the value returned by the
	 * {@link #initialValue(Object,Vbl) initialValue()} method. If the given
	 * <TT>value</TT> is non-null, the <TT>value</TT> is reduced into the value
	 * associated with the <TT>key</TT> inside this combiner (either the
	 * existing value, or the newly created value).
	 *
	 * @param  key    Key; must be non-null.
	 * @param  value  Value; may be null.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>key</TT> is null. Thrown if
	 *     <TT>value</TT> is non-null but the value associated with <TT>key</TT>
	 *     inside this combiner is null.
	 */
	public void add
		(K key,
		 V value)
		{
		if (! contains (key))
			put (key, initialValue (key, value));
		if (value != null)
			get (key) .reduce (value);
		}

	/**
	 * Create a clone of this shared variable. The clone contains references to
	 * the keys in this combiner. The clone contains clones of the values in
	 * this combiner.
	 *
	 * @return  The cloned object; a combiner.
	 */
	public Object clone()
		{
		return super.clone();
		}

	/**
	 * Set this shared variable to the given shared variable. This combiner's
	 * keys are set to the key references in the given combiner. This combiner's
	 * values are set to clones of the values in the given combiner.
	 *
	 * @param  vbl  Shared variable; a combiner.
	 *
	 * @exception  ClassCastException
	 *     (unchecked exception) Thrown if the class of <TT>vbl</TT> is not
	 *     compatible with the class of this shared variable.
	 */
	public void set
		(Vbl vbl)
		{
		copy ((Combiner<K,V>)vbl);
		}

	/**
	 * Reduce the given shared variable into this shared variable. The two
	 * combiners are combined together, and the result is stored in this
	 * combiner.
	 *
	 * @param  vbl  Shared variable; a combiner.
	 *
	 * @exception  ClassCastException
	 *     (unchecked exception) Thrown if the class of <TT>vbl</TT> is not
	 *     compatible with the class of this shared variable.
	 */
	public void reduce
		(Vbl vbl)
		{
		((Combiner<K,V>)vbl).forEachItemDo (new Action<Pair<K,V>>()
			{
			public void run (Pair<K,V> pair)
				{
				add (pair.key(), pair.value());
				}
			});
		}

// Hidden operations.

	/**
	 * Copy the given value. The <TT>copyValue()</TT> method in class Combiner
	 * returns a clone of the given value.
	 *
	 * @param  value  Value to copy.
	 *
	 * @return  Copy of <TT>value</TT>.
	 */
	protected V copyValue
		(V value)
		{
		return (V) value.clone();
		}

	/**
	 * The {@link #add(Object,Vbl) add()} method calls the
	 * <TT>initialValue()</TT> method when a certain key is being added to this
	 * combiner for the first time. The <TT>initialValue()</TT> method returns a
	 * new object which is the initial value to be associated with the new key.
	 * <P>
	 * The <TT>initialValue()</TT> method in class Combiner does the following.
	 * If <TT>value</TT> is null, null is returned. Otherwise, a new instance of
	 * the <TT>value</TT>'s class is returned; the new instance is created by
	 * the class's no-argument constructor. If this is not the desired behavior,
	 * write a subclass of class Combiner and override the
	 * <TT>initialValue()</TT> method.
	 *
	 * @param  key    Key that is being added; non-null.
	 * @param  value  Value that is being added; may be null.
	 *
	 * @return  Initial value to be associated with the <TT>key</TT>, or null.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if the <TT>value</TT>'s class does not
	 *     have a no-argument constructor, or if an instance of that class
	 *     cannot be created.
	 */
	protected V initialValue
		(K key,
		 V value)
		{
		try
			{
			return value == null ?
				null :
				(V) Instance.newDefaultInstance (value.getClass(), true);
			}
		catch (Throwable exc)
			{
			throw new IllegalArgumentException
				("Combiner.initialValue(): Cannot create initial value",
				 exc);
			}
		}

	}
