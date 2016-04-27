//******************************************************************************
//
// File:    Customizer.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.Customizer
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

package edu.rit.pjmr;

import edu.rit.pj2.Vbl;

/**
 * Class Customizer is the base class for a customizer in the Parallel Java Map
 * Reduce Framework. See class {@linkplain PjmrJob} for further information
 * about configuring customizers as part of a PJMR job.
 * <P>
 * A {@linkplain MapperTask} calls a customizer's methods as follows:
 * <OL TYPE=1>
 * <P><LI>
 * The mapper task calls the customizer's {@link #start(String[],Combiner)
 * start()} method, in a single thread, before running the {@linkplain Mapper}s.
 * The arguments are the customizer's argument strings if any, and the
 * {@linkplain Combiner} that will accumulate the mappers' results. The
 * <TT>start()</TT> method may initialize the customizer based on the argument
 * strings. The <TT>start()</TT> method may do preprocessing operations on the
 * combiner.
 * <P><LI>
 * The mapper task calls the customizer's {@link #finish(Combiner) finish()}
 * method, in a single thread, after all the {@linkplain Mapper}s have finished.
 * The argument is the {@linkplain Combiner}. The <TT>finish()</TT> method may
 * do postprocessing operations on the combiner.
 * </OL>
 * <P>
 * A {@linkplain ReducerTask} calls a customizer's methods as follows:
 * <OL TYPE=1>
 * <P><LI>
 * The reducer task calls the customizer's {@link #start(String[],Combiner)
 * start()} method, in a single thread, before running the {@linkplain
 * Reducer}s. The arguments are the customizer's argument strings if any, and
 * the {@linkplain Combiner} that contains all the mappers' results. The
 * <TT>start()</TT> method may initialize the customizer based on the argument
 * strings. The <TT>start()</TT> method may do preprocessing operations on the
 * combiner.
 * <P><LI>
 * If the customizer's {@link #comesBefore(Object,Vbl,Object,Vbl) comesBefore()}
 * method has been overridden, the reducer task sorts the combiner's (key,
 * value) pairs, in a single thread, before running the {@linkplain Reducer}s.
 * The reducer task calls the customizer's {@link
 * #comesBefore(Object,Vbl,Object,Vbl) comesBefore()} method to determine the
 * ordering of the pairs. The reducer task passes the pairs to the {@linkplain
 * Reducer}s in the resulting sorted order. If the customizer's {@link
 * #comesBefore(Object,Vbl,Object,Vbl) comesBefore()} method has not been
 * overridden, the reducer task does not sort the combiner's pairs, and the
 * reducer task passes the pairs to the {@linkplain Reducer}s in an unspecified
 * order.
 * <P><LI>
 * The reducer task calls the customizer's {@link #finish(Combiner) finish()}
 * method, in a single thread, after all the {@linkplain Reducer}s have
 * finished. The argument is the {@linkplain Combiner}. The <TT>finish()</TT>
 * method may do postprocessing operations on the combiner.
 * </OL>
 *
 * @param  <K>  Mapper output key data type.
 * @param  <V>  Mapper output value data type; must implement interface
 *              {@linkplain edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 28-Nov-2013
 */
public class Customizer <K,V extends Vbl>
	{

// Exported constructors.

	/**
	 * Construct a new customizer.
	 */
	public Customizer()
		{
		}

// Exported operations.

	/**
	 * Start this customizer.
	 * <P>
	 * The base class <TT>start()</TT> method does nothing. A subclass may
	 * override the <TT>start()</TT> method to do something.
	 *
	 * @param  args      Array of zero or more argument strings.
	 * @param  combiner  Combiner.
	 */
	public void start
		(String[] args,
		 Combiner<K,V> combiner)
		{
		}

	/**
	 * Determine if the first (key, value) pair comes before the second (key,
	 * value) pair in the desired sorted order.
	 * <P>
	 * The Customizer base class does not sort the pairs. A subclass may
	 * override the <TT>comesBefore()</TT> method to sort the pairs into the
	 * desired order.
	 *
	 * @param  key_1    Key from first pair.
	 * @param  value_1  Value from first pair.
	 * @param  key_2    Key from second pair.
	 * @param  value_2  Value from second pair.
	 *
	 * @return  True if the first pair comes before the second pair, false
	 *          otherwise.
	 */
	public boolean comesBefore
		(K key_1,
		 V value_1,
		 K key_2,
		 V value_2)
		{
		throw new UnsupportedOperationException();
		}

	/**
	 * Finish this customizer.
	 * <P>
	 * The base class <TT>finish()</TT> method does nothing. A subclass may
	 * override the <TT>finish()</TT> method to do something.
	 *
	 * @param  combiner  Combiner.
	 */
	public void finish
		(Combiner<K,V> combiner)
		{
		}

	}
