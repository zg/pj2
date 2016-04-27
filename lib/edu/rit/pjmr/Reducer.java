//******************************************************************************
//
// File:    Reducer.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.Reducer
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
 * Class Reducer is the abstract base class for a reducer in the Parallel Java
 * Map Reduce Framework. See class {@linkplain PjmrJob} for further information
 * about configuring reducers as part of a PJMR job.
 * <P>
 * A {@linkplain ReducerTask} calls a reducer's methods as follows:
 * <OL TYPE=1>
 * <P><LI>
 * The reducer task calls the reducer's {@link #start(String[]) start()} method.
 * The arguments are the reducer's argument strings if any. The <TT>start()</TT>
 * method may initialize the reducer based on the argument strings.
 * <P><LI>
 * The reducer task repeatedly calls the reducer's {@link #reduce(Object,Vbl)
 * reduce()} method. The arguments are a (key, value) pair from a {@linkplain
 * Combiner}. The combiner holds the combined contents of all the mappers'
 * combiners in the PJMR job. The <TT>reduce()</TT> method may do whatever it
 * likes with the (key, value) pair, such as doing computations on the pair,
 * printing the pair, or storing the pair in a file. The <TT>reduce()</TT>
 * method may alter the state of the value (but not the key).
 * <P><LI>
 * When all pairs have been processed, the reducer task calls the reducer's
 * {@link #finish() finish()} method. The <TT>finish()</TT> method may do
 * postprocessing operations.
 * </OL>
 * <P>
 * If a reducer task has more than one reducer, the reducers are run in parallel
 * in separate threads. There is one global combiner, and the reducers share the
 * work of processing the pairs in the combiner.
 *
 * @param  <K>  Output key data type.
 * @param  <V>  Output value data type; must implement interface {@linkplain
 *              edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 27-Nov-2013
 */
public abstract class Reducer<K,V extends Vbl>
	{

// Exported constructors.

	/**
	 * Construct a new reducer.
	 */
	public Reducer()
		{
		}

// Exported operations.

	/**
	 * Start this reducer.
	 * <P>
	 * The base class <TT>start()</TT> method does nothing. A subclass may
	 * override the <TT>start()</TT> method to do something.
	 *
	 * @param  args  Array of zero or more argument strings.
	 */
	public void start
		(String[] args)
		{
		}

	/**
	 * Reduce the given (key, value) pair. The state of the <TT>key</TT> must
	 * not be altered. The state of the <TT>value</TT> may be altered.
	 * <P>
	 * The <TT>reduce()</TT> method must be overridden in a subclass.
	 *
	 * @param  key     Key; non-null.
	 * @param  value   Value; may be null.
	 */
	public abstract void reduce
		(K key,
		 V value);

	/**
	 * Finish this reducer.
	 * <P>
	 * The base class <TT>finish()</TT> method does nothing. A subclass may
	 * override the <TT>finish()</TT> method to do something.
	 */
	public void finish()
		{
		}

	}
