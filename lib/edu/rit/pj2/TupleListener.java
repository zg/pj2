//******************************************************************************
//
// File:    TupleListener.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.TupleListener
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

package edu.rit.pj2;

/**
 * Class TupleListener is the abstract base class for an object that is notified
 * when a certain tuple is put into tuple space. To add a tuple listener in a
 * {@linkplain Task}, call the task's {@link
 * Task#addTupleListener(TupleListener) addTupleListener()} method.
 * <P>
 * When a tuple listener is constructed, two things are specified:
 * <OL TYPE=1>
 * <P><LI>
 * A <B>template.</B> The tuple listener will be triggered when a tuple that
 * matches the template is put into tuple space.
 * <P><LI>
 * The <B>operation</B> to be performed when the tuple listener is triggered,
 * one of the following:
 * <UL>
 * <P><LI>
 * <B>Read</B> (constructor <TT>take</TT> argument = false) -- A copy of the
 * matching tuple is passed to the tuple listener's {@link #run(Tuple) run()}
 * method. The original matching tuple remains in tuple space.
 * <P><LI>
 * <B>Take</B> (constructor <TT>take</TT> argument = true) -- The matching tuple
 * is taken out of tuple space and is passed to the tuple listener's {@link
 * #run(Tuple) run()} method.
 * </UL>
 * </OL>
 * <P>
 * The {@link #run(Tuple) run()} method must be overridden in a subclass to
 * perform the desired actions on the matching tuple.
 * <P>
 * Once a tuple listener has been triggered, the tuple listener will not be
 * triggered again. If the tuple listener is to be triggered again when another
 * matching tuple appears, re-add the tuple listener to the task. This only
 * makes sense for a tuple listener with the take operation.
 *
 * @param  <T>  Data type of the template and matching tuple.
 *
 * @author  Alan Kaminsky
 * @version 25-May-2015
 */
public abstract class TupleListener<T extends Tuple>
	{

// Hidden data members.

	T template;
	boolean take;

// Exported constructors.

	/**
	 * Construct a new tuple listener with the given template. The tuple
	 * listener's operation is read.
	 *
	 * @param  template  Template.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>template</TT> is null.
	 */
	public TupleListener
		(T template)
		{
		this (template, false);
		}

	/**
	 * Construct a new tuple listener with the given template and operation.
	 *
	 * @param  template  Template.
	 * @param  take      True to do a take, false to do a read.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>template</TT> is null.
	 */
	public TupleListener
		(T template,
		 boolean take)
		{
		if (template == null)
			throw new NullPointerException
				("TupleListener(): template is null");
		this.template = template;
		this.take = take;
		}

// Exported operations.

	/**
	 * Perform the desired actions on the given matching tuple.
	 *
	 * @param  tuple  Matching tuple.
	 *
	 * @exception  Exception
	 *     The <TT>run()</TT> method may throw any exception.
	 */
	public abstract void run
		(T tuple)
		throws Exception;

	}
