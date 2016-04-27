//******************************************************************************
//
// File:    Vbl.java
// Package: edu.rit.pj2
// Unit:    Interface edu.rit.pj2.Vbl
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

package edu.rit.pj2;

/**
 * Interface Vbl specifies the interface for a variable shared by multiple
 * threads executing a {@linkplain ParallelStatement}. Subclasses provide shared
 * variables of each primitive type. For a shared variable of a nonprimitive
 * type, define a subclass with the appropriate fields that implements interface
 * Vbl.
 * <P>
 * The shared variable classes support the <I>parallel reduction</I> pattern.
 * Each thread creates a thread-local copy of the shared variable by calling the
 * {@link Loop#threadLocal(Vbl) threadLocal()} method of class {@linkplain Loop}
 * or the {@link Section#threadLocal(Vbl) threadLocal()} method of class
 * {@linkplain Section}. Each thread performs operations on its own copy,
 * without needing to synchronize with the other threads. At the end of the
 * parallel statement, the thread-local copies are automatically <I>reduced</I>
 * together, and the result is stored in the original shared variable. The
 * reduction is performed by the shared variable's {@link #reduce(Vbl) reduce()}
 * method.
 * <P>
 * Interface Vbl extends interface {@linkplain java.lang.Cloneable Cloneable}. A
 * subclass must implement the {@link #clone() clone()} method to create a
 * <B><I>deep copy</I></B> of the shared variable. If this is not done, the
 * reduction pattern will not work.
 *
 * @author  Alan Kaminsky
 * @version 13-Sep-2013
 */
public interface Vbl
	extends Cloneable
	{

// Exported operations.

	/**
	 * Create a clone of this shared variable. The clone must be a deep copy of
	 * this shared variable.
	 *
	 * @return  The cloned object.
	 */
	public Object clone();

	/**
	 * Set this shared variable to the given shared variable. This variable must
	 * be set to a deep copy of the given variable.
	 *
	 * @param  vbl  Shared variable.
	 *
	 * @exception  ClassCastException
	 *     (unchecked exception) Thrown if the class of <TT>vbl</TT> is not
	 *     compatible with the class of this shared variable.
	 */
	public void set
		(Vbl vbl);

	/**
	 * Reduce the given shared variable into this shared variable. The two
	 * variables are combined together, and the result is stored in this shared
	 * variable. The <TT>reduce()</TT> method does not need to be multiple
	 * thread safe (thread synchronization is handled by the caller).
	 *
	 * @param  vbl  Shared variable.
	 *
	 * @exception  ClassCastException
	 *     (unchecked exception) Thrown if the class of <TT>vbl</TT> is not
	 *     compatible with the class of this shared variable.
	 */
	public void reduce
		(Vbl vbl);

	}
