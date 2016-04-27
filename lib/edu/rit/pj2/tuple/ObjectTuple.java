//******************************************************************************
//
// File:    ObjectTuple.java
// Package: edu.rit.pj2.tuple
// Unit:    Class edu.rit.pj2.tuple.ObjectTuple
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

package edu.rit.pj2.tuple;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.pj2.Tuple;
import java.io.IOException;

/**
 * Class ObjectTuple provides a {@linkplain Tuple Tuple} that contains a single
 * object. The object must be streamable or serializable.
 *
 * @param  <T>  Content data type.
 *
 * @author  Alan Kaminsky
 * @version 14-Jan-2015
 */
public class ObjectTuple<T>
	extends Tuple
	{

// Exported data members.

	/**
	 * Content object.
	 */
	public T item;

// Exported constructors.

	/**
	 * Construct a new object tuple. The content object reference is null.
	 */
	public ObjectTuple()
		{
		}

	/**
	 * Construct a new object tuple with the given content object.
	 *
	 * @param  item  Object.
	 */
	public ObjectTuple
		(T item)
		{
		this.item = item;
		}

// Exported operations.

	/**
	 * Determine if the given target tuple's content matches this template's
	 * content. The target tuple is assumed to be an instance of this template's
	 * matching class or a subclass thereof.
	 * <P>
	 * The ObjectTuple class's <TT>matchContent()</TT> method returns true if
	 * any of the following are true:
	 * <UL>
	 * <LI>
	 * This template tuple's <TT>item</TT> field is null.
	 * <LI>
	 * The target tuple's <TT>item</TT> field is null.
	 * <LI>
	 * The target tuple's <TT>item</TT> is an instance of this template tuple's
	 * <TT>item</TT>'s class or a subclass thereof.
	 * </UL>
	 *
	 * @param  target  Target tuple.
	 *
	 * @return  True if the target tuple's content matches this template tuple's
	 *          content, false otherwise.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>target</TT> is null.
	 */
	public boolean matchContent
		(Tuple target)
		{
		Object targetItem = ((ObjectTuple)target).item;
		return
			this.item == null ||
			targetItem == null ||
			this.item.getClass().isAssignableFrom (targetItem.getClass());
		}

	/**
	 * Write this tuple's fields to the given out stream. The content object is
	 * written using {@link edu.rit.io.OutStream#writeObject(Object)
	 * writeObject()}.
	 *
	 * @param  out  Out stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeOut
		(OutStream out)
		throws IOException
		{
		out.writeObject (item);
		}

	/**
	 * Read this tuple's fields from the given in stream. The content object is
	 * read using {@link edu.rit.io.InStream#readObject() readObject()}.
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
		item = (T) in.readObject();
		}

	}
