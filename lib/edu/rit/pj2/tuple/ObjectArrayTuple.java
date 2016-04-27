//******************************************************************************
//
// File:    ObjectArrayTuple.java
// Package: edu.rit.pj2.tuple
// Unit:    Class edu.rit.pj2.tuple.ObjectArrayTuple
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
 * Class ObjectArrayTuple provides a {@linkplain Tuple Tuple} that contains an
 * array of objects. The objects must be streamable or serializable.
 *
 * @param  <T>  Content data type.
 *
 * @author  Alan Kaminsky
 * @version 14-Jan-2015
 */
public class ObjectArrayTuple<T>
	extends Tuple
	{

// Exported data members.

	/**
	 * Array of content objects.
	 */
	public T[] item;

// Exported constructors.

	/**
	 * Construct a new object array tuple. The content object array is null.
	 */
	public ObjectArrayTuple()
		{
		}

	/**
	 * Construct a new object tuple with the given array of content objects.
	 * This tuple's <TT>item</TT> field is set to a new array whose elements are
	 * references to the objects in the given array.
	 *
	 * @param  item  Object array.
	 */
	public ObjectArrayTuple
		(T[] item)
		{
		this.item = (T[]) item.clone();
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
	 * This template tuple's <TT>item</TT> array is null.
	 * <LI>
	 * The target tuple's <TT>item</TT> array is null.
	 * <LI>
	 * The element type of the target tuple's <TT>item</TT> array is the same as
	 * (or a subclass of) the element type of this template tuple's
	 * <TT>item</TT> array.
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
		Object[] targetItem = ((ObjectArrayTuple)target).item;
		return
			this.item == null ||
			targetItem == null ||
			this.item.getClass().getComponentType().isAssignableFrom
				(targetItem.getClass().getComponentType());
		}

	/**
	 * Write this tuple's fields to the given out stream. The content object
	 * array is written using {@link
	 * edu.rit.io.OutStream#writeObjectArray(Object[]) writeObjectArray()}.
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
		out.writeObjectArray (item);
		}

	/**
	 * Read this tuple's fields from the given in stream. The content object
	 * array is read using {@link edu.rit.io.InStream#readObjectArray()
	 * readObjectArray()}.
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
		item = (T[]) in.readObjectArray();
		}

	}
