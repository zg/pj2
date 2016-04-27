//******************************************************************************
//
// File:    CombinerConfig.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.CombinerConfig
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

import edu.rit.io.Streamable;
import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.util.Instance;
import edu.rit.pj2.Vbl;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;

/**
 * Class CombinerConfig contains specifications for configuring a {@linkplain
 * Combiner} in a mapper task or reducer task.
 *
 * @param  <K>  Mapper output key data type.
 * @param  <V>  Mapper output value data type; must implement interface
 *              {@linkplain edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 10-Dec-2013
 */
class CombinerConfig<K,V extends Vbl>
	implements Streamable
	{

// Exported data members.

	// Combiner class name.
	String combinerClassName;

// Exported constructors.

	/**
	 * Construct a new uninitialized combiner config object. This constructor is
	 * for use only by object deserialization.
	 */
	CombinerConfig()
		{
		}

	/**
	 * Construct a new combiner config object.
	 *
	 * @param  combinerClass  Combiner class.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>combinerClass</TT> is null.
	 */
	<T extends Combiner<K,V>> CombinerConfig
		(Class<T> combinerClass)
		{
		if (combinerClass == null)
			throw new NullPointerException
				("CombinerConfig(): combinerClass is null");
		this.combinerClassName = combinerClass.getName();
		}

// Exported operations.

	/**
	 * Write this combiner config object to the given out stream.
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
		out.writeString (combinerClassName);
		}

	/**
	 * Read this combiner config object from the given in stream.
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
		combinerClassName = in.readString();
		}

// Hidden operations.

	/**
	 * Construct a new combiner as specified by this combiner config object.
	 *
	 * @return  Combiner.
	 *
	 * @exception  ClassNotFoundException
	 *     Thrown if the combiner class could not be found.
	 * @exception  NoSuchMethodException
	 *     Thrown if the combiner class does not have a no-argument constructor.
	 * @exception  InstantiationException
	 *     Thrown if the combiner class could not be instantiated.
	 * @exception  IllegalAccessException
	 *     Thrown if the combiner class or its no-argument constructor is not
	 *     accessible.
	 * @exception  InvocationTargetException
	 *     Thrown if the combiner class constructor threw an exception.
	 */
	Combiner<K,V> newInstance()
		throws ClassNotFoundException,
			NoSuchMethodException,
			InstantiationException,
			IllegalAccessException,
			InvocationTargetException
		{
		return (Combiner<K,V>)
			Instance.newDefaultInstance (combinerClassName, true);
		}

	}
