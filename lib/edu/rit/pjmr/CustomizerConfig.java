//******************************************************************************
//
// File:    CustomizerConfig.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.CustomizerConfig
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
import edu.rit.pj2.Vbl;
import edu.rit.util.Instance;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;

/**
 * Class CustomizerConfig contains specifications for configuring a {@linkplain
 * Customizer} in a mapper task or a reducer task.
 *
 * @param  <K>  Mapper output key data type.
 * @param  <V>  Mapper output value data type; must implement interface
 *              {@linkplain edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 10-Dec-2013
 */
class CustomizerConfig<K,V extends Vbl>
	implements Streamable
	{

// Exported data members.

	// Customizer class name and argument strings.
	String customizerClassName;
	String[] customizerArgs;

// Exported constructors.

	/**
	 * Construct a new uninitialized customizer config object. This constructor
	 * is for use only by object deserialization.
	 */
	CustomizerConfig()
		{
		}

	/**
	 * Construct a new customizer config object.
	 *
	 * @param  customizerClass  Customizer class.
	 * @param  args             Zero or more argument strings for the
	 *                          customizer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>customizerClass</TT> is null.
	 */
	<T extends Customizer<K,V>> CustomizerConfig
		(Class<T> customizerClass,
		 String... args)
		{
		if (customizerClass == null)
			throw new NullPointerException
				("CustomizerConfig(): customizerClass is null");
		this.customizerClassName = customizerClass.getName();
		this.customizerArgs = args;
		}

// Exported operations.

	/**
	 * Write this customizer config object to the given out stream.
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
		out.writeString (customizerClassName);
		out.writeStringArray (customizerArgs);
		}

	/**
	 * Read this customzier config object from the given in stream.
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
		customizerClassName = in.readString();
		customizerArgs = in.readStringArray();
		}

// Hidden operations.

	/**
	 * Construct a new customizer as specified by this customizer config object.
	 *
	 * @return  Customizer.
	 *
	 * @exception  ClassNotFoundException
	 *     Thrown if the customizer class could not be found.
	 * @exception  NoSuchMethodException
	 *     Thrown if the customizer class does not have a no-argument
	 *     constructor.
	 * @exception  InstantiationException
	 *     Thrown if the customizer class could not be instantiated.
	 * @exception  IllegalAccessException
	 *     Thrown if the customizer class or its no-argument constructor is not
	 *     accessible.
	 * @exception  InvocationTargetException
	 *     Thrown if the customizer class constructor threw an exception.
	 */
	Customizer<K,V> newInstance()
		throws ClassNotFoundException,
			NoSuchMethodException,
			InstantiationException,
			IllegalAccessException,
			InvocationTargetException
		{
		return (Customizer<K,V>)
			Instance.newDefaultInstance (customizerClassName, true);
		}

	}
