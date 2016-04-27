//******************************************************************************
//
// File:    ReducerConfig.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.ReducerConfig
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
 * Class ReducerConfig contains specifications for configuring a {@linkplain
 * Reducer} in a reducer task.
 *
 * @param  <K>  Mapper output key data type.
 * @param  <V>  Mapper output value data type; must implement interface
 *              {@linkplain edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 10-Dec-2013
 */
public class ReducerConfig<K,V extends Vbl>
	implements Streamable
	{

// Exported data members.

	// Reducer task config object that contains this reducer config object.
	ReducerTaskConfig<K,V> reducerTaskConfig;

	// Reducer class name and argument strings.
	String reducerClassName;
	String[] reducerArgs;

	// Whether this reducer needs a GPU.
	boolean needsGpu = false;

// Exported constructors.

	/**
	 * Construct a new uninitialized reducer config object. This constructor is
	 * for use only by object deserialization.
	 */
	ReducerConfig()
		{
		}

	/**
	 * Construct a new reducer config object.
	 *
	 * @param  reducerTaskConfig  Reducer task config object that contains this
	 *                            reducer config object.
	 * @param  reducerClass       Reducer class.
	 * @param  args               Zero or more argument strings for the reducer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>reducerClass</TT> is null.
	 */
	<T extends Reducer<K,V>> ReducerConfig
		(ReducerTaskConfig<K,V> reducerTaskConfig,
		 Class<T> reducerClass,
		 String... args)
		{
		if (reducerClass == null)
			throw new NullPointerException
				("ReducerConfig(): reducerClass is null");
		this.reducerTaskConfig = reducerTaskConfig;
		this.reducerClassName = reducerClass.getName();
		this.reducerArgs = args;
		}

// Exported operations.

	/**
	 * Specify that this reducer needs a GPU. If not specified, the default is
	 * that the reducer does not need a GPU.
	 *
	 * @return  This reducer config object.
	 */
	public ReducerConfig<K,V> needsGpu()
		{
		needsGpu = true;
		return this;
		}

	/**
	 * Add the given {@linkplain Reducer} to this reducer's reducer task. For
	 * further information, see {@link ReducerTaskConfig#reducer(Class,String[])
	 * ReducerTaskConfig.reducer()}.
	 *
	 * @param  reducerClass  Reducer class.
	 * @param  args          Zero or more argument strings for the reducer.
	 *
	 * @return  Reducer config object for <TT>reducerClass</TT>.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>reducerClass</TT> is null.
	 */
	public <T extends Reducer<K,V>> ReducerConfig<K,V> reducer
		(Class<T> reducerClass,
		 String... args)
		{
		return reducerTaskConfig.reducer (reducerClass, args);
		}

	/**
	 * Add the given number of copies of the given {@linkplain Reducer} to this
	 * reducer's reducer task. For further information, see {@link
	 * ReducerTaskConfig#reducer(int,Class,String[])
	 * ReducerTaskConfig.reducer()}.
	 *
	 * @param  copies        Number of copies to add (&ge; 1).
	 * @param  reducerClass  Reducer class.
	 * @param  args          Zero or more argument strings for the reducers.
	 *
	 * @return  Reducer config object for <TT>reducerClass</TT>.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>copies</TT> &lt; 1.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>reducerClass</TT> is null.
	 */
	public <T extends Reducer<K,V>> ReducerConfig<K,V> reducer
		(int copies,
		 Class<T> reducerClass,
		 String... args)
		{
		return reducerTaskConfig.reducer (copies, reducerClass, args);
		}

	/**
	 * Write this reducer config object to the given out stream.
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
		out.writeString (reducerClassName);
		out.writeStringArray (reducerArgs);
		}

	/**
	 * Read this reducer config object from the given in stream.
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
		reducerClassName = in.readString();
		reducerArgs = in.readStringArray();
		}

// Hidden operations.

	/**
	 * Construct a new reducer as specified by this reducer config object.
	 *
	 * @return  Reducer.
	 *
	 * @exception  ClassNotFoundException
	 *     Thrown if the reducer class could not be found.
	 * @exception  NoSuchMethodException
	 *     Thrown if the reducer class does not have a no-argument constructor.
	 * @exception  InstantiationException
	 *     Thrown if the reducer class could not be instantiated.
	 * @exception  IllegalAccessException
	 *     Thrown if the reducer class or its no-argument constructor is not
	 *     accessible.
	 * @exception  InvocationTargetException
	 *     Thrown if the reducer class constructor threw an exception.
	 */
	Reducer<K,V> newInstance()
		throws ClassNotFoundException,
			NoSuchMethodException,
			InstantiationException,
			IllegalAccessException,
			InvocationTargetException
		{
		return (Reducer<K,V>)
			Instance.newDefaultInstance (reducerClassName, true);
		}

	}
