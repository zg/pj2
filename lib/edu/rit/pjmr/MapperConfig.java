//******************************************************************************
//
// File:    MapperConfig.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.MapperConfig
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
 * Class MapperConfig contains specifications for configuring a {@linkplain
 * Mapper} in a mapper task.
 *
 * @param  <IK>  Mapper input key data type.
 * @param  <IV>  Mapper input value data type.
 * @param  <OK>  Mapper output key data type.
 * @param  <OV>  Mapper output value data type; must implement interface
 *               {@linkplain edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 10-Dec-2013
 */
public class MapperConfig<IK,IV,OK,OV extends Vbl>
	implements Streamable
	{

// Exported data members.

	// Source config object that contains this mapper config object.
	SourceConfig<IK,IV,OK,OV> sourceConfig;

	// Mapper class name and argument strings.
	String mapperClassName;
	String[] mapperArgs;

	// Whether this mapper needs a GPU.
	boolean needsGpu = false;

// Exported constructors.

	/**
	 * Construct a new uninitialized mapper config object. This constructor is
	 * for use only by object deserialization.
	 */
	MapperConfig()
		{
		}

	/**
	 * Construct a new mapper config object.
	 *
	 * @param  sourceConfig  Source config object that contains this mapper
	 *                       config object.
	 * @param  mapperClass   Mapper class.
	 * @param  args          Zero or more argument strings for the mapper.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>mapperClass</TT> is null.
	 */
	<T extends Mapper<IK,IV,OK,OV>> MapperConfig
		(SourceConfig<IK,IV,OK,OV> sourceConfig,
		 Class<T> mapperClass,
		 String... args)
		{
		if (mapperClass == null)
			throw new NullPointerException
				("MapperConfig(): mapperClass is null");
		this.sourceConfig = sourceConfig;
		this.mapperClassName = mapperClass.getName();
		this.mapperArgs = args;
		}

// Exported operations.

	/**
	 * Specify that this mapper needs a GPU. If not specified, the default is
	 * that the mapper does not need a GPU.
	 *
	 * @return  This mapper config object.
	 */
	public MapperConfig<IK,IV,OK,OV> needsGpu()
		{
		needsGpu = true;
		return this;
		}

	/**
	 * Add the given {@linkplain Mapper} to this mapper's source. For further
	 * information, see {@link SourceConfig#mapper(Class,String[])
	 * SourceConfig.mapper()}.
	 *
	 * @param  mapperClass  Mapper class.
	 * @param  args         Zero or more argument strings for the mapper.
	 *
	 * @return  Mapper config object for <TT>mapperClass</TT>.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>mapperClass</TT> is null.
	 */
	public <T extends Mapper<IK,IV,OK,OV>> MapperConfig<IK,IV,OK,OV> mapper
		(Class<T> mapperClass,
		 String... args)
		{
		return sourceConfig.mapper (mapperClass, args);
		}

	/**
	 * Add the given number of copies of the given {@linkplain Mapper} to this
	 * mapper's source. For further information, see {@link
	 * SourceConfig#mapper(int,Class,String[]) SourceConfig.mapper()}.
	 *
	 * @param  copies       Number of copies to add (&ge; 1).
	 * @param  mapperClass  Mapper class.
	 * @param  args         Zero or more argument strings for the mappers.
	 *
	 * @return  Mapper config object for <TT>mapperClass</TT>.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>copies</TT> &lt; 1.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>mapperClass</TT> is null.
	 */
	public <T extends Mapper<IK,IV,OK,OV>> MapperConfig<IK,IV,OK,OV> mapper
		(int copies,
		 Class<T> mapperClass,
		 String... args)
		{
		return sourceConfig.mapper (copies, mapperClass, args);
		}

	/**
	 * Add the given {@linkplain Source} to this mapper's source's mapper task.
	 * For further information, see {@link MapperTaskConfig#source(Source)
	 * MapperTaskConfig.source()}.
	 *
	 * @param  source  Source.
	 *
	 * @return  Source config object for <TT>source</TT>.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>source</TT> is null.
	 */
	public SourceConfig<IK,IV,OK,OV> source
		(Source<IK,IV> source)
		{
		return sourceConfig.mapperTaskConfig.source (source);
		}

	/**
	 * Write this mapper config object to the given out stream.
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
		out.writeString (mapperClassName);
		out.writeStringArray (mapperArgs);
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
		mapperClassName = in.readString();
		mapperArgs = in.readStringArray();
		}

// Hidden operations.

	/**
	 * Construct a new mapper as specified by this mapper config object.
	 *
	 * @return  Mapper.
	 *
	 * @exception  ClassNotFoundException
	 *     Thrown if the mapper class could not be found.
	 * @exception  NoSuchMethodException
	 *     Thrown if the mapper class does not have a no-argument constructor.
	 * @exception  InstantiationException
	 *     Thrown if the mapper class could not be instantiated.
	 * @exception  IllegalAccessException
	 *     Thrown if the mapper class or its no-argument constructor is not
	 *     accessible.
	 * @exception  InvocationTargetException
	 *     Thrown if the mapper class constructor threw an exception.
	 */
	Mapper<IK,IV,OK,OV> newInstance()
		throws ClassNotFoundException,
			NoSuchMethodException,
			InstantiationException,
			IllegalAccessException,
			InvocationTargetException
		{
		return (Mapper<IK,IV,OK,OV>)
			Instance.newDefaultInstance (mapperClassName, true);
		}

	}
