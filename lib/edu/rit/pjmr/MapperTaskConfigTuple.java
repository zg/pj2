//******************************************************************************
//
// File:    MapperTaskConfigTuple.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.MapperTaskConfigTuple
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

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.pj2.Tuple;
import edu.rit.pj2.Vbl;
import java.io.IOException;

/**
 * Class MapperTaskConfigTuple is a tuple that contains specifications for
 * configuring a mapper task.
 *
 * @param  <IK>  Mapper input key data type.
 * @param  <IV>  Mapper input value data type.
 * @param  <OK>  Mapper output key data type.
 * @param  <OV>  Mapper output value data type; must implement interface
 *               {@linkplain edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 30-Nov-2013
 */
class MapperTaskConfigTuple<IK,IV,OK,OV extends Vbl>
	extends Tuple
	{

// Exported data members.

	// Node name.
	public String nodeName;

	// Mapper task configuration.
	public MapperTaskConfig<IK,IV,OK,OV> config;

	// Combiner configuration.
	public CombinerConfig<OK,OV> combinerConfig;

// Exported constructors.

	/**
	 * Construct a new mapper task config tuple.
	 */
	public MapperTaskConfigTuple()
		{
		}

	/**
	 * Construct a new mapper task config tuple with the given node name.
	 *
	 * @param  nodeName  Node name.
	 */
	public MapperTaskConfigTuple
		(String nodeName)
		{
		this.nodeName = nodeName;
		}

	/**
	 * Construct a new mapper task config tuple with the given configuration.
	 * The node name is that of the configuration.
	 *
	 * @param  config          Mapper task configuration.
	 * @param  combinerConfig  Combiner configuration.
	 */
	public MapperTaskConfigTuple
		(MapperTaskConfig<IK,IV,OK,OV> config,
		 CombinerConfig<OK,OV> combinerConfig)
		{
		this.nodeName = config.nodeName;
		this.config = config;
		this.combinerConfig = combinerConfig;
		}

// Exported operations.

	/**
	 * Determine if the given target tuple's content matches this template's
	 * content. This is true if the node names match.
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
		return this.nodeName.equals
			(((MapperTaskConfigTuple<IK,IV,OK,OV>)target).nodeName);
		}

	/**
	 * Write this mapper task config tuple to the given out stream.
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
		out.writeString (nodeName);
		out.writeObject (config);
		out.writeObject (combinerConfig);
		}

	/**
	 * Read this mapper task config tuple from the given in stream.
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
		nodeName = in.readString();
		config = (MapperTaskConfig<IK,IV,OK,OV>) in.readObject();
		combinerConfig = (CombinerConfig<OK,OV>) in.readObject();
		}

	}
