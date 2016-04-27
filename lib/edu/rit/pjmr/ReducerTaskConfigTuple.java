//******************************************************************************
//
// File:    ReducerTaskConfigTuple.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.ReducerTaskConfigTuple
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
 * Class ReducerTaskConfigTuple is a tuple that contains specifications for
 * configuring a reducer task.
 *
 * @param  <K>  Mapper output key data type.
 * @param  <V>  Mapper output value data type; must implement interface
 *              {@linkplain edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 30-Nov-2013
 */
class ReducerTaskConfigTuple<K,V extends Vbl>
	extends Tuple
	{

// Exported data members.

	// Reducer task configuration.
	public ReducerTaskConfig<K,V> config;

	// Combiner configuration.
	public CombinerConfig<K,V> combinerConfig;

// Exported constructors.

	/**
	 * Construct a new reducer task config tuple.
	 */
	public ReducerTaskConfigTuple()
		{
		}

	/**
	 * Construct a new reducer task config tuple with the given configuration.
	 *
	 * @param  config          Reducer task configuration.
	 * @param  combinerConfig  Combiner configuration.
	 */
	public ReducerTaskConfigTuple
		(ReducerTaskConfig<K,V> config,
		 CombinerConfig<K,V> combinerConfig)
		{
		this.config = config;
		this.combinerConfig = combinerConfig;
		}

// Exported operations.

	/**
	 * Write this reducer task config tuple to the given out stream.
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
		out.writeObject (config);
		out.writeObject (combinerConfig);
		}

	/**
	 * Read this reducer task config tuple from the given in stream.
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
		config = (ReducerTaskConfig<K,V>) in.readObject();
		combinerConfig = (CombinerConfig<K,V>) in.readObject();
		}

	}
