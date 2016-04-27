//******************************************************************************
//
// File:    EmptyTuple.java
// Package: edu.rit.pj2.tuple
// Unit:    Class edu.rit.pj2.tuple.EmptyTuple
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
 * Class EmptyTuple provides a {@linkplain Tuple Tuple} with no content.
 *
 * @author  Alan Kaminsky
 * @version 14-Jan-2015
 */
public class EmptyTuple
	extends Tuple
	{

// Exported constructors.

	/**
	 * Construct a new empty tuple.
	 */
	public EmptyTuple()
		{
		}

// Exported operations.

	/**
	 * Write this tuple's fields to the given out stream.
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
		}

	/**
	 * Read this tuple's fields from the given in stream.
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
		}

	}
