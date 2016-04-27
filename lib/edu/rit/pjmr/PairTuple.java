//******************************************************************************
//
// File:    PairTuple.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.PairTuple
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

package edu.rit.pjmr;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.pj2.Tuple;
import edu.rit.pj2.Vbl;
import edu.rit.util.AList;
import edu.rit.util.Pair;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

/**
 * Class PairTuple is a tuple that contains a list of (key, value) pairs from a
 * mapper task.
 *
 * @param  <K>  Key data type.
 * @param  <V>  Value data type; must implement interface {@linkplain
 *              edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 12-Jan-2015
 */
class PairTuple<K,V extends Vbl>
	extends Tuple
	{

// Hidden data members.

	private byte[] content;

// Exported constructors.

	/**
	 * Construct a new pair tuple with a null pair list.
	 */
	public PairTuple()
		{
		}

	/**
	 * Construct a new pair tuple with the given pair list.
	 *
	 * @param  pairList  Pair list, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred while streaming the pair list.
	 */
	public PairTuple
		(AList<Pair<K,V>> pairList)
		throws IOException
		{
		if (pairList != null)
			{
			ByteArrayOutputStream baos = new ByteArrayOutputStream();
			OutStream out = new OutStream (baos);
			out.writeObject (pairList);
			out.close();
			content = baos.toByteArray();
			}
		}

// Exported operations.

	/**
	 * Get this pair tuple's list of pairs.
	 *
	 * @return  Pair list, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred while destreaming the pair list.
	 */
	public AList<Pair<K,V>> pairList()
		throws IOException
		{
		if (content == null)
			return null;
		else
			{
			ByteArrayInputStream bais = new ByteArrayInputStream (content);
			InStream in = new InStream (bais);
			return (AList<Pair<K,V>>) in.readObject();
			}
		}

	/**
	 * Write this pair tuple to the given out stream.
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
		out.writeByteArray (content);
		}

	/**
	 * Read this pair tuple from the given in stream.
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
		content = in.readByteArray();
		}

	}
