//******************************************************************************
//
// File:    PairSender.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.PairSender
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

import edu.rit.pj2.Task;
import edu.rit.pj2.Vbl;
import edu.rit.util.AList;
import edu.rit.util.Pair;
import java.io.IOException;

/**
 * Class PairSender provides an object that sends a series of (key, value) pairs
 * from a mapper task to a reducer task via tuple space.
 * <P>
 * The chunk size, a constructor argument, specifies the maximum number of pairs
 * that will be sent in a single {@linkplain PairTuple}. Before sending a chunk
 * of pairs, the mapper waits for the reducer to give permission to send a
 * chunk; see class {@linkplain GoAheadTuple}.
 *
 * @param  <K>  Key data type.
 * @param  <V>  Value data type; must implement interface {@linkplain
 *              edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 12-Jan-2015
 */
class PairSender<K,V extends Vbl>
	{

// Hidden data members.

	private Task task;
	private int chunk;
	private AList<Pair<K,V>> pairList;
	private GoAheadTuple goAhead;

// Exported constructors.

	/**
	 * Construct a new pair sender with the default chunk size (1000).
	 *
	 * @param  task  Mapper task.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>task</TT> is null.
	 */
	public PairSender
		(Task task)
		{
		this (task, 1000);
		}

	/**
	 * Construct a new pair sender with the given chunk size.
	 *
	 * @param  task   Mapper task.
	 * @param  chunk  Chunk size &ge; 1.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>task</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>chunk</TT> &lt; 1.
	 */
	public PairSender
		(Task task,
		 int chunk)
		{
		if (task == null)
			throw new NullPointerException
				("PairSender(): task is null");
		if (chunk < 1)
			throw new IllegalArgumentException (String.format
				("PairSender(): chunk = %d illegal", chunk));
		this.task = task;
		this.chunk = chunk;
		this.pairList = new AList<Pair<K,V>>();
		this.goAhead = new GoAheadTuple();
		}

// Exported operations.

	/**
	 * Send the given (key, value) pair.
	 *
	 * @param  pair  Pair.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void send
		(Pair<K,V> pair)
		throws IOException
		{
		if (pairList.size() == chunk) flush();
		pairList.addLast (pair);
		}

	/**
	 * Indicate that no more (key, value) pairs will be sent.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void close()
		throws IOException
		{
		send (null);
		if (pairList.size() > 0) flush();
		pairList = null;
		}

// Hidden operations.

	/**
	 * Send the pairs in the pair list.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private void flush()
		throws IOException
		{
		task.takeTuple (goAhead);
		task.putTuple (new PairTuple<K,V> (pairList));
		pairList.clear();
		}

	}
