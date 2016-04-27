//******************************************************************************
//
// File:    PairReceiver.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.PairReceiver
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
import edu.rit.util.Action;
import edu.rit.util.Pair;
import java.io.IOException;

/**
 * Class PairReceiver provides an object that receives a series of (key, value)
 * pairs sent from multiple mapper tasks to a reducer task via tuple space.
 *
 * @param  <K>  Key data type.
 * @param  <V>  Value data type; must implement interface {@linkplain
 *              edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 12-Jan-2015
 */
class PairReceiver<K,V extends Vbl>
	{

// Hidden data members.

	private Task task;
	private int mapperCount;

// Exported constructors.

	/**
	 * Construct a new pair receiver.
	 *
	 * @param  task         Reducer task.
	 * @param  mapperCount  Number of mapper tasks &ge; 1.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>task</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>mapperCount</TT> &lt; 1.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public PairReceiver
		(Task task,
		 int mapperCount)
		throws IOException
		{
		if (task == null)
			throw new NullPointerException
				("PairReceiver(): task is null");
		if (mapperCount < 1)
			throw new IllegalArgumentException (String.format
				("PairReceiver(): mapperCount = %d illegal", mapperCount));

		this.task = task;
		this.mapperCount = mapperCount;
		}

// Exported operations.

	/**
	 * Perform the given action on each (key, value) pair received.
	 *
	 * @param  action  Action.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void forEachPairDo
		(Action<Pair<K,V>> action)
		throws IOException
		{
		int mappersRemaining = mapperCount;
		PairTuple<K,V> pairTemplate = new PairTuple<K,V>();
		AList<Pair<K,V>> pairList = null;
		Pair<K,V> pair = null;

		task.putTuple (mapperCount, new GoAheadTuple());

		while (mappersRemaining > 0)
			{
			pairList = task.takeTuple (pairTemplate) .pairList();
			for (int i = 0; i < pairList.size(); ++ i)
				{
				pair = pairList.get (i);
				if (pair == null)
					-- mappersRemaining;
				else
					action.run (pair);
				}
			task.putTuple (new GoAheadTuple());
			}
		}

	}
