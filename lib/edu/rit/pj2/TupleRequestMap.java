//******************************************************************************
//
// File:    TupleRequestMap.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.TupleRequestMap
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

package edu.rit.pj2;

import edu.rit.util.Map;

/**
 * Class TupleRequestMap provides an object for coordinating requests by a
 * {@linkplain Task Task} to take tuples out of tuple space in a {@linkplain Job
 * Job}.
 *
 * @author  Alan Kaminsky
 * @version 25-May-2015
 */
class TupleRequestMap
	{

// Hidden helper classes.

	private static class TupleRef
		{
		public Tuple tuple;

		public TupleRef
			(Tuple tuple)
			{
			this.tuple = tuple;
			}
		}

	private static class Info
		{
		public long taskID;
		public long requestID;
		public TupleListener listener;
		public TupleRef tupleRef;

		public Info
			(long taskID,
			 long requestID,
			 TupleListener listener)
			{
			this.taskID = taskID;
			this.requestID = requestID;
			this.listener = listener;
			}

		public boolean equals
			(Object obj)
			{
			return
				(obj instanceof Info) &&
				((Info) obj).taskID == this.taskID &&
				((Info) obj).requestID == this.requestID;
			}

		public int hashCode()
			{
			return (int)((taskID << 16) + requestID);
			}
		}

// Hidden data members.

	private Map<Info,Info> requestMap = new Map<Info,Info>();
	private long nextRequestID = 1L;

// Exported constructors.

	/**
	 * Construct a new tuple request map.
	 */
	public TupleRequestMap()
		{
		}

// Exported operations.

	/**
	 * Add a new request to take or read a tuple.
	 *
	 * @param  taskID    Task ID.
	 * @param  listener  Tuple listener, or null if none.
	 *
	 * @return  Request ID.
	 */
	public synchronized long addRequest
		(long taskID,
		 TupleListener listener)
		{
		long id = nextRequestID ++;
		Info info = new Info (taskID, id, listener);
		requestMap.put (info, info);
		return id;
		}

	/**
	 * Take a tuple. This method blocks until the tuple is taken.
	 *
	 * @param  taskID     Task ID.
	 * @param  requestID  Request ID.
	 *
	 * @return  Tuple.
	 */
	public synchronized Tuple takeTuple
		(long taskID,
		 long requestID)
		{
		Info key = new Info (taskID, requestID, null);
		Info info = requestMap.get (key);
		if (info == null)
			throw new IllegalStateException (String.format
				("TupleRequestMap.takeTuple(%d,%d): Unknown taskID/requestID",
				 taskID, requestID));
		while (info.tupleRef == null)
			try { wait(); }
				catch (InterruptedException exc) {}
		requestMap.remove (key);
		return info.tupleRef.tuple;
		}

	/**
	 * Report that the given tuple was taken.
	 *
	 * @param  taskID     Task ID.
	 * @param  requestID  Request ID.
	 * @param  tuple      Tuple.
	 */
	public synchronized void tupleTaken
		(long taskID,
		 long requestID,
		 Tuple tuple)
		{
		Info key = new Info (taskID, requestID, null);
		Info info = requestMap.get (key);
		if (info == null)
			throw new IllegalStateException (String.format
				("TupleRequestMap.tupleTaken(%d,%d): Unknown taskID/requestID",
				 taskID, requestID));
		if (info.listener == null)
			{
			info.tupleRef = new TupleRef (tuple);
			notifyAll();
			}
		else
			{
			requestMap.remove (key);
			try
				{
				info.listener.run (tuple);
				}
			catch (Throwable exc)
				{
				throw new TerminateException
					("Exception in tuple listener", exc);
				}
			}
		}

	}
