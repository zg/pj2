//******************************************************************************
//
// File:    TaskSpecInfo.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.TaskSpecInfo
//
// This Java source file is copyright (C) 2014 by Alan Kaminsky. All rights
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

package edu.rit.pj2.tracker;

import edu.rit.io.Streamable;
import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import java.io.IOException;

/**
 * Class TaskSpecInfo is a record of information about a {@linkplain
 * edu.rit.pj2.TaskSpec TaskSpec} in a {@linkplain edu.rit.pj2.Rule Rule} in a
 * {@linkplain edu.rit.pj2.Job Job}. It carries the information a {@linkplain
 * Tracker Tracker} needs to schedule a {@linkplain edu.rit.pj2.Task Task} as
 * part of a {@linkplain edu.rit.pj2.Job Job}.
 *
 * @author  Alan Kaminsky
 * @version 23-Mar-2014
 */
public class TaskSpecInfo
	implements Streamable
	{

// Exported data members.

	/**
	 * Task ID.
	 */
	public long taskID;

	/**
	 * Task node requirements.
	 */
	public NodeProperties node;

	/**
	 * Array of JVM flags for the task's backend process.
	 */
	public String[] jvmFlags;

	/**
	 * Additional data, or null if none.
	 * <P>
	 * <I>Note:</I> The additional data is not included when a task
	 * specification info object is serialized.
	 */
	public Object moreData;

// Exported constructors.

	/**
	 * Construct a new task specification info object.
	 */
	public TaskSpecInfo()
		{
		}

// Exported operations.

	/**
	 * Write this task specification info object to the given out stream.
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
		out.writeLong (taskID);
		out.writeFields (node);
		out.writeStringArray (jvmFlags);
		}

	/**
	 * Read this task specification info object from the given in stream.
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
		taskID = in.readLong();
		node = in.readFields (new NodeProperties());
		jvmFlags = in.readStringArray();
		moreData = null;
		}

	/**
	 * Returns a string version of this task specification info object.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		StringBuilder b = new StringBuilder();
		b.append ("TaskSpecInfo(taskID=");
		b.append (taskID);
		b.append (",node=");
		b.append (node);
		b.append (",jvmFlags={");
		for (int i = 0; i < jvmFlags.length; ++ i)
			{
			if (i > 0) b.append (',');
			b.append ('"');
			b.append (jvmFlags[i]);
			b.append ('"');
			}
		b.append ("})");
		return b.toString();
		}

	}
