//******************************************************************************
//
// File:    TaskInfo.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.TaskInfo
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

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import edu.rit.pj2.Tuple;
import edu.rit.util.AList;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

/**
 * Class TaskInfo is a record of information about a {@linkplain
 * edu.rit.pj2.Task Task} in a {@linkplain edu.rit.pj2.Job Job}. It carries the
 * information a {@linkplain edu.rit.pj2.Backend Backend} process needs to
 * execute a {@linkplain edu.rit.pj2.Task Task} as part of a {@linkplain
 * edu.rit.pj2.Job Job}.
 * <P>
 * Task info objects are streamable. When a task info object is serialized and
 * later deserialized, the input tuples are stored in serialized form in a
 * private field, and the public <TT>inputTuples</TT> field is set to null. To
 * get the input tuples back, call the {@link #unmarshalInputTuples()
 * unmarshalInputTuples()} method. This is done to avoid deserializing the input
 * tuples if the recipient does not need to use them or does not have the class
 * files needed to deserialize them.
 *
 * @author  Alan Kaminsky
 * @version 24-Mar-2014
 */
public class TaskInfo
	implements Streamable
	{

// Exported data members.

	/**
	 * Task ID.
	 */
	public long taskID;

	/**
	 * Task subclass name.
	 */
	public String taskClassName;

	/**
	 * Array of command line argument strings.
	 */
	public String[] args;

	/**
	 * List of input tuples.
	 */
	public AList<Tuple> inputTuples;

	/**
	 * Task properties.
	 */
	public TaskProperties properties;

	/**
	 * Java archive (JAR) containing task's class files, or null if none.
	 */
	public byte[] jar;

	/**
	 * Size of the task's task group.
	 */
	public int size;

	/**
	 * Rank of the task within the task group.
	 */
	public int rank;

	/**
	 * Array of GPU device numbers the task is allowed to use, or null to use
	 * all devices.
	 */
	public int[] devnum;

	/**
	 * Additional data, or null if none.
	 * <P>
	 * <I>Note:</I> The additional data is not included when a task info object
	 * is serialized.
	 */
	public Object moreData;

// Hidden data members.

	private byte[] marshaledInputTuples;

// Exported constructors.

	/**
	 * Construct a new task info object.
	 */
	public TaskInfo()
		{
		}

// Exported operations.

	/**
	 * Write this task info object to the given out stream.
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
		// Marshal input tuples if necessary.
		if (inputTuples != null)
			{
			ByteArrayOutputStream baos = new ByteArrayOutputStream();
			OutStream oos = new OutStream (baos);
			oos.writeFields (inputTuples);
			oos.close();
			inputTuples = null;
			marshaledInputTuples = baos.toByteArray();
			}

		// Write fields.
		out.writeLong (taskID);
		out.writeString (taskClassName);
		out.writeStringArray (args);
		out.writeFields (properties);
		out.writeByteArray (jar);
		out.writeInt (size);
		out.writeInt (rank);
		out.writeIntArray (devnum);
		out.writeByteArray (marshaledInputTuples);
		}

	/**
	 * Read this task info object from the given in stream.
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
		// Read fields.
		taskID = in.readLong();
		taskClassName = in.readString();
		args = in.readStringArray();
		inputTuples = null;
		properties = in.readFields (new TaskProperties());
		jar = in.readByteArray();
		size = in.readInt();
		rank = in.readInt();
		devnum = in.readIntArray();
		moreData = null;
		marshaledInputTuples = in.readByteArray();
		}

	/**
	 * Unmarshal the {@link #inputTuples} field.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void unmarshalInputTuples()
		throws IOException
		{
		if (marshaledInputTuples != null)
			{
			ByteArrayInputStream bais =
				new ByteArrayInputStream (marshaledInputTuples);
			InStream ois = new InStream (bais);
			inputTuples = ois.readFields (new AList<Tuple>());
			marshaledInputTuples = null;
			}
		}

	/**
	 * Returns a string version of this task info object.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		StringBuilder b = new StringBuilder();
		b.append ("TaskInfo(taskID=");
		b.append (taskID);
		b.append ("\",taskClassName=");
		b.append (taskClassName);
		b.append (",args={");
		for (int i = 0; i < args.length; ++ i)
			{
			if (i > 0) b.append (',');
			b.append ('"');
			b.append (args[i]);
			b.append ('"');
			}
		b.append ("},inputTuples={");
		for (int i = 0; i < inputTuples.size(); ++ i)
			{
			if (i > 0) b.append (',');
			b.append (inputTuples.get(i).getClass().getName());
			}
		b.append ("},properties=");
		b.append (properties);
		b.append (",jar=");
		if (jar == null)
			b.append ("null");
		else
			{
			b.append ("byte[");
			b.append (jar.length);
			b.append (']');
			}
		b.append (",size=");
		b.append (size);
		b.append (",rank=");
		b.append (rank);
		if (devnum == null)
			b.append (",devnum=null");
		else
			{
			b.append (",devnum={");
			for (int i = 0; i < devnum.length; ++ i)
				{
				if (i > 0) b.append (',');
				b.append (devnum[i]);
				}
			b.append ("}");
			}
		b.append ("})");
		return b.toString();
		}

	}
