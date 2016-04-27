//******************************************************************************
//
// File:    TaskMap.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.TaskMap
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

package edu.rit.pj2.tracker;

import edu.rit.util.Action;
import edu.rit.util.Map;

/**
 * Class TaskMap holds information about the {@linkplain edu.rit.pj2.Task Task}s
 * for a {@linkplain edu.rit.pj2.Job Job}.
 * <P>
 * <I>Note:</I> Class TaskMap is multiple thread safe.
 *
 * @author  Alan Kaminsky
 * @version 10-Jan-2015
 */
public class TaskMap
	{

// Hidden data members.

	// Map from task ID to task info objects.
	private Map<Long,TaskInfo> taskMap = new Map<Long,TaskInfo>();

	// Next task ID to be assigned.
	private long nextTaskID = 1;

// Exported constructors.

	/**
	 * Construct a new task map.
	 */
	public TaskMap()
		{
		}

// Exported operations.

	/**
	 * Determine if this task map is empty.
	 *
	 * @return  True if this task map is empty, false if it isn't.
	 */
	public synchronized boolean isEmpty()
		{
		return taskMap.isEmpty();
		}

	/**
	 * Construct a new task info object and add it to this task map. The
	 * <TT>taskID</TT> field is set to the next available task ID. The other
	 * fields are not initialized.
	 *
	 * @return  Task info object.
	 */
	public synchronized TaskInfo add()
		{
		TaskInfo info = new TaskInfo();
		info.taskID = nextTaskID ++;
		taskMap.put (info.taskID, info);
		return info;
		}

	/**
	 * Get the given task info object in this task map. If the given task ID
	 * does not exist, null is returned.
	 *
	 * @param  taskID  Task ID.
	 *
	 * @return  Task info object, or null.
	 */
	public synchronized TaskInfo get
		(long taskID)
		{
		return taskMap.get (taskID);
		}

	/**
	 * Remove the given task info object from this task map. If the given task
	 * ID does not exist, nothing happens and null is returned.
	 *
	 * @param  taskID  Task ID.
	 *
	 * @return  Task info object that was removed, or null.
	 */
	public synchronized TaskInfo remove
		(long taskID)
		{
		return taskMap.remove (taskID);
		}

	/**
	 * Perform the given action on each task info object in this task map.
	 *
	 * @param  action  Action.
	 */
	public synchronized void forEachItemDo
		(Action<TaskInfo> action)
		{
		taskMap.forEachValueDo (action);
		}

	}
