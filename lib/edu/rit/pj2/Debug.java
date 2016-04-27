//******************************************************************************
//
// File:    Debug.java
// Package: edu.rit.pj2
// Unit:    Enum edu.rit.pj2.Debug
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

package edu.rit.pj2;

/**
 * Enum Debug specifies various debugging messages that can be turned on or off
 * in a PJ2 {@linkplain Job Job}.
 *
 * @see  Job
 *
 * @author  Alan Kaminsky
 * @version 05-Oct-2013
 */
public enum Debug
	{

// Exported enumerals.

	/**
	 * Debug printout when the job is launched. Includes the job ID, time, and
	 * date. On by default.
	 */
	jobLaunch,

	/**
	 * Debug printout when the job's first task starts executing. Includes the
	 * job ID, time, and date. On by default.
	 */
	jobStart,

	/**
	 * Debug printout when a task is launched. Includes the job ID, task ID,
	 * time, and date. Off by default.
	 */
	taskLaunch,

	/**
	 * Debug printout of a task's class name and command line arguments when a
	 * task is launched. Off by default.
	 */
	taskArguments,

	/**
	 * Debug printout of a task's input tuples. This includes tuples the task
	 * takes or reads from tuple space, as well as the matching tuples that
	 * fired an on-demand task. Off by default.
	 */
	taskInputTuples,

	/**
	 * Debug printout when a task starts executing. Includes the job ID, task
	 * ID, time, date, and node. Off by default.
	 */
	taskStart,

	/**
	 * Debug printout when a task finishes executing. Includes the job ID, task
	 * ID, time, date, and elapsed time from task start to task finish. Off by
	 * default.
	 */
	taskFinish,

	/**
	 * Debug printout of a task's output tuples. This includes tuples the task
	 * puts into tuple space. Off by default.
	 */
	taskOutputTuples,

	/**
	 * Debug printout when the job finishes executing. Includes the job ID,
	 * time, date, and elapsed time from job launch to job finish. On by
	 * default.
	 */
	jobFinish,

	/**
	 * Debug printout of any tuples remaining in tuple space when the job
	 * finishes executing. Off by default.
	 */
	remainingTuples,

	/**
	 * Debug printout of the job's makespan. This is the elapsed time from when
	 * the first task actually started until when the last task actually
	 * finished. This might be different from the job's elapsed time if the job
	 * sat in the Tracker queue for a while. Off by default.
	 */
	makespan,

	}
