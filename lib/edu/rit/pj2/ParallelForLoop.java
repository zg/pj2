//******************************************************************************
//
// File:    ParallelForLoop.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.ParallelForLoop
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

package edu.rit.pj2;

import edu.rit.pj2.tracker.TaskProperties;

/**
 * Class ParallelForLoop is the abstract base class for a work sharing parallel
 * for loop executed by multiple threads. A parallel for loop is constructed by
 * the <TT>parallelFor()</TT> method of class {@linkplain Task}. For further
 * information, refer to subclasses {@linkplain IntParallelForLoop},
 * {@linkplain LongParallelForLoop}, and {@linkplain ObjectParallelForLoop}.
 *
 * @see  ParallelStatement
 * @see  IntParallelForLoop
 * @see  LongParallelForLoop
 * @see  ObjectParallelForLoop
 * @see  Loop
 * @see  LongLoop
 * @see  ObjectLoop
 *
 * @author  Alan Kaminsky
 * @version 23-Mar-2014
 */
public abstract class ParallelForLoop
	extends ParallelStatement
	{

// Hidden data members.

	TaskProperties properties;
	int threads;
	Schedule schedule;
	int chunk;
	volatile boolean stop;

// Hidden constructors.

	/**
	 * Construct a new parallel for loop.
	 *
	 * @param  task  Task in which the parallel for loop is executing.
	 */
	ParallelForLoop
		(Task task)
		{
		super (task);
		properties = new TaskProperties() .chain (task.properties);
		}

	}
