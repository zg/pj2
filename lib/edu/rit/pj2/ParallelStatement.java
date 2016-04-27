//******************************************************************************
//
// File:    ParallelStatement.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.ParallelStatement
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
 * Class ParallelStatement is the abstract base class for a parallel statement
 * executed by multiple threads. A parallel statement is constructed by the
 * <TT>parallelFor()</TT>, <TT>parallelDo()</TT>, or <TT>parallelWhile()</TT>
 * methods of class {@linkplain Task}.
 *
 * @see  ParallelForLoop
 * @see  IntParallelForLoop
 * @see  LongParallelForLoop
 * @see  ObjectParallelForLoop
 * @see  Loop
 * @see  LongLoop
 * @see  ObjectLoop
 *
 * @author  Alan Kaminsky
 * @version 31-Dec-2013
 */
public abstract class ParallelStatement
	{

// Hidden data members.

	// Task in which the parallel statement is executing.
	Task task;

// Hidden constructors.

	/**
	 * Construct a new parallel statement.
	 *
	 * @param  task  Task in which the parallel statement is executing.
	 */
	ParallelStatement
		(Task task)
		{
		this.task = task;
		}

// Hidden operations.

	/**
	 * Execute this parallel statement.
	 *
	 * @param  rank          Rank of the team thread.
	 * @param  reductionMap  Reduction map of the team thread.
	 *
	 * @exception  Exception
	 *     The <TT>run()</TT> method may throw any exception.
	 */
	abstract void run
		(int rank,
		 ReductionMap reductionMap)
		throws Exception;

	}
