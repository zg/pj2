//******************************************************************************
//
// File:    LoopBody.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.LoopBody
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

/**
 * Class LoopBody is the abstract base class for a loop body executed by a
 * parallel for loop.
 *
 * @see  ParallelStatement
 * @see  ParallelForLoop
 * @see  Loop
 * @see  LongLoop
 * @see  ObjectLoop
 * @see  Schedule
 *
 * @author  Alan Kaminsky
 * @version 20-Jan-2014
 */
public abstract class LoopBody
	implements Cloneable
	{

// Kludge to avert false sharing in multithreaded programs.

	// Padding fields.
	volatile long p0 = 1000L;
	volatile long p1 = 1001L;
	volatile long p2 = 1002L;
	volatile long p3 = 1003L;
	volatile long p4 = 1004L;
	volatile long p5 = 1005L;
	volatile long p6 = 1006L;
	volatile long p7 = 1007L;
	volatile long p8 = 1008L;
	volatile long p9 = 1009L;
	volatile long pa = 1010L;
	volatile long pb = 1011L;
	volatile long pc = 1012L;
	volatile long pd = 1013L;
	volatile long pe = 1014L;
	volatile long pf = 1015L;

	// Method to prevent the JDK from optimizing away the padding fields.
	long preventOptimization()
		{
		return p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 +
			p8 + p9 + pa + pb + pc + pd + pe + pf;
		}

// Hidden data members.

	ParallelForLoop parallelForLoop;
	int rank;
	ReductionMap reductionMap;

// Exported constructors.

	/**
	 * Construct a new loop object.
	 */
	public LoopBody()
		{
		}

// Exported operations.

	/**
	 * Perform one-time initialization actions for this loop object.
	 * <P>
	 * The <TT>start()</TT> method may be overridden in a subclass. The default
	 * <TT>start()</TT> method does nothing.
	 *
	 * @exception  Exception
	 *     The <TT>start()</TT> method may throw any exception.
	 */
	public void start()
		throws Exception
		{
		}

	/**
	 * Perform one-time finalization actions for this loop object.
	 * <P>
	 * The <TT>finish()</TT> method may be overridden in a subclass. The default
	 * <TT>finish()</TT> method does nothing.
	 *
	 * @exception  Exception
	 *     The <TT>finish()</TT> method may throw any exception.
	 */
	public void finish()
		throws Exception
		{
		}

	/**
	 * Get a thread-local copy of the given global shared variable. The global
	 * variable and the local variable are instances of a subclass of class
	 * {@linkplain Vbl}. The loop can update the local variable without needing
	 * to synchronize with other threads. When the loop finishes execution, all
	 * the local copies are automatically reduced together using the {@linkplain
	 * Vbl} subclass's {@link Vbl#reduce(Vbl) reduce()} method, and the result
	 * is stored into the global variable (overwriting whatever was there).
	 * <P>
	 * The <TT>threadLocal()</TT> method may only be called from the
	 * <TT>start()</TT>, <TT>run()</TT>, or <TT>finish()</TT> methods.
	 *
	 * @param  <V>     Shared variable data type.
	 * @param  global  Global shared variable.
	 *
	 * @return  Thread-local copy of <TT>global</TT>.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if the <TT>threadLocal()</TT> method is
	 *     not called from the <TT>start()</TT>, <TT>run()</TT>, or
	 *     <TT>finish()</TT> methods.
	 */
	public <V extends Vbl> V threadLocal
		(V global)
		{
		if (reductionMap == null)
			throw new IllegalStateException();
		return (V) reductionMap.add (global);
		}

	/**
	 * Stop this loop (early loop exit). Loop iterations will no longer be
	 * performed in any thread executing this loop object.
	 * <P>
	 * The <TT>stop()</TT> method may only be called from the
	 * <TT>start()</TT>, <TT>run()</TT>, or <TT>finish()</TT> methods.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if the <TT>stop()</TT> method is not
	 *     called from the <TT>start()</TT>, <TT>run()</TT>, or
	 *     <TT>finish()</TT> methods.
	 */
	public void stop()
		{
		if (parallelForLoop == null)
			throw new IllegalStateException();
		parallelForLoop.stop = true;
		}

	/**
	 * Returns the number of threads in the thread team executing this loop
	 * object.
	 * <P>
	 * The <TT>threads()</TT> method may only be called from the
	 * <TT>start()</TT>, <TT>run()</TT>, or <TT>finish()</TT> methods.
	 *
	 * @return  Number of threads.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if the <TT>threads()</TT> method is not
	 *     called from the <TT>start()</TT>, <TT>run()</TT>, or
	 *     <TT>finish()</TT> methods.
	 */
	public int threads()
		{
		if (parallelForLoop == null)
			throw new IllegalStateException();
		return parallelForLoop.threads;
		}

	/**
	 * Returns the rank of the thread executing this loop object.
	 * <P>
	 * The <TT>rank()</TT> method may only be called from the
	 * <TT>start()</TT>, <TT>run()</TT>, or <TT>finish()</TT> methods.
	 *
	 * @return  Rank in the range 0 .. {@link #threads() threads()}&minus;1.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if the <TT>rank()</TT> method is not
	 *     called from the <TT>start()</TT>, <TT>run()</TT>, or
	 *     <TT>finish()</TT> methods.
	 */
	public int rank()
		{
		if (rank == -1)
			throw new IllegalStateException();
		return rank;
		}

	/**
	 * Create a clone of this loop object.
	 *
	 * @return  The cloned object.
	 */
	public Object clone()
		{
		try
			{
			return super.clone();
			}
		catch (CloneNotSupportedException exc)
			{
			throw new TerminateException ("Shouldn't happen", exc);
			}
		}

	}
