//******************************************************************************
//
// File:    LoopProperties.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.LoopProperties
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

import edu.rit.pj2.Schedule;
import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import java.io.IOException;

/**
 * Class LoopProperties encapsulates the properties of a parallel loop. The
 * following properties may be specified:
 * <UL>
 * <LI><TT>threads</TT> &mdash; The number of threads executing the parallel
 * loop.
 * <LI><TT>schedule</TT> &mdash; The parallel loop schedule.
 * <LI><TT>chunk</TT> &mdash; The parallel loop chunk size.
 * </UL>
 *
 * @author  Alan Kaminsky
 * @version 23-Mar-2014
 */
public class LoopProperties
	implements Streamable
	{

// Exported constants.

	/**
	 * Indicates that the <TT>threads</TT> property is defaulted.
	 */
	public static final int DEFAULT_THREADS = -1;

	/**
	 * Indicates that the <TT>schedule</TT> property is defaulted.
	 */
	public static final Schedule DEFAULT_SCHEDULE = null;

	/**
	 * Indicates that the <TT>chunk</TT> property is defaulted.
	 */
	public static final int DEFAULT_CHUNK = -1;

	/**
	 * Indicates that a parallel for loop will be executed by as many threads as
	 * there are cores on the machine.
	 */
	public static final int THREADS_EQUALS_CORES = 0;

	/**
	 * Indicates to use the standard chunk size for the <TT>schedule</TT>
	 * property.
	 */
	public static final int STANDARD_CHUNK = 0;

// Hidden data members.

	int threads = DEFAULT_THREADS;
	Schedule schedule = DEFAULT_SCHEDULE;
	int chunk = DEFAULT_CHUNK;

// Exported constructors.

	/**
	 * Construct a new loop properties object. All settings are defaulted.
	 */
	public LoopProperties()
		{
		}

	/**
	 * Construct a new loop properties object with the given settings.
	 *
	 * @param  threads   Number of threads for a parallel for loop, {@link
	 *                   #THREADS_EQUALS_CORES}, or {@link #DEFAULT_THREADS}.
	 * @param  schedule  Schedule for a parallel for loop, or {@link
	 *                   #DEFAULT_SCHEDULE}.
	 * @param  chunk     Chunk size for a parallel for loop, {@link
	 *                   #STANDARD_CHUNK}, or {@link #DEFAULT_CHUNK}.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>threads</TT> or <TT>chunk</TT> is
	 *     illegal.
	 */
	public LoopProperties
		(int threads,
		 Schedule schedule,
		 int chunk)
		{
		this.threads (threads);
		this.schedule (schedule);
		this.chunk (chunk);
		}

	/**
	 * Construct a new loop properties object that is a copy of the given loop
	 * properties object.
	 *
	 * @param  props  Loop properties object to copy.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>props</TT> is null.
	 */
	public LoopProperties
		(LoopProperties props)
		{
		this.threads (props.threads);
		this.schedule (props.schedule);
		this.chunk (props.chunk);
		}

// Exported operations.

	/**
	 * Set the <TT>threads</TT> property. The <TT>threads</TT> property
	 * specifies the number of threads that will execute a parallel for loop.
	 *
	 * @param  threads  Number of threads (&ge; 1), {@link
	 *                  #THREADS_EQUALS_CORES}, or {@link #DEFAULT_THREADS}.
	 *
	 * @return  This loop properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>threads</TT> is illegal.
	 *
	 * @see  #threads()
	 * @see  #actualThreads()
	 */
	public LoopProperties threads
		(int threads)
		{
		if (threads < DEFAULT_THREADS)
			throw new IllegalArgumentException (String.format
				("LoopProperties.threads(): threads = %d illegal", threads));
		this.threads = threads;
		return this;
		}

	/**
	 * Get the <TT>threads</TT> property. The <TT>threads</TT> property
	 * specifies the number of threads that will execute a parallel for loop. If
	 * the <TT>threads</TT> property is defaulted, {@link #THREADS_EQUALS_CORES}
	 * is returned, indicating that a parallel for loop will be executed by as
	 * many threads as there are cores on the machine.
	 * <P>
	 * <I>Note:</I> The <TT>threads()</TT> method returns the <TT>threads</TT>
	 * property <I>setting,</I> which may be {@link #THREADS_EQUALS_CORES}. For
	 * the actual number of threads, see the {@link #actualThreads()
	 * actualThreads()} method.
	 *
	 * @return  Number of threads (&ge; 1), or {@link #THREADS_EQUALS_CORES}.
	 *
	 * @see  #threads(int)
	 * @see  #actualThreads()
	 */
	public int threads()
		{
		return threads == DEFAULT_THREADS ? THREADS_EQUALS_CORES : threads;
		}

	/**
	 * Get the number of threads that will execute a parallel for loop.
	 * <P>
	 * <I>Note:</I> If the <TT>threads</TT> property is {@link
	 * #THREADS_EQUALS_CORES}, the <TT>actualThreads()</TT> method returns the
	 * actual number of threads that will execute a parallel for loop, namely
	 * the number of cores on the machine. For the <TT>threads</TT> property
	 * setting, see the {@link #threads() threads()} method.
	 *
	 * @return  Number of threads (&ge; 1).
	 *
	 * @see  #threads(int)
	 * @see  #threads()
	 */
	public int actualThreads()
		{
		int rv = threads();
		return
			rv == THREADS_EQUALS_CORES ?
				Runtime.getRuntime().availableProcessors() :
				rv;
		}

	/**
	 * Set the <TT>schedule</TT> property. The <TT>schedule</TT> property, along
	 * with the <TT>chunk</TT> property, specifies how the iterations of a
	 * parallel for loop are partitioned among the threads executing the
	 * parallel for loop. Refer to enum {@linkplain Schedule} for descriptions
	 * of the possible schedules.
	 *
	 * @param  schedule  Parallel for loop schedule.
	 *
	 * @return  This loop properties object.
	 *
	 * @see  #schedule()
	 * @see  #chunk(int)
	 * @see  #chunk()
	 */
	public LoopProperties schedule
		(Schedule schedule)
		{
		this.schedule = schedule;
		return this;
		}

	/**
	 * Get the <TT>schedule</TT> property. The <TT>schedule</TT> property, along
	 * with the <TT>chunk</TT> property, specifies how the iterations of a
	 * parallel for loop are partitioned among the threads executing the
	 * parallel for loop. If the <TT>schedule</TT> property is defaulted, a
	 * {@link Schedule#fixed fixed} schedule is returned.
	 *
	 * @return  Parallel for loop schedule.
	 *
	 * @see  #schedule(Schedule)
	 * @see  #chunk(int)
	 * @see  #chunk()
	 */
	public Schedule schedule()
		{
		return schedule == DEFAULT_SCHEDULE ? Schedule.fixed : schedule;
		}

	/**
	 * Set the <TT>chunk</TT> property. The <TT>chunk</TT> property, along with
	 * the <TT>schedule</TT> property, specifies how the iterations of a
	 * parallel for loop are partitioned among the threads executing the
	 * parallel for loop. Refer to enum {@linkplain Schedule} for descriptions
	 * of the possible schedules.
	 *
	 * @param  chunk  Chunk size (&ge; 1), {@link #STANDARD_CHUNK}, or {@link
	 *                #DEFAULT_CHUNK}.
	 *
	 * @return  This loop properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>chunk</TT> is illegal.
	 *
	 * @see  #schedule(Schedule)
	 * @see  #schedule()
	 * @see  #chunk()
	 */
	public LoopProperties chunk
		(int chunk)
		{
		if (chunk < DEFAULT_CHUNK)
			throw new IllegalArgumentException (String.format
				("LoopProperties.chunk(): chunk = %d illegal", chunk));
		this.chunk = chunk;
		return this;
		}

	/**
	 * Get the <TT>chunk</TT> property. The <TT>chunk</TT> property, along with
	 * the <TT>schedule</TT> property, specifies how the iterations of a
	 * parallel for loop are partitioned among the threads executing the
	 * parallel for loop. If the <TT>chunk</TT> property is defaulted, {@link
	 * #STANDARD_CHUNK} is returned, indicating the standard chunk size for the
	 * <TT>schedule</TT> property; see enum {@linkplain Schedule Schedule} for
	 * further information.
	 *
	 * @return  Chunk size (&ge; 1), or {@link #STANDARD_CHUNK}.
	 *
	 * @see  #schedule(Schedule)
	 * @see  #schedule()
	 * @see  #chunk(int)
	 */
	public int chunk()
		{
		return chunk == DEFAULT_CHUNK ? STANDARD_CHUNK : chunk;
		}

	/**
	 * Write this object's fields to the given out stream.
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
		out.writeInt (threads);
		Schedule.writeOut (schedule, out);
		out.writeInt (chunk);
		}

	/**
	 * Read this object's fields from the given in stream.
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
		threads = in.readInt();
		schedule = Schedule.readIn (in);
		chunk = in.readInt();
		}

	/**
	 * Returns a string version of this loop properties object.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		return String.format
			("LoopProperties(threads=%d,schedule=%s,chunk=%d)",
			 threads(), schedule(), chunk());
		}

	}
