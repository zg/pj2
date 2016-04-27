//******************************************************************************
//
// File:    HistogramVbl.java
// Package: edu.rit.pj2.vbl
// Unit:    Class edu.rit.pj2.vbl.HistogramVbl
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

package edu.rit.pj2.vbl;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.pj2.Tuple;
import edu.rit.pj2.Vbl;
import edu.rit.numeric.Histogram;
import java.io.IOException;

/**
 * Class HistogramVbl provides a reduction variable for a histogram shared by
 * multiple threads executing a {@linkplain edu.rit.pj2.ParallelStatement
 * ParallelStatement}. Class HistogramVbl is a {@linkplain Tuple} wrapping an
 * instance of class {@linkplain Histogram}, which is stored in the {@link #hist
 * hist} field.
 * <P>
 * Class HistogramVbl supports the <I>parallel reduction</I> pattern. Each
 * thread creates a thread-local copy of the shared variable by calling the
 * {@link edu.rit.pj2.Loop#threadLocal(Vbl) threadLocal()} method of class
 * {@linkplain edu.rit.pj2.Loop Loop} or the {@link
 * edu.rit.pj2.Section#threadLocal(Vbl) threadLocal()} method of class
 * {@linkplain edu.rit.pj2.Section Section}. Each thread performs operations on
 * its own copy, without needing to synchronize with the other threads. At the
 * end of the parallel statement, the thread-local copies are automatically
 * <I>reduced</I> together, and the result is stored in the original shared
 * variable. The reduction is performed by the shared variable's {@link
 * #reduce(Vbl) reduce()} method.
 *
 * @author  Alan Kaminsky
 * @version 26-Mar-2015
 */
public class HistogramVbl
	extends Tuple
	implements Vbl
	{

// Exported data members.

	/**
	 * The histogram itself.
	 */
	public Histogram hist;

// Exported constructors.

	/**
	 * Construct a new histogram reduction variable wrapping a null histogram.
	 */
	public HistogramVbl()
		{
		}

	/**
	 * Construct a new histogram reduction variable wrapping the given
	 * histogram.
	 */
	public HistogramVbl
		(Histogram hist)
		{
		this.hist = hist;
		}

// Exported operations.

	/**
	 * Create a clone of this shared variable.
	 *
	 * @return  The cloned object.
	 */
	public Object clone()
		{
		HistogramVbl vbl = (HistogramVbl) super.clone();
		if (this.hist != null)
			vbl.hist = (Histogram) this.hist.clone();
		return vbl;
		}

	/**
	 * Set this shared variable to the given shared variable.
	 *
	 * @param  vbl  Shared variable.
	 *
	 * @exception  ClassCastException
	 *     (unchecked exception) Thrown if the class of <TT>vbl</TT> is not
	 *     compatible with the class of this shared variable.
	 */
	public void set
		(Vbl vbl)
		{
		this.hist.copy (((HistogramVbl)vbl).hist);
		}

	/**
	 * Reduce the given shared variable into this shared variable. The two
	 * variables are combined together, and the result is stored in this shared
	 * variable. The <TT>reduce()</TT> method does not need to be multiple
	 * thread safe (thread synchronization is handled by the caller).
	 * <P>
	 * Class HistogramVbl's <TT>reduce()</TT> method adds the given variable's
	 * histogram into this variable's histogram.
	 *
	 * @param  vbl  Shared variable.
	 *
	 * @exception  ClassCastException
	 *     (unchecked exception) Thrown if the class of <TT>vbl</TT> is not
	 *     compatible with the class of this shared variable.
	 */
	public void reduce
		(Vbl vbl)
		{
		this.hist.add (((HistogramVbl)vbl).hist);
		}

	/**
	 * Write this histogram reduction variable to the given out stream.
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
		out.writeObject (hist);
		}

	/**
	 * Read this histogram reduction variable from the given in stream.
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
		hist = (Histogram) in.readObject();
		}

	}
