//******************************************************************************
//
// File:    BitSet32Vbl.java
// Package: edu.rit.pj2.vbl
// Unit:    Class edu.rit.pj2.vbl.BitSet32Vbl
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
import edu.rit.util.BitSet32;
import java.io.IOException;

/**
 * Class BitSet32Vbl provides a reduction variable for a set of integers from 0
 * to 31 shared by multiple threads executing a {@linkplain
 * edu.rit.pj2.ParallelStatement ParallelStatement}. Class BitSet32Vbl is a
 * {@linkplain Tuple} wrapping an instance of class {@linkplain BitSet32}, which
 * is stored in the {@link #bitset bitset} field.
 * <P>
 * Class BitSet32Vbl supports the <I>parallel reduction</I> pattern. Each thread
 * creates a thread-local copy of the shared variable by calling the {@link
 * edu.rit.pj2.Loop#threadLocal(Vbl) threadLocal()} method of class {@linkplain
 * edu.rit.pj2.Loop Loop} or the {@link edu.rit.pj2.Section#threadLocal(Vbl)
 * threadLocal()} method of class {@linkplain edu.rit.pj2.Section Section}. Each
 * thread performs operations on its own copy, without needing to synchronize
 * with the other threads. At the end of the parallel statement, the
 * thread-local copies are automatically <I>reduced</I> together, and the result
 * is stored in the original shared variable. The reduction is performed by the
 * shared variable's {@link #reduce(Vbl) reduce()} method.
 * <P>
 * The following subclasses provide various predefined reduction operations. You
 * can also define your own subclasses with customized reduction operations.
 * <UL>
 * <LI>Minimum size -- Class {@linkplain BitSet32Vbl.MinSize}
 * <LI>Maximum size -- Class {@linkplain BitSet32Vbl.MaxSize}
 * <LI>Set union -- Class {@linkplain BitSet32Vbl.Union}
 * <LI>Set intersection -- Class {@linkplain BitSet32Vbl.Intersection}
 * </UL>
 *
 * @author  Alan Kaminsky
 * @version 25-Mar-2015
 */
public class BitSet32Vbl
	extends Tuple
	implements Vbl
	{

// Exported data members.

	/**
	 * The bitset itself.
	 */
	public BitSet32 bitset;

// Exported constructors.

	/**
	 * Construct a new bitset reduction variable wrapping an empty bitset.
	 */
	public BitSet32Vbl()
		{
		this.bitset = new BitSet32();
		}

	/**
	 * Construct a new bitset reduction variable wrapping the given bitset.
	 */
	public BitSet32Vbl
		(BitSet32 bitset)
		{
		this.bitset = bitset;
		}

// Exported operations.

	/**
	 * Create a clone of this shared variable.
	 *
	 * @return  The cloned object.
	 */
	public Object clone()
		{
		BitSet32Vbl vbl = (BitSet32Vbl) super.clone();
		vbl.bitset = new BitSet32 (this.bitset);
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
		this.bitset.copy (((BitSet32Vbl)vbl).bitset);
		}

	/**
	 * Reduce the given shared variable into this shared variable. The two
	 * variables are combined together, and the result is stored in this shared
	 * variable. The <TT>reduce()</TT> method does not need to be multiple
	 * thread safe (thread synchronization is handled by the caller).
	 * <P>
	 * The BitSet32Vbl base class's <TT>reduce()</TT> method leaves this shared
	 * variable unchanged.
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
		}

	/**
	 * Write this bitset reduction variable to the given out stream.
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
		out.writeObject (bitset);
		}

	/**
	 * Read this bitset reduction variable from the given in stream.
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
		bitset = (BitSet32) in.readObject();
		}

// Exported classes.

	/**
	 * Class BitSet32Vbl.MinSize provides a reduction variable for a set of
	 * integers from 0 to 31, where the reduction operation is to keep the set
	 * with the smallest size. The set elements are stored in a bitmap
	 * representation.
	 *
	 * @author  Alan Kaminsky
	 * @version 25-Mar-2015
	 */
	public static class MinSize
		extends BitSet32Vbl
		{

		/**
		 * Construct a new bitset reduction variable wrapping an empty bitset.
		 */
		public MinSize()
			{
			super();
			}

		/**
		 * Construct a new bitset reduction variable wrapping the given bitset.
		 */
		public MinSize
			(BitSet32 bitset)
			{
			super (bitset);
			}

		/**
		 * Reduce the given shared variable into this shared variable. The two
		 * variables are combined together, and the result is stored in this
		 * shared variable. The <TT>reduce()</TT> method does not need to be
		 * multiple thread safe (thread synchronization is handled by the
		 * caller).
		 * <P>
		 * The BitSet32Vbl.MinSize class's <TT>reduce()</TT> method changes this
		 * variable's {@link #bitset bitset} to be a copy of the given
		 * variable's {@link #bitset bitset} if the latter's size is smaller.
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
			BitSet32Vbl v = (BitSet32Vbl)vbl;
			if (v.bitset.size() < this.bitset.size())
				this.bitset.copy (v.bitset);
			}
		}

	/**
	 * Class BitSet32Vbl.MaxSize provides a reduction variable for a set of
	 * integers from 0 to 31, where the reduction operation is to keep the set
	 * with the largest size. The set elements are stored in a bitmap
	 * representation.
	 *
	 * @author  Alan Kaminsky
	 * @version 25-Mar-2015
	 */
	public static class MaxSize
		extends BitSet32Vbl
		{

		/**
		 * Construct a new bitset reduction variable wrapping an empty bitset.
		 */
		public MaxSize()
			{
			super();
			}

		/**
		 * Construct a new bitset reduction variable wrapping the given bitset.
		 */
		public MaxSize
			(BitSet32 bitset)
			{
			super (bitset);
			}

		/**
		 * Reduce the given shared variable into this shared variable. The two
		 * variables are combined together, and the result is stored in this
		 * shared variable. The <TT>reduce()</TT> method does not need to be
		 * multiple thread safe (thread synchronization is handled by the
		 * caller).
		 * <P>
		 * The BitSet32Vbl.MaxSize class's <TT>reduce()</TT> method changes this
		 * variable's {@link #bitset bitset} to be a copy of the given
		 * variable's {@link #bitset bitset} if the latter's size is larger.
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
			BitSet32Vbl v = (BitSet32Vbl)vbl;
			if (v.bitset.size() > this.bitset.size())
				this.bitset.copy (v.bitset);
			}
		}

	/**
	 * Class BitSet32Vbl.Union provides a reduction variable for a set of
	 * integers from 0 to 31, where the reduction operation is set union. The
	 * set elements are stored in a bitmap representation.
	 *
	 * @author  Alan Kaminsky
	 * @version 25-Mar-2015
	 */
	public static class Union
		extends BitSet32Vbl
		{

		/**
		 * Construct a new bitset reduction variable wrapping an empty bitset.
		 */
		public Union()
			{
			super();
			}

		/**
		 * Construct a new bitset reduction variable wrapping the given bitset.
		 */
		public Union
			(BitSet32 bitset)
			{
			super (bitset);
			}

		/**
		 * Reduce the given shared variable into this shared variable. The two
		 * variables are combined together, and the result is stored in this
		 * shared variable. The <TT>reduce()</TT> method does not need to be
		 * multiple thread safe (thread synchronization is handled by the
		 * caller).
		 * <P>
		 * The BitSet32Vbl.Union class's <TT>reduce()</TT> method changes this
		 * variable's {@link #bitset bitset} to be the set union of itself with
		 * the given variable's {@link #bitset bitset}.
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
			this.bitset.union (((BitSet32Vbl)vbl).bitset);
			}
		}

	/**
	 * Class BitSet32Vbl.Intersection provides a reduction variable for a set of
	 * integers from 0 to 31, where the reduction operation is set intersection.
	 * The set elements are stored in a bitmap representation.
	 *
	 * @author  Alan Kaminsky
	 * @version 25-Mar-2015
	 */
	public static class Intersection
		extends BitSet32Vbl
		{

		/**
		 * Construct a new bitset reduction variable wrapping an empty bitset.
		 */
		public Intersection()
			{
			super();
			}

		/**
		 * Construct a new bitset reduction variable wrapping the given bitset.
		 */
		public Intersection
			(BitSet32 bitset)
			{
			super (bitset);
			}

		/**
		 * Reduce the given shared variable into this shared variable. The two
		 * variables are combined together, and the result is stored in this
		 * shared variable. The <TT>reduce()</TT> method does not need to be
		 * multiple thread safe (thread synchronization is handled by the
		 * caller).
		 * <P>
		 * The BitSet32Vbl.Intersection class's <TT>reduce()</TT> method changes
		 * this variable's {@link #bitset bitset} to be the set intersection of
		 * itself with the given variable's {@link #bitset bitset}.
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
			this.bitset.intersection (((BitSet32Vbl)vbl).bitset);
			}
		}

	}
