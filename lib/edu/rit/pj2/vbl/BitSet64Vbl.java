//******************************************************************************
//
// File:    BitSet64Vbl.java
// Package: edu.rit.pj2.vbl
// Unit:    Class edu.rit.pj2.vbl.BitSet64Vbl
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
import edu.rit.util.BitSet64;
import java.io.IOException;

/**
 * Class BitSet64Vbl provides a reduction variable for a set of integers from 0
 * to 63 shared by multiple threads executing a {@linkplain
 * edu.rit.pj2.ParallelStatement ParallelStatement}. Class BitSet64Vbl is a
 * {@linkplain Tuple} wrapping an instance of class {@linkplain BitSet64}, which
 * is stored in the {@link #bitset bitset} field.
 * <P>
 * Class BitSet64Vbl supports the <I>parallel reduction</I> pattern. Each thread
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
 * <LI>Minimum size -- Class {@linkplain BitSet64Vbl.MinSize}
 * <LI>Maximum size -- Class {@linkplain BitSet64Vbl.MaxSize}
 * <LI>Set union -- Class {@linkplain BitSet64Vbl.Union}
 * <LI>Set intersection -- Class {@linkplain BitSet64Vbl.Intersection}
 * </UL>
 *
 * @author  Alan Kaminsky
 * @version 25-Mar-2015
 */
public class BitSet64Vbl
	extends Tuple
	implements Vbl
	{

// Exported data members.

	/**
	 * The bitset itself.
	 */
	public BitSet64 bitset;

// Exported constructors.

	/**
	 * Construct a new bitset reduction variable wrapping an empty bitset.
	 */
	public BitSet64Vbl()
		{
		this.bitset = new BitSet64();
		}

	/**
	 * Construct a new bitset reduction variable wrapping the given bitset.
	 */
	public BitSet64Vbl
		(BitSet64 bitset)
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
		BitSet64Vbl vbl = (BitSet64Vbl) super.clone();
		vbl.bitset = new BitSet64 (this.bitset);
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
		this.bitset.copy (((BitSet64Vbl)vbl).bitset);
		}

	/**
	 * Reduce the given shared variable into this shared variable. The two
	 * variables are combined together, and the result is stored in this shared
	 * variable. The <TT>reduce()</TT> method does not need to be multiple
	 * thread safe (thread synchronization is handled by the caller).
	 * <P>
	 * The BitSet64Vbl base class's <TT>reduce()</TT> method leaves this shared
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
		bitset = (BitSet64) in.readObject();
		}

// Exported classes.

	/**
	 * Class BitSet64Vbl.MinSize provides a reduction variable for a set of
	 * integers from 0 to 63, where the reduction operation is to keep the set
	 * with the smallest size. The set elements are stored in a bitmap
	 * representation.
	 *
	 * @author  Alan Kaminsky
	 * @version 25-Mar-2015
	 */
	public static class MinSize
		extends BitSet64Vbl
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
			(BitSet64 bitset)
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
		 * The BitSet64Vbl.MinSize class's <TT>reduce()</TT> method changes this
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
			BitSet64Vbl v = (BitSet64Vbl)vbl;
			if (v.bitset.size() < this.bitset.size())
				this.bitset.copy (v.bitset);
			}
		}

	/**
	 * Class BitSet64Vbl.MaxSize provides a reduction variable for a set of
	 * integers from 0 to 63, where the reduction operation is to keep the set
	 * with the largest size. The set elements are stored in a bitmap
	 * representation.
	 *
	 * @author  Alan Kaminsky
	 * @version 25-Mar-2015
	 */
	public static class MaxSize
		extends BitSet64Vbl
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
			(BitSet64 bitset)
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
		 * The BitSet64Vbl.MaxSize class's <TT>reduce()</TT> method changes this
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
			BitSet64Vbl v = (BitSet64Vbl)vbl;
			if (v.bitset.size() > this.bitset.size())
				this.bitset.copy (v.bitset);
			}
		}

	/**
	 * Class BitSet64Vbl.Union provides a reduction variable for a set of
	 * integers from 0 to 63, where the reduction operation is set union. The
	 * set elements are stored in a bitmap representation.
	 *
	 * @author  Alan Kaminsky
	 * @version 25-Mar-2015
	 */
	public static class Union
		extends BitSet64Vbl
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
			(BitSet64 bitset)
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
		 * The BitSet64Vbl.Union class's <TT>reduce()</TT> method changes this
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
			this.bitset.union (((BitSet64Vbl)vbl).bitset);
			}
		}

	/**
	 * Class BitSet64Vbl.Intersection provides a reduction variable for a set of
	 * integers from 0 to 63, where the reduction operation is set intersection.
	 * The set elements are stored in a bitmap representation.
	 *
	 * @author  Alan Kaminsky
	 * @version 25-Mar-2015
	 */
	public static class Intersection
		extends BitSet64Vbl
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
			(BitSet64 bitset)
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
		 * The BitSet64Vbl.Intersection class's <TT>reduce()</TT> method changes
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
			this.bitset.intersection (((BitSet64Vbl)vbl).bitset);
			}
		}

	}
