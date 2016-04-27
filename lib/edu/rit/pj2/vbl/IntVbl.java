//******************************************************************************
//
// File:    IntVbl.java
// Package: edu.rit.pj2.vbl
// Unit:    Class edu.rit.pj2.vbl.IntVbl
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

package edu.rit.pj2.vbl;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.pj2.Tuple;
import edu.rit.pj2.Vbl;
import java.io.IOException;

/**
 * Class IntVbl provides an integer reduction variable shared by multiple
 * threads executing a {@linkplain edu.rit.pj2.ParallelStatement
 * ParallelStatement}. An IntVbl is also a {@linkplain Tuple}.
 * <P>
 * Class IntVbl supports the <I>parallel reduction</I> pattern. Each thread
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
 * <LI>Sum -- Class {@linkplain IntVbl.Sum}
 * <LI>Minimum -- Class {@linkplain IntVbl.Min}
 * <LI>Maximum -- Class {@linkplain IntVbl.Max}
 * <LI>Bitwise Boolean and -- Class {@linkplain IntVbl.And}
 * <LI>Bitwise Boolean or -- Class {@linkplain IntVbl.Or}
 * <LI>Bitwise Boolean exclusive-or -- Class {@linkplain IntVbl.Xor}
 * </UL>
 *
 * @author  Alan Kaminsky
 * @version 06-Nov-2014
 */
public class IntVbl
	extends Tuple
	implements Vbl
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

// Exported data members.

	/**
	 * The shared integer item.
	 */
	public int item;

// Exported constructors.

	/**
	 * Construct a new shared integer variable. The item's initial value is 0.
	 */
	public IntVbl()
		{
		}

	/**
	 * Construct a new shared integer variable with the given initial value.
	 *
	 * @param  value  Initial value.
	 */
	public IntVbl
		(int value)
		{
		this.item = value;
		}

// Exported operations.

	/**
	 * Returns the integer value of this shared variable.
	 * <P>
	 * The base class <TT>intValue()</TT> method returns the {@link #item}
	 * field. A subclass may override the <TT>intValue()</TT> method to
	 * return something else.
	 *
	 * @return  Integer value.
	 */
	public int intValue()
		{
		return item;
		}

	/**
	 * Set this shared variable to the given shared variable. This variable must
	 * be set to a deep copy of the given variable.
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
		this.item = ((IntVbl)vbl).intValue();
		}

	/**
	 * Reduce the given shared variable into this shared variable. The two
	 * variables are combined together, and the result is stored in this shared
	 * variable. The <TT>reduce()</TT> method does not need to be multiple
	 * thread safe (thread synchronization is handled by the caller).
	 * <P>
	 * The IntVbl base class's <TT>reduce()</TT> method leaves this shared
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
	 * Returns a string version of this shared variable.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		return "" + intValue();
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
		out.writeInt (item);
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
		item = in.readInt();
		}

// Exported subclasses.

	/**
	 * Class IntVbl.Sum provides an integer reduction variable, with addition as
	 * the reduction operation.
	 *
	 * @author  Alan Kaminsky
	 * @version 13-Sep-2013
	 */
	public static class Sum
		extends IntVbl
		{
		/**
		 * Construct a new shared integer variable. The item's initial value is
		 * 0.
		 */
		public Sum()
			{
			super();
			}

		/**
		 * Construct a new shared integer variable with the given initial value.
		 *
		 * @param  value  Initial value.
		 */
		public Sum
			(int value)
			{
			super (value);
			}

		/**
		 * Reduce the given shared variable into this shared variable. The two
		 * items are combined together using the addition operation, and the
		 * result is stored in this shared variable.
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
			this.item += ((IntVbl)vbl).intValue();
			}
		}

	/**
	 * Class IntVbl.Min provides an integer reduction variable, with minimum as
	 * the reduction operation.
	 *
	 * @author  Alan Kaminsky
	 * @version 13-Sep-2013
	 */
	public static class Min
		extends IntVbl
		{
		/**
		 * Construct a new shared integer variable. The item's initial value is
		 * 0.
		 */
		public Min()
			{
			super();
			}

		/**
		 * Construct a new shared integer variable with the given initial value.
		 *
		 * @param  value  Initial value.
		 */
		public Min
			(int value)
			{
			super (value);
			}

		/**
		 * Reduce the given shared variable into this shared variable. The two
		 * items are combined together using the minimum operation, and the
		 * result is stored in this shared variable.
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
			this.item = Math.min (this.item, ((IntVbl)vbl).intValue());
			}
		}

	/**
	 * Class IntVbl.Max provides an integer reduction variable, with maximum as
	 * the reduction operation.
	 *
	 * @author  Alan Kaminsky
	 * @version 13-Sep-2013
	 */
	public static class Max
		extends IntVbl
		{
		/**
		 * Construct a new shared integer variable. The item's initial value is
		 * 0.
		 */
		public Max()
			{
			super();
			}

		/**
		 * Construct a new shared integer variable with the given initial value.
		 *
		 * @param  value  Initial value.
		 */
		public Max
			(int value)
			{
			super (value);
			}

		/**
		 * Reduce the given shared variable into this shared variable. The two
		 * items are combined together using the maximum operation, and the
		 * result is stored in this shared variable.
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
			this.item = Math.max (this.item, ((IntVbl)vbl).intValue());
			}
		}

	/**
	 * Class IntVbl.And provides an integer reduction variable, with bitwise
	 * Boolean and as the reduction operation.
	 *
	 * @author  Alan Kaminsky
	 * @version 13-Sep-2013
	 */
	public static class And
		extends IntVbl
		{
		/**
		 * Construct a new shared integer variable. The item's initial value is
		 * 0.
		 */
		public And()
			{
			super();
			}

		/**
		 * Construct a new shared integer variable with the given initial value.
		 *
		 * @param  value  Initial value.
		 */
		public And
			(int value)
			{
			super (value);
			}

		/**
		 * Reduce the given shared variable into this shared variable. The two
		 * items are combined together using the bitwise Boolean and operation,
		 * and the result is stored in this shared variable.
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
			this.item &= ((IntVbl)vbl).intValue();
			}
		}

	/**
	 * Class IntVbl.Or provides an integer reduction variable, with bitwise
	 * Boolean or as the reduction operation.
	 *
	 * @author  Alan Kaminsky
	 * @version 13-Sep-2013
	 */
	public static class Or
		extends IntVbl
		{
		/**
		 * Construct a new shared integer variable. The item's initial value is
		 * 0.
		 */
		public Or()
			{
			super();
			}

		/**
		 * Construct a new shared integer variable with the given initial value.
		 *
		 * @param  value  Initial value.
		 */
		public Or
			(int value)
			{
			super (value);
			}

		/**
		 * Reduce the given shared variable into this shared variable. The two
		 * items are combined together using the bitwise Boolean or operation,
		 * and the result is stored in this shared variable.
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
			this.item |= ((IntVbl)vbl).intValue();
			}
		}

	/**
	 * Class IntVbl.Xor provides an integer reduction variable, with bitwise
	 * Boolean exclusive-or as the reduction operation.
	 *
	 * @author  Alan Kaminsky
	 * @version 13-Sep-2013
	 */
	public static class Xor
		extends IntVbl
		{
		/**
		 * Construct a new shared integer variable. The item's initial value is
		 * 0.
		 */
		public Xor()
			{
			super();
			}

		/**
		 * Construct a new shared integer variable with the given initial value.
		 *
		 * @param  value  Initial value.
		 */
		public Xor
			(int value)
			{
			super (value);
			}

		/**
		 * Reduce the given shared variable into this shared variable. The two
		 * items are combined together using the bitwise Boolean exclusive-or
		 * operation, and the result is stored in this shared variable.
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
			this.item ^= ((IntVbl)vbl).intValue();
			}
		}

	}
