//******************************************************************************
//
// File:    FloatVbl.java
// Package: edu.rit.pj2.vbl
// Unit:    Class edu.rit.pj2.vbl.FloatVbl
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
 * Class FloatVbl provides a single precision floating point reduction variable
 * shared by multiple threads executing a {@linkplain
 * edu.rit.pj2.ParallelStatement ParallelStatement}. A FloatVbl is also a
 * {@linkplain Tuple}.
 * <P>
 * Class FloatVbl supports the <I>parallel reduction</I> pattern. Each thread
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
 * <LI>Sum -- Class {@linkplain FloatVbl.Sum}
 * <LI>Minimum -- Class {@linkplain FloatVbl.Min}
 * <LI>Maximum -- Class {@linkplain FloatVbl.Max}
 * <LI>Mean -- Class {@linkplain FloatVbl.Mean}
 * </UL>
 *
 * @author  Alan Kaminsky
 * @version 06-Nov-2014
 */
public class FloatVbl
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
	 * The shared single precision floating point item.
	 */
	public float item;

// Exported constructors.

	/**
	 * Construct a new shared single precision floating point variable. The
	 * item's initial value is 0.0f.
	 */
	public FloatVbl()
		{
		}

	/**
	 * Construct a new shared single precision floating point variable with the
	 * given initial value.
	 *
	 * @param  value  Initial value.
	 */
	public FloatVbl
		(float value)
		{
		this.item = value;
		}

// Exported operations.

	/**
	 * Returns the single precision floating point value of this shared
	 * variable.
	 * <P>
	 * The base class <TT>floatValue()</TT> method returns the {@link #item}
	 * field. A subclass may override the <TT>floatValue()</TT> method to return
	 * something else.
	 *
	 * @return  Float value.
	 */
	public float floatValue()
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
		this.item = ((FloatVbl)vbl).floatValue();
		}

	/**
	 * Reduce the given shared variable into this shared variable. The two
	 * variables are combined together, and the result is stored in this shared
	 * variable. The <TT>reduce()</TT> method does not need to be multiple
	 * thread safe (thread synchronization is handled by the caller).
	 * <P>
	 * The FloatVbl base class's <TT>reduce()</TT> method leaves this shared
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
		return "" + floatValue();
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
		out.writeFloat (item);
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
		item = in.readFloat();
		}

// Exported subclasses.

	/**
	 * Class FloatVbl.Sum provides a single precision floating point reduction
	 * variable, with addition as the reduction operation.
	 *
	 * @author  Alan Kaminsky
	 * @version 13-Sep-2013
	 */
	public static class Sum
		extends FloatVbl
		{
		/**
		 * Construct a new shared single precision floating point variable. The
		 * item's initial value is 0.0f.
		 */
		public Sum()
			{
			super();
			}

		/**
		 * Construct a new shared single precision floating point variable with
		 * the given initial value.
		 *
		 * @param  value  Initial value.
		 */
		public Sum
			(float value)
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
			this.item += ((FloatVbl)vbl).floatValue();
			}
		}

	/**
	 * Class FloatVbl.Min provides a single precision floating point reduction
	 * variable, with minimum as the reduction operation.
	 *
	 * @author  Alan Kaminsky
	 * @version 13-Sep-2013
	 */
	public static class Min
		extends FloatVbl
		{
		/**
		 * Construct a new shared single precision floating point variable. The
		 * item's initial value is 0.0f.
		 */
		public Min()
			{
			super();
			}

		/**
		 * Construct a new shared single precision floating point variable with
		 * the given initial value.
		 *
		 * @param  value  Initial value.
		 */
		public Min
			(float value)
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
			this.item = Math.min (this.item, ((FloatVbl)vbl).floatValue());
			}
		}

	/**
	 * Class FloatVbl.Max provides a single precision floating point reduction
	 * variable, with maximum as the reduction operation.
	 *
	 * @author  Alan Kaminsky
	 * @version 13-Sep-2013
	 */
	public static class Max
		extends FloatVbl
		{
		/**
		 * Construct a new shared single precision floating point variable. The
		 * item's initial value is 0.0f.
		 */
		public Max()
			{
			super();
			}

		/**
		 * Construct a new shared single precision floating point variable with
		 * the given initial value.
		 *
		 * @param  value  Initial value.
		 */
		public Max
			(float value)
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
			this.item = Math.max (this.item, ((FloatVbl)vbl).floatValue());
			}
		}

	/**
	 * Class FloatVbl.Mean provides a single precision floating point reduction
	 * variable, with mean as the reduction operation.
	 * <P>
	 * Call the {@link #accumulate(float) accumulate()} method or the {@link
	 * #reduce(Vbl) reduce()} method to accumulate a value into the running mean
	 * computation. Call the {@link #floatValue()} method to get the mean of the
	 * values accumulated so far. The {@link #item item} field holds the sum of
	 * the accumulated values. The {@link #count count} field holds the number
	 * of accumulated values.
	 *
	 * @author  Alan Kaminsky
	 * @version 10-Aug-2015
	 */
	public static class Mean
		extends FloatVbl
		{
		/**
		 * The number of accumulated values.
		 */
		public int count;

		/**
		 * Construct a new shared single precision floating point variable. The
		 * item's initial value is 0.0. The count's initial value is 0.
		 */
		public Mean()
			{
			super();
			count = 0;
			}

		/**
		 * Construct a new shared single precision floating point variable. The
		 * item's initial value is the given value. The count's initial value is
		 * 1.
		 *
		 * @param  value  Value.
		 */
		public Mean
			(float value)
			{
			super (value);
			count = 1;
			}

		/**
		 * Accumulate the given value into this shared variable.
		 *
		 * @param  value  Value.
		 *
		 * @return  This shared variable.
		 */
		public Mean accumulate
			(float value)
			{
			item += value;
			++ count;
			return this;
			}

		/**
		 * Returns the single precision floating point value of this shared
		 * variable. The value returned is the mean of the values that have been
		 * accumulated. If no values have been accumulated, <TT>Float.NaN</TT>
		 * is returned.
		 *
		 * @return  Float value.
		 */
		public float floatValue()
			{
			return count == 0 ? Float.NaN : item/count;
			}

		/**
		 * Set this shared variable to the given shared variable. This variable
		 * must be set to a deep copy of the given variable.
		 * <P>
		 * If <TT>vbl</TT> is an instance of class FloatVbl.Mean, then this
		 * shared variable consists of the same values that had been accumulated
		 * into <TT>vbl</TT>. If <TT>vbl</TT> is an instance of class FloatVbl
		 * or a subclass thereof, but not an instance of class FloatVbl.Mean,
		 * then this shared variable consists of just the one value
		 * <TT>vbl.floatValue()</TT>.
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
			if (vbl instanceof Mean)
				{
				Mean meanVbl = (Mean) vbl;
				this.item = meanVbl.item;
				this.count = meanVbl.count;
				}
			else if (vbl instanceof FloatVbl)
				{
				this.item = ((FloatVbl)vbl).floatValue();
				this.count = 1;
				}
			else
				throw new ClassCastException();
			}

		/**
		 * Reduce the given shared variable into this shared variable. The two
		 * items are combined together using the mean operation, and the
		 * result is stored in this shared variable.
		 * <P>
		 * If <TT>vbl</TT> is an instance of class FloatVbl.Mean, then all the
		 * values that had been accumulated into <TT>vbl</TT> are accumulated
		 * into this shared variable. If <TT>vbl</TT> is an instance of class
		 * FloatVbl or a subclass thereof, but not an instance of class
		 * FloatVbl.Mean, then <TT>vbl.floatValue()</TT> is accumulated into
		 * this shared variable.
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
			if (vbl instanceof Mean)
				{
				Mean meanVbl = (Mean) vbl;
				this.item += meanVbl.item;
				this.count += meanVbl.count;
				}
			else if (vbl instanceof FloatVbl)
				accumulate (((FloatVbl)vbl).floatValue());
			else
				throw new ClassCastException();
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
			super.writeOut (out);
			out.writeInt (count);
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
			super.readIn (in);
			count = in.readInt();
			}
		}

	}
