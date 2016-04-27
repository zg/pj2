//******************************************************************************
//
// File:    Int96.java
// Package: edu.rit.numeric
// Unit:    Class edu.rit.numeric.Int96
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

package edu.rit.numeric;

/**
 * Class Int96 provides a fixed-size 96-bit signed integer.
 * <P>
 * Int96 objects are mutable. Many of the methods alter the state of the Int96
 * object on which they are invoked. This avoids creation of unnecessary Int96
 * objects when performing a series of operations. However, if necessary, you
 * must explicitly create a new Int96 object upon which to perform a series of
 * operations.
 * <P>
 * Class Int96 defines the {@link #equals(Object) equals()} and {@link
 * #hashCode() hashCode()} methods, so Int96 objects can be used as keys in
 * hashed data structures. However, be careful not to alter the state of an
 * Int96 object once it is used as a key. Also, never alter the state of the
 * predefined {@link #ZERO}, {@link #ONE}, {@link #TWO}, and {@link #TEN}
 * objects.
 * <P>
 * Class Int96 is not multiple thread safe. However, each constructor and method
 * only alters the Int96 object on which the constructor or method is called.
 * The constructors and methods do not alter their arguments.
 *
 * @author  Alan Kaminsky
 * @version 18-Jan-2014
 */
public class Int96
	implements Comparable<Int96>
	{

// Hidden data members.

	// 96-bit two's complement value.
	private int x0; // Least significant word
	private int x1;
	private int x2; // Most significant word

	// Mask for least significant word of a long.
	private static final long LSW = 0xFFFFFFFFL;

// Exported constants.

	/**
	 * 96-bit integer with the value 0.
	 */
	public static final Int96 ZERO = Int96.of(0);

	/**
	 * 96-bit integer with the value 1.
	 */
	public static final Int96 ONE = Int96.of(1);

	/**
	 * 96-bit integer with the value 2.
	 */
	public static final Int96 TWO = Int96.of(2);

	/**
	 * 96-bit integer with the value 10.
	 */
	public static final Int96 TEN = Int96.of(10);

// Exported constructors.

	/**
	 * Construct a new integer. Its value is 0.
	 */
	public Int96()
		{
		}

	/**
	 * Construct a new integer with the given <TT>int</TT> value.
	 *
	 * @param  val  Value.
	 */
	public Int96
		(int val)
		{
		assign (val);
		}

	/**
	 * Construct a new integer with the given <TT>long</TT> value.
	 *
	 * @param  val  Value.
	 */
	public Int96
		(long val)
		{
		assign (val);
		}

	/**
	 * Construct a new integer with the given Int96 value.
	 *
	 * @param  val  Value.
	 */
	public Int96
		(Int96 val)
		{
		assign (val);
		}

// Exported operations.

	/**
	 * Returns a new integer with the given <TT>int</TT> value.
	 *
	 * @param  val  Value.
	 *
	 * @return  New integer, with its value set to <TT>val</TT>.
	 */
	public static Int96 of
		(int val)
		{
		return new Int96 (val);
		}

	/**
	 * Returns a new integer with the given <TT>long</TT> value.
	 *
	 * @param  val  Value.
	 *
	 * @return  New integer, with its value set to <TT>val</TT>.
	 */
	public static Int96 of
		(long val)
		{
		return new Int96 (val);
		}

	/**
	 * Returns a new integer with the given Int96 value.
	 *
	 * @param  val  Value.
	 *
	 * @return  New integer, with its value set to <TT>val</TT>.
	 */
	public static Int96 of
		(Int96 val)
		{
		return new Int96 (val);
		}

	/**
	 * Returns this integer converted to type <TT>int</TT>. The most significant
	 * 64 bits are discarded.
	 *
	 * @return  <TT>int</TT> value.
	 */
	public int intval()
		{
		return x0;
		}

	/**
	 * Returns this integer converted to type <TT>long</TT>. The most
	 * significant 32 bits are discarded.
	 *
	 * @return  <TT>long</TT> value.
	 */
	public long longval()
		{
		return ((x1 & LSW) << 32) | (x0 & LSW);
		}

	/**
	 * Set this integer to the given <TT>int</TT> value.
	 *
	 * @param  val  Value.
	 *
	 * @return  This integer, with its value set to <TT>val</TT>.
	 */
	public Int96 assign
		(int val)
		{
		x0 = val;
		x1 = x2 = val >> 31;
		return this;
		}

	/**
	 * Set this integer to the given <TT>long</TT> value.
	 *
	 * @param  val  Value.
	 *
	 * @return  This integer, with its value set to <TT>val</TT>.
	 */
	public Int96 assign
		(long val)
		{
		x0 = (int)(val);
		x1 = (int)(val >> 32);
		x2 = (int)(val >> 63);
		return this;
		}

	/**
	 * Set this integer to the given Int96 value.
	 *
	 * @param  val  Value.
	 *
	 * @return  This integer, with its value set to <TT>val</TT>.
	 */
	public Int96 assign
		(Int96 val)
		{
		x0 = val.x0;
		x1 = val.x1;
		x2 = val.x2;
		return this;
		}

	/**
	 * Negate this integer.
	 *
	 * @return  This integer, with its value set to <TT>-this</TT>.
	 */
	public Int96 neg()
		{
		long sum;
		sum = (~x0 & LSW) + 1L;
		x0 = (int)sum;
		sum >>>= 32;
		sum += (~x1 & LSW);
		x1 = (int)sum;
		sum >>>= 32;
		sum += (~x2 & LSW);
		x2 = (int)sum;
		return this;
		}

	/**
	 * Set this integer to the sum of itself and the given value.
	 *
	 * @param  val  Value.
	 *
	 * @return  This integer, with its value set to <TT>this+val</TT>.
	 */
	public Int96 add
		(Int96 val)
		{
		long sum;
		sum = (x0 & LSW) + (val.x0 & LSW);
		x0 = (int)sum;
		sum >>>= 32;
		sum += (x1 & LSW) + (val.x1 & LSW);
		x1 = (int)sum;
		sum >>>= 32;
		sum += (x2 & LSW) + (val.x2 & LSW);
		x2 = (int)sum;
		return this;
		}

	/**
	 * Set this integer to itself incremented by 1.
	 *
	 * @return  This integer, with its value set to <TT>this+1</TT>.
	 */
	public Int96 incr()
		{
		long sum;
		sum = (x0 & LSW) + 1L;
		x0 = (int)sum;
		sum >>>= 32;
		sum += (x1 & LSW);
		x1 = (int)sum;
		sum >>>= 32;
		sum += (x2 & LSW);
		x2 = (int)sum;
		return this;
		}

	/**
	 * Set this integer to the difference of itself and the given value.
	 *
	 * @param  val  Value.
	 *
	 * @return  This integer, with its value set to <TT>this-val</TT>.
	 */
	public Int96 sub
		(Int96 val)
		{
		long sum;
		sum = (x0 & LSW) + (~val.x0 & LSW) + 1L;
		x0 = (int)sum;
		sum >>>= 32;
		sum += (x1 & LSW) + (~val.x1 & LSW);
		x1 = (int)sum;
		sum >>>= 32;
		sum += (x2 & LSW) + (~val.x2 & LSW);
		x2 = (int)sum;
		return this;
		}

	/**
	 * Set this integer to itself decremented by 1.
	 *
	 * @return  This integer, with its value set to <TT>this-1</TT>.
	 */
	public Int96 decr()
		{
		long sum;
		sum = (x0 & LSW) + LSW;
		x0 = (int)sum;
		sum >>>= 32;
		sum += (x1 & LSW) + LSW;
		x1 = (int)sum;
		sum >>>= 32;
		sum += (x2 & LSW) + LSW;
		x2 = (int)sum;
		return this;
		}

	/**
	 * Set this integer to the product of itself and the given value.
	 *
	 * @param  val  Value.
	 *
	 * @return  This integer, with its value set to <TT>this*val</TT>.
	 */
	public Int96 mul
		(Int96 val)
		{
		Int96 def = Int96.of(val);
		boolean negprod = false;
		if (x2 < 0) { negprod = ! negprod; neg(); }
		if (def.x2 < 0) { negprod = ! negprod; def.neg(); }
		long a = x2 & LSW;
		long b = x1 & LSW;
		long c = x0 & LSW;
		long d = def.x2 & LSW;
		long e = def.x1 & LSW;
		long f = def.x0 & LSW;
		long cf = c*f;
		long bf = b*f;
		long af = a*f;
		long ce = c*e;
		long be = b*e;
		long cd = c*d;
		long sum;
		x0 = (int)cf;
		sum = (cf >>> 32) + (bf & LSW) + (ce & LSW);
		x1 = (int)sum;
		sum >>>= 32;
		sum += (bf >>> 32) + (af & LSW) + (ce >>> 32) + (be & LSW) + (cd & LSW);
		x2 = (int)sum;
		if (negprod) neg();
		return this;
		}

	/**
	 * Set this integer to the quotient of itself and the given value. The
	 * remainder is discarded.
	 *
	 * @param  val  Value.
	 *
	 * @return  This integer, with its value set to <TT>this/val</TT>.
	 *
	 * @exception  ArithmeticException
	 *     (unchecked exception) Thrown if <TT>val</TT> is 0.
	 */
	public Int96 div
		(Int96 val)
		{
		if (val.eqzero())
			throw new ArithmeticException ("Int96.div(): Divide by zero");

		Int96 numer = this;
		Int96 denom = Int96.of (val);
		Int96 quot = new Int96();
		boolean negquot = false;
		if (numer.x2 < 0) { negquot = ! negquot; numer.neg(); }
		if (denom.x2 < 0) { negquot = ! negquot; denom.neg(); }

		int shift = 0;
		while (numer.compareTo (denom) > 0)
			{
			denom.lsh();
			++ shift;
			}
		do
			{
			quot.lsh();
			if (numer.compareTo (denom) >= 0)
				{
				quot.incr();
				numer.sub (denom);
				}
			denom.rsh();
			-- shift;
			}
		while (shift >= 0);

		assign (quot);
		if (negquot) neg();
		return this;
		}

	/**
	 * Set this integer to itself left-shifted one bit.
	 *
	 * @return  This integer, with its value set to <TT>this&lt;&lt;1</TT>.
	 */
	public Int96 lsh()
		{
		x2 = (x2 << 1) | (x1 >>> 31);
		x1 = (x1 << 1) | (x0 >>> 31);
		x0 <<= 1;
		return this;
		}

	/**
	 * Set this integer to itself right-shifted one bit.
	 *
	 * @return  This integer, with its value set to <TT>this&gt;&gt;1</TT>.
	 */
	public Int96 rsh()
		{
		x0 = (x0 >>> 1) | (x1 << 31);
		x1 = (x1 >>> 1) | (x2 << 31);
		x2 >>= 1;
		return this;
		}

	/**
	 * Set this integer to itself unsigned-right-shifted one bit.
	 *
	 * @return  This integer, with its value set to <TT>this&gt;&gt;&gt;1</TT>.
	 */
	public Int96 ursh()
		{
		x0 = (x0 >>> 1) | (x1 << 31);
		x1 = (x1 >>> 1) | (x2 << 31);
		x2 >>>= 1;
		return this;
		}

	/**
	 * Set this integer to the smaller of itself and the given value.
	 *
	 * @param  val  Value.
	 *
	 * @return  This integer, with its value set to <TT>min(this,val)</TT>.
	 */
	public Int96 min
		(Int96 val)
		{
		if (compareTo (val) > 0) assign (val);
		return this;
		}

	/**
	 * Set this integer to the larger of itself and the given value.
	 *
	 * @param  val  Value.
	 *
	 * @return  This integer, with its value set to <TT>max(this,val)</TT>.
	 */
	public Int96 max
		(Int96 val)
		{
		if (compareTo (val) < 0) assign (val);
		return this;
		}

	/**
	 * Determine if this integer equals 0.
	 *
	 * @return  True if this integer equals 0, false if it doesn't.
	 */
	public boolean eqzero()
		{
		return x0 == 0 && x1 == 0 && x2 == 0;
		}

	/**
	 * Compare this integer to the given integer.
	 *
	 * @param  val  Value.
	 *
	 * @return  A number less than, equal to, or greater than 0 if this integer
	 *          is less than, equal to, or greater than <TT>val</TT>,
	 *          respectively.
	 */
	public int compareTo
		(Int96 val)
		{
		long sum, d0, d1, d2;
		sum = (x0 & LSW) + (~val.x0 & LSW) + 1L;
		d0 = (int)sum;
		sum >>>= 32;
		sum += (x1 & LSW) + (~val.x1 & LSW);
		d1 = (int)sum;
		sum >>>= 32;
		sum += (x2 & LSW) + (~val.x2 & LSW);
		d2 = (int)sum;
		if (d0 == 0 && d1 == 0 && d2 == 0)
			return 0;
		else if (d2 >= 0)
			return 1;
		else
			return -1;
		}

	/**
	 * Determine if this integer equals the given object.
	 * <P>
	 * <I>Note:</I> Class Int96 provides the <TT>equals()</TT> and
	 * <TT>hashCode()</TT> methods, and Int96 objects can be used as keys in
	 * hashed data structures. However, Int96 objects are mutable. If an Int96
	 * object is used as a hash key, be sure not to change its value.
	 *
	 * @param  obj  Object to compare.
	 *
	 * @return  True if this integer equals <TT>obj</TT>, false otherwise.
	 */
	public boolean equals
		(Object obj)
		{
		if (! (obj instanceof Int96)) return false;
		Int96 val = (Int96)obj;
		return x0 == val.x0 && x1 == val.x1 && x2 == val.x2;
		}

	/**
	 * Returns a hash code for this integer.
	 * <P>
	 * <I>Note:</I> Class Int96 provides the <TT>equals()</TT> and
	 * <TT>hashCode()</TT> methods, and Int96 objects can be used as keys in
	 * hashed data structures. However, Int96 objects are mutable. If an Int96
	 * object is used as a hash key, be sure not to change its value.
	 *
	 * @return  Hash code.
	 */
	public int hashCode()
		{
		return ((x2*31) + x1)*31 + x0;
		}

	/**
	 * Returns a string version of this integer. The string is a 24-digit
	 * hexadecimal number.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		return String.format ("%08x%08x%08x", x2, x1, x0);
		}

// Unit test main program.

//	/**
//	 * Unit test main program.
//	 */
//	public static void main
//		(String[] args)
//		{
//		if (args.length != 2) usage();
//		Int96 a = Int96.of (Long.parseLong (args[0]));
//		Int96 b = Int96.of (Long.parseLong (args[1]));
//		System.out.printf ("a\t%s%n", a);
//		System.out.printf ("b\t%s%n", b);
//		System.out.printf ("-a\t%s%n", Int96.of(a) .neg());
//		System.out.printf ("-b\t%s%n", Int96.of(b) .neg());
//		System.out.printf ("a+b\t%s%n", Int96.of(a) .add(b));
//		System.out.printf ("a-b\t%s%n", Int96.of(a) .sub(b));
//		System.out.printf ("a*b\t%s%n", Int96.of(a) .mul(b));
//		System.out.printf ("a/b\t%s%n", Int96.of(a) .div(b));
//		System.out.printf ("a?b\t%s%n", a.compareTo (b));
//		System.out.printf ("a+1\t%s%n", Int96.of(a) .incr());
//		System.out.printf ("a-1\t%s%n", Int96.of(a) .decr());
//		System.out.printf ("a<<1\t%s%n", Int96.of(a) .lsh());
//		System.out.printf ("a>>1\t%s%n", Int96.of(a) .rsh());
//		System.out.printf ("a>>>1\t%s%n", Int96.of(a) .ursh());
//		System.out.printf ("b+1\t%s%n", Int96.of(b) .incr());
//		System.out.printf ("b-1\t%s%n", Int96.of(b) .decr());
//		System.out.printf ("b<<1\t%s%n", Int96.of(b) .lsh());
//		System.out.printf ("b>>1\t%s%n", Int96.of(b) .rsh());
//		System.out.printf ("b>>>1\t%s%n", Int96.of(b) .ursh());
//		}
//
//	/**
//	 * Print a usage message and exit.
//	 */
//	private static void usage()
//		{
//		System.err.println ("Usage: java edu.rit.numeric.Int96 <a> <b>");
//		System.exit (1);
//		}

	}
