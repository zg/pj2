//******************************************************************************
//
// File:    BigInteger.java
// Package: edu.rit.numeric
// Unit:    Class edu.rit.numeric.BigInteger
//
// This Java source file is copyright (C) 2016 by Alan Kaminsky. All rights
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

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import edu.rit.util.Random;
import java.io.IOException;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;

/**
 * Class BigInteger encapsulates an unsigned fixed-precision integer. A big
 * integer's bit size is specified when it is constructed. The big integer's bit
 * size remains the same for all subsequent operations. Bit positions in the big
 * integer are numbered from 0 (least significant bit) to <I>n</I>&minus;1 (most
 * significant bit), where <I>n</I> is the bit size.
 * <P>
 * <B>Bit truncation and extension.</B>
 * When a value of fewer than <I>n</I> bits is stored in a big integer of
 * <I>n</I> bits, the extra most significant bits of the big integer are set to
 * zero. When a value of more than <I>n</I> bits is stored in a big integer of
 * <I>n</I> bits, the extra most significant bits of the value are omitted.
 * <P>
 * <B>Comparison with java.math.BigInteger.</B>
 * Class edu.rit.numeric.BigInteger differs from the Java Platform class
 * {@linkplain java.math.BigInteger java.math.BigInteger} in these ways:
 * <UL>
 * <P><LI>
 * Instances of class java.math.BigInteger are immutable; every method
 * constructs a new instance to hold the result. Instances of class
 * edu.rit.numeric.BigInteger are mutable; every operation stores the result in
 * the instance on which the method was called. (This helps avoid the
 * performance reduction from incessant constructor execution and garbage
 * collection when a program does numerous computations on instances of class
 * java.math.BigInteger.)
 * <P><LI>
 * An instance of class java.math.BigInteger can have any number of bits,
 * depending on the value. An instance of class edu.rit.numeric.BigInteger
 * always has a fixed number of bits. Consequently, an arithmetic operation will
 * overflow if the answer is too large to fit in the bit size of the big integer
 * that will hold the result.
 * <P><LI>
 * Instances of class java.math.BigInteger are signed. Instances of class
 * edu.rit.numeric.BigInteger are unsigned.
 * </UL>
 * <P>
 * <B>Hashed data structure keys.</B>
 * Class BigInteger provides the {@link #equals(Object) equals()} and {@link
 * #hashCode() hashCode()} methods, so a big integer can be used as a key in a
 * hashed data structure. However, big integers are mutable. If a big integer is
 * used as a key, make sure not to alter the big integer's value.
 * <P>
 * <B>Multiple threads.</B>
 * <I>Note:</I> Class BigInteger is not multiple thread safe.
 *
 * @author  Alan Kaminsky
 * @version 06-Apr-2016
 */
public class BigInteger
	implements Comparable<BigInteger>, Streamable, Cloneable
	{

// Exported helper interface.

	/**
	 * Interface BigInteger.Bits specifies the interface for a group of bit
	 * positions to be extracted from a big integer.
	 *
	 * @see  BigInteger#extractBits(BigInteger,Bits)
	 *
	 * @author  Alan Kaminsky
	 * @version 19-Nov-2015
	 */
	public static interface Bits
		{
		/**
		 * Returns the number of bits to be extracted.
		 *
		 * @return  Number of bits.
		 */
		public int size();

		/**
		 * Returns the <TT>i</TT>-th bit position to be extracted.
		 *
		 * @param  i  Index, 0 &le; <TT>i</TT> &le; <TT>size()</TT>&minus;1.
		 *
		 * @return  <TT>i</TT>-th bit position.
		 *
		 * @exception  IndexOutOfBoundsException
		 *     (unchecked exception) Thrown if <TT>i</TT> is out of bounds.
		 */
		public int bit
			(int i);
		}

// Hidden data members.

	private static final long MASK_32 = 0x00000000ffffffffL;

	private int bitSize; // Number of bits in the big integer
	private int[] value; // Value of the big integer
	private int[] tmp;   // Temporary storage used by various operations
	private int[] quo;   // Temporary storage for quotient

// Exported constructors.

	/**
	 * Construct a new uninitialized big integer. This constructor is for use
	 * only by object streaming.
	 */
	private BigInteger()
		{
		}

	/**
	 * Construct a new big integer with the given bit size and a value of 0.
	 *
	 * @param  bitSize  Bit size.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>bitSize</TT> &lt; 1.
	 */
	public BigInteger
		(int bitSize)
		{
		if (bitSize < 1)
			throw new IllegalArgumentException (String.format
				("BigInteger(): bitSize = %d illegal", bitSize));
		this.bitSize = bitSize;
		this.value = new int [wordSize(bitSize)];
		}

	/**
	 * Construct a new big integer that is a copy of the given big integer.
	 *
	 * @param  bigint  Big integer to copy.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>bigint</TT> is null.
	 */
	public BigInteger
		(BigInteger bigint)
		{
		this.bitSize = bigint.bitSize;
		this.value = (int[]) bigint.value.clone();
		}

	/**
	 * Returns a clone of this big integer.
	 *
	 * @return  Clone.
	 */
	public Object clone()
		{
		try
			{
			BigInteger rv = (BigInteger) super.clone();
			rv.value = (int[]) this.value.clone();
			rv.tmp = null;
			rv.quo = null;
			return rv;
			}
		catch (CloneNotSupportedException exc)
			{
			throw new IllegalStateException ("Shouldn't happen", exc);
			}
		}

// Exported operations.

	/**
	 * Returns the size of this big integer in bits.
	 *
	 * @return  Bit size.
	 */
	public int bitSize()
		{
		return bitSize;
		}

	/**
	 * Returns the size of this big integer in 32-bit words.
	 *
	 * @return  Word size.
	 */
	public int wordSize()
		{
		return value.length;
		}

	/**
	 * Returns the word size of a big integer with the given bit size. The word
	 * size is <TT>bitSize</TT>/32 rounded up to the nearest integer, except the
	 * minimum word size is 2.
	 *
	 * @param  bitSize  Bit size.
	 *
	 * @return  Word size.
	 */
	public static int wordSize
		(int bitSize)
		{
		return Math.max ((bitSize + 31) >>> 5, 2);
		}

	/**
	 * Convert this big integer to an unsigned <TT>int</TT>.
	 *
	 * @return  Unsigned integer value.
	 */
	public int intValue()
		{
		return value[0];
		}

	/**
	 * Convert this big integer to an unsigned <TT>long</TT>.
	 *
	 * @return  Unsigned long integer value.
	 */
	public long longValue()
		{
		return ((long)value[1] << 32) | (value[0] & MASK_32);
		}

	/**
	 * Convert this big integer to a <TT>float</TT>.
	 *
	 * @return  Float value.
	 */
	public float floatValue()
		{
		return (float) doubleValue();
		}

	/**
	 * Convert this big integer to a <TT>double</TT>.
	 *
	 * @return  Double value.
	 */
	public double doubleValue()
		{
		double rv = 0.0;
		for (int i = value.length - 1; i >= 0; -- i)
			rv = rv*4294967296.0 + (double)(value[i] & MASK_32);
		return rv;
		}

	/**
	 * Assign the value 0 to this big integer.
	 *
	 * @return  This big integer.
	 */
	public BigInteger assignZero()
		{
		clearWords (value, 0);
		return this;
		}

	/**
	 * Assign the given unsigned <TT>int</TT> value to this big integer.
	 *
	 * @param  v  <TT>int</TT> value.
	 *
	 * @return  This big integer.
	 */
	public BigInteger assign
		(int v)
		{
		copyWord (v, value);
		clearUnusedBits();
		return this;
		}

	/**
	 * Assign the given unsigned <TT>long</TT> value to this big integer.
	 *
	 * @param  v  <TT>long</TT> value.
	 *
	 * @return  This big integer.
	 */
	public BigInteger assign
		(long v)
		{
		copyWords (v, value);
		clearUnusedBits();
		return this;
		}

	/**
	 * Assign the given big integer to this big integer.
	 *
	 * @param  bigint  Big integer to assign.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>bigint</TT> is null.
	 */
	public BigInteger assign
		(BigInteger bigint)
		{
		copyWords (bigint.value, this.value);
		clearUnusedBits();
		return this;
		}

	/**
	 * Assign to this big integer the value parsed from the given decimal
	 * string.
	 *
	 * @param  s  String.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>s</TT> is null.
	 * @exception  NumberFormatException
	 *     (unchecked exception) Thrown if <TT>s</TT> is zero-length. Thrown if
	 *     <TT>s</TT> contains characters other than decimal digits.
	 */
	public BigInteger assign
		(String s)
		{
		int n = s.length();
		if (n == 0)
			throw new NumberFormatException
				("BigInteger.assign(): s is zero-length");
		assignZero();
		for (int i = 0; i < n; ++ i)
			mul (10) .add (charToDecimalDigit (s.charAt (i)));
		clearUnusedBits();
		return this;
		}

	private static int charToDecimalDigit
		(char c)
		{
		if ('0' <= c && c <= '9')
			return c - '0';
		else
			throw new NumberFormatException (String.format
				("BigInteger.assign(): Illegal digit '%c'", c));
		}

	/**
	 * Assign to this big integer the value parsed from the given hexadecimal
	 * string.
	 *
	 * @param  s  String.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>s</TT> is null.
	 * @exception  NumberFormatException
	 *     (unchecked exception) Thrown if <TT>s</TT> is zero-length. Thrown if
	 *     <TT>s</TT> contains characters other than hexadecimal digits.
	 */
	public BigInteger assignHex
		(String s)
		{
		int n = s.length();
		if (n == 0)
			throw new NumberFormatException
				("BigInteger.assign(): s is zero-length");
		assignZero();
		for (int i = 0; i < n; ++ i)
			leftShift (4) .add (charToHexDigit (s.charAt (i)));
		clearUnusedBits();
		return this;
		}

	private static int charToHexDigit
		(char c)
		{
		if ('0' <= c && c <= '9')
			return c - '0';
		else if ('a' <= c && c <= 'z')
			return c - 'a' + 10;
		else if ('A' <= c && c <= 'Z')
			return c - 'A' + 10;
		else
			throw new NumberFormatException (String.format
				("BigInteger.assign(): Illegal digit '%c'", c));
		}

	/**
	 * Assign to this big integer a random value obtained from the given PRNG.
	 * The value is chosen uniformly at random in the range 0 through
	 * 2<SUP><I>n</I></SUP>&minus;1, where <I>n</I> is this big integer's bit
	 * size.
	 *
	 * @param  prng  Pseudorandom number generator.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>prng</TT> is null.
	 */
	public BigInteger assign
		(Random prng)
		{
		for (int i = 0; i < value.length; ++ i)
			value[i] = prng.nextInteger();
		clearUnusedBits();
		return this;
		}

	/**
	 * Assign the value 2<SUP><I>n</I></SUP> to this big integer.
	 *
	 * @param  n  Power of 2 in the range 0 ..
	 *            {@link #bitSize() bitSize()}&minus;1.
	 *
	 * @return  This big integer.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>n</TT> is out of range.
	 */
	public BigInteger assignPowerOfTwo
		(int n)
		{
		if (0 > n || n >= bitSize)
			throw new IllegalArgumentException (String.format
				("BigInteger.assignPowerOfTwo(): n = %d illegal", n));
		clearWords (value, 0);
		setBit (value, n);
		return this;
		}

	/**
	 * Assign the Mersenne number 2<SUP><I>n</I></SUP>&minus;1 to this big
	 * integer.
	 *
	 * @param  n  Power of 2 in the range 0 ..
	 *            {@link #bitSize() bitSize()}.
	 *
	 * @return  This big integer.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>n</TT> is out of range.
	 */
	public BigInteger assignMersenne
		(int n)
		{
		if (0 > n || n > bitSize)
			throw new IllegalArgumentException (String.format
				("BigInteger.assignMersenne(): n = %d illegal", n));
		clearWords (value, 0);
		for (int i = 0; i < n; ++ i)
			setBit (value, i);
		return this;
		}

	/**
	 * Right-shift this big integer by the given amount. The most significant
	 * bits are filled in with 0s.
	 *
	 * @param  shift  Number of bit positions to right-shift.
	 *
	 * @return  This big integer.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>shift</TT> &lt; 0.
	 */
	public BigInteger rightShift
		(int shift)
		{
		if (shift < 0)
			throw new IllegalArgumentException (String.format
				("BigInteger.rightShift(): shift = %d illegal", shift));
		else if (shift == 0)
			return this;
		else if (shift < bitSize)
			{
			rightShift (value, shift);
			clearUnusedBits();
			return this;
			}
		else
			return assignZero();
		}

	/**
	 * Left-shift this big integer by the given amount. The least significant
	 * bits are filled in with 0s.
	 *
	 * @param  shift  Number of bit positions to left-shift.
	 *
	 * @return  This big integer.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>shift</TT> &lt; 0.
	 */
	public BigInteger leftShift
		(int shift)
		{
		if (shift < 0)
			throw new IllegalArgumentException (String.format
				("BigInteger.leftShift(): shift = %d illegal", shift));
		else if (shift == 0)
			return this;
		else if (shift < bitSize)
			{
			leftShift (value, shift);
			clearUnusedBits();
			return this;
			}
		else
			return assignZero();
		}

	/**
	 * Right-rotate this big integer by the given amount.
	 *
	 * @param  shift  Number of bit positions to right-rotate.
	 *
	 * @return  This big integer.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>shift</TT> &lt; 0.
	 */
	public BigInteger rightRotate
		(int shift)
		{
		if (shift < 0)
			throw new IllegalArgumentException (String.format
				("BigInteger.rightRotate(): shift = %d illegal", shift));
		shift = shift % bitSize;
		if (shift > 0)
			{
			allocateTmp();
			copyWords (value, tmp);
			rightShift (value, shift);
			leftShift (tmp, bitSize - shift);
			bitwiseOr (value, tmp);
			}
		return this;
		}

	/**
	 * Left-rotate this big integer by the given amount.
	 *
	 * @param  shift  Number of bit positions to left-rotate.
	 *
	 * @return  This big integer.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>shift</TT> &lt; 0.
	 */
	public BigInteger leftRotate
		(int shift)
		{
		if (shift < 0)
			throw new IllegalArgumentException (String.format
				("BigInteger.leftRotate(): shift = %d illegal", shift));
		shift = shift % bitSize;
		if (shift > 0)
			{
			allocateTmp();
			copyWords (value, tmp);
			leftShift (value, shift);
			rightShift (tmp, bitSize - shift);
			bitwiseOr (value, tmp);
			}
		return this;
		}

	/**
	 * Get the given bit in this big integer.
	 *
	 * @param  pos  Bit position to get, in the range 0 ..
	 *              {@link #bitSize() bitSize()}&minus;1.
	 *
	 * @return  Bit value, either 0 or 1.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>pos</TT> is out of bounds.
	 */
	public int getBit
		(int pos)
		{
		if (0 > pos || pos >= bitSize)
			throw new IndexOutOfBoundsException (String.format
				("BigInteger.getBit(): pos = %d out of bounds", pos));
		return getBit (value, pos);
		}

	/**
	 * Test the given bit in this big integer.
	 *
	 * @param  pos  Bit position to test, in the range 0 ..
	 *              {@link #bitSize() bitSize()}&minus;1.
	 *
	 * @return  True if the bit value is 1, false if the bit value is 0.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>pos</TT> is out of bounds.
	 */
	public boolean testBit
		(int pos)
		{
		if (0 > pos || pos >= bitSize)
			throw new IndexOutOfBoundsException (String.format
				("BigInteger.testBit(): pos = %d out of bounds", pos));
		return getBit (value, pos) == 1;
		}

	/**
	 * Put the given bit into this big integer.
	 *
	 * @param  pos  Bit position to put, in the range 0 ..
	 *              {@link #bitSize() bitSize()}&minus;1.
	 * @param  bit  Bit value, either 0 or 1.
	 *
	 * @return  This big integer.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>pos</TT> is out of bounds.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>bit</TT> is not 0 or 1.
	 */
	public BigInteger putBit
		(int pos,
		 int bit)
		{
		if (0 > pos || pos >= bitSize)
			throw new IndexOutOfBoundsException (String.format
				("BigInteger.putBit(): pos = %d out of bounds", pos));
		if (bit != 0 && bit != 1)
			throw new IllegalArgumentException (String.format
				("BigInteger.putBit(): bit = %d illegal", bit));
		putBit (value, pos, bit);
		return this;
		}

	/**
	 * Clear the given bit in this big integer. The bit's value becomes 0.
	 *
	 * @param  pos  Bit position to clear, in the range 0 ..
	 *              {@link #bitSize() bitSize()}&minus;1.
	 *
	 * @return  This big integer.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>pos</TT> is out of bounds.
	 */
	public BigInteger clearBit
		(int pos)
		{
		if (0 > pos || pos >= bitSize)
			throw new IndexOutOfBoundsException (String.format
				("BigInteger.clearBit(): pos = %d out of bounds", pos));
		clearBit (value, pos);
		return this;
		}

	/**
	 * Set the given bit in this big integer. The bit's value becomes 1.
	 *
	 * @param  pos  Bit position to set, in the range 0 ..
	 *              {@link #bitSize() bitSize()}&minus;1.
	 *
	 * @return  This big integer.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>pos</TT> is out of bounds.
	 */
	public BigInteger setBit
		(int pos)
		{
		if (0 > pos || pos >= bitSize)
			throw new IndexOutOfBoundsException (String.format
				("BigInteger.setBit(): pos = %d out of bounds", pos));
		setBit (value, pos);
		return this;
		}

	/**
	 * Flip the given bit in this big integer. The bit's value becomes its
	 * opposite.
	 *
	 * @param  pos  Bit position to flip, in the range 0 ..
	 *              {@link #bitSize() bitSize()}&minus;1.
	 *
	 * @return  This big integer.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>pos</TT> is out of bounds.
	 */
	public BigInteger flipBit
		(int pos)
		{
		if (0 > pos || pos >= bitSize)
			throw new IndexOutOfBoundsException (String.format
				("BigInteger.flipBit(): pos = %d out of bounds", pos));
		flipBit (value, pos);
		return this;
		}

	/**
	 * Negate this big integer.
	 *
	 * @return  This big integer.
	 */
	public BigInteger neg()
		{
		return not() .increment();
		}

	/**
	 * Increment this big integer.
	 *
	 * @return  This big integer.
	 */
	public BigInteger increment()
		{
		long sum = 1L;
		for (int i = 0; i < value.length; ++ i)
			{
			sum += value[i] & MASK_32;
			value[i] = (int)sum;
			sum >>>= 32;
			}
		clearUnusedBits();
		return this;
		}

	/**
	 * Decrement this big integer.
	 *
	 * @return  This big integer.
	 */
	public BigInteger decrement()
		{
		long sum = 0L;
		for (int i = 0; i < value.length; ++ i)
			{
			sum += (value[i] & MASK_32) + MASK_32;
			value[i] = (int)sum;
			sum >>>= 32;
			}
		clearUnusedBits();
		return this;
		}

	/**
	 * Add the given unsigned integer to this big integer and assign the result
	 * to this big integer.
	 *
	 * @param  n  Integer to add.
	 *
	 * @return  This big integer.
	 */
	public BigInteger add
		(int n)
		{
		long sum = 0L;
		sum += (value[0] & MASK_32) + (n & MASK_32);
		value[0] = (int)sum;
		sum >>>= 32;
		for (int i = 1; i < value.length; ++ i)
			{
			sum += value[i] & MASK_32;
			value[i] = (int)sum;
			sum >>>= 32;
			}
		clearUnusedBits();
		return this;
		}

	/**
	 * Add the given unsigned long integer to this big integer and assign the
	 * result to this big integer.
	 *
	 * @param  n  Long integer to add.
	 *
	 * @return  This big integer.
	 */
	public BigInteger add
		(long n)
		{
		long sum = 0L;
		sum += (value[0] & MASK_32) + (n & MASK_32);
		value[0] = (int)sum;
		sum >>>= 32;
		sum += (value[1] & MASK_32) + (n >>> 32);
		value[1] = (int)sum;
		sum >>>= 32;
		for (int i = 2; i < value.length; ++ i)
			{
			sum += value[i] & MASK_32;
			value[i] = (int)sum;
			sum >>>= 32;
			}
		clearUnusedBits();
		return this;
		}

	/**
	 * Add the given big integer to this big integer and assign the result to
	 * this big integer.
	 *
	 * @param  bigint  Big integer to add.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>bigint</TT> is null.
	 */
	public BigInteger add
		(BigInteger bigint)
		{
		add (this.value, bigint.value, 0L);
		clearUnusedBits();
		return this;
		}

	/**
	 * Subtract the given unsigned integer from this big integer and assign the
	 * result to this big integer.
	 *
	 * @param  n  Integer to subtract.
	 *
	 * @return  This big integer.
	 */
	public BigInteger sub
		(int n)
		{
		long sum = 1L;
		sum += (value[0] & MASK_32) + ((~ n) & MASK_32);
		value[0] = (int)sum;
		sum >>>= 32;
		for (int i = 1; i < value.length; ++ i)
			{
			sum += (value[i] & MASK_32) + MASK_32;
			value[i] = (int)sum;
			sum >>>= 32;
			}
		clearUnusedBits();
		return this;
		}

	/**
	 * Subtract the given unsigned long integer from this big integer and assign
	 * the result to this big integer.
	 *
	 * @param  n  Long integer to subtract.
	 *
	 * @return  This big integer.
	 */
	public BigInteger sub
		(long n)
		{
		long sum = 1L;
		sum += (value[0] & MASK_32) + ((~ n) & MASK_32);
		value[0] = (int)sum;
		sum >>>= 32;
		sum += (value[1] & MASK_32) + ((~ n) >>> 32);
		value[1] = (int)sum;
		sum >>>= 32;
		for (int i = 2; i < value.length; ++ i)
			{
			sum += (value[i] & MASK_32) + MASK_32;
			value[i] = (int)sum;
			sum >>>= 32;
			}
		clearUnusedBits();
		return this;
		}

	/**
	 * Subtract the given big integer from this big integer and assign the
	 * result to this big integer.
	 *
	 * @param  bigint  Big integer to subtract.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>bigint</TT> is null.
	 */
	public BigInteger sub
		(BigInteger bigint)
		{
		addComplement (this.value, bigint.value, 1L);
		clearUnusedBits();
		return this;
		}

	/**
	 * Multiply the given unsigned integer by this big integer and assign the
	 * result to this big integer.
	 *
	 * @param  n  Integer to multiply.
	 *
	 * @return  This big integer.
	 */
	public BigInteger mul
		(int n)
		{
		allocateTmp();
		clearWords (tmp, 0);
		multiplyAtPosition (0, n);
		copyWords (tmp, value);
		clearUnusedBits();
		return this;
		}

	/**
	 * Multiply the given unsigned long integer by this big integer and assign
	 * the result to this big integer.
	 *
	 * @param  n  Long integer to multiply.
	 *
	 * @return  This big integer.
	 */
	public BigInteger mul
		(long n)
		{
		allocateTmp();
		clearWords (tmp, 0);
		multiplyAtPosition (0, (int)n);
		multiplyAtPosition (1, (int)(n >>> 32));
		copyWords (tmp, value);
		clearUnusedBits();
		return this;
		}

	/**
	 * Multiply the given big integer by this big integer and assign the result
	 * to this big integer.
	 *
	 * @param  bigint  Big integer to multiply.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>bigint</TT> is null.
	 */
	public BigInteger mul
		(BigInteger bigint)
		{
		allocateTmp();
		clearWords (tmp, 0);
		for (int i = 0; i < bigint.value.length; ++ i)
			multiplyAtPosition (i, bigint.value[i]);
		copyWords (tmp, value);
		clearUnusedBits();
		return this;
		}

	private void multiplyAtPosition
		(int posn,
		 int n)
		{
		long n_long = n & MASK_32;
		long prev_hi = 0L;
		long prod, prod_hi, prod_lo;
		long sum = 0L;
		for (int s = 0, d = posn; d < value.length; ++ s, ++ d)
			{
			prod = (value[s] & MASK_32) * n_long;
			prod_hi = prod >>> 32;
			prod_lo = prod & MASK_32;
			sum += (tmp[d] & MASK_32) + prev_hi + prod_lo;
			tmp[d] = (int)sum;
			sum >>>= 32;
			prev_hi = prod_hi;
			}
		}

	/**
	 * Divide this big integer by the given unsigned integer and assign the
	 * quotient to this big integer. The remainder is discarded.
	 *
	 * @param  divisor  Unsigned integer divisor.
	 *
	 * @return  This big integer.
	 *
	 * @exception  ArithmeticException
	 *     (unchecked exception) Thrown if <TT>divisor</TT> is zero.
	 */
	public BigInteger div
		(int divisor)
		{
		if (divisor == 0)
			throw new ArithmeticException
				("BigInteger.div(): Divide by zero");
		allocateTmp();
		copyWord (divisor, tmp);
		int cmp = compare (tmp, value);
		if (cmp > 0)
			assignZero();
		else if (cmp == 0)
			assign (1);
		else
			{
			allocateQuo();
			doDivide (value, tmp, quo);
			copyWords (quo, value);
			}
		return this;
		}

	/**
	 * Divide this big integer by the given unsigned long integer and assign the
	 * quotient to this big integer. The remainder is discarded.
	 *
	 * @param  divisor  Unsigned long integer divisor.
	 *
	 * @return  This big integer.
	 *
	 * @exception  ArithmeticException
	 *     (unchecked exception) Thrown if <TT>divisor</TT> is zero.
	 */
	public BigInteger div
		(long divisor)
		{
		if (divisor == 0L)
			throw new ArithmeticException
				("BigInteger.div(): Divide by zero");
		allocateTmp();
		copyWords (divisor, tmp);
		int cmp = compare (tmp, value);
		if (cmp > 0)
			assignZero();
		else if (cmp == 0)
			assign (1);
		else
			{
			allocateQuo();
			doDivide (value, tmp, quo);
			copyWords (quo, value);
			}
		return this;
		}

	/**
	 * Divide this big integer by the given big integer and assign the quotient
	 * to this big integer. The remainder is discarded.
	 *
	 * @param  divisor  Big integer divisor.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>divisor</TT> is null.
	 * @exception  ArithmeticException
	 *     (unchecked exception) Thrown if <TT>divisor</TT> is zero.
	 */
	public BigInteger div
		(BigInteger divisor)
		{
		if (eqZero (divisor.value))
			throw new ArithmeticException
				("BigInteger.div(): Divide by zero");
		int cmp = compare (divisor.value, this.value);
		if (cmp > 0)
			assignZero();
		else if (cmp == 0)
			assign (1);
		else
			{
			allocateTmp();
			allocateQuo();
			copyWords (divisor.value, this.tmp);
			doDivide (this.value, this.tmp, this.quo);
			copyWords (this.quo, this.value);
			}
		return this;
		}

	/**
	 * Divide this big integer by the given unsigned integer and assign the
	 * quotient to this big integer. The remainder is returned.
	 *
	 * @param  divisor  Unsigned integer divisor.
	 *
	 * @return  Unsigned integer remainder.
	 *
	 * @exception  ArithmeticException
	 *     (unchecked exception) Thrown if <TT>divisor</TT> is zero.
	 */
	public int divRem
		(int divisor)
		{
		if (divisor == 0)
			throw new ArithmeticException
				("BigInteger.divRem(): Divide by zero");
		allocateTmp();
		copyWord (divisor, tmp);
		int rem;
		int cmp = compare (tmp, value);
		if (cmp > 0)
			{
			rem = intValue();
			assignZero();
			}
		else if (cmp == 0)
			{
			rem = 0;
			assign (1);
			}
		else
			{
			allocateQuo();
			doDivide (value, tmp, quo);
			rem = intValue();
			copyWords (quo, value);
			}
		return rem;
		}

	/**
	 * Divide this big integer by the given unsigned long integer and assign the
	 * quotient to this big integer. The remainder is returned.
	 *
	 * @param  divisor  Unsigned long integer divisor.
	 *
	 * @return  Unsigned long integer remainder.
	 *
	 * @exception  ArithmeticException
	 *     (unchecked exception) Thrown if <TT>divisor</TT> is zero.
	 */
	public long divRem
		(long divisor)
		{
		if (divisor == 0L)
			throw new ArithmeticException
				("BigInteger.divRem(): Divide by zero");
		allocateTmp();
		copyWords (divisor, tmp);
		long rem;
		int cmp = compare (tmp, value);
		if (cmp > 0)
			{
			rem = longValue();
			assignZero();
			}
		else if (cmp == 0)
			{
			rem = 0L;
			assign (1);
			}
		else
			{
			allocateQuo();
			doDivide (value, tmp, quo);
			rem = longValue();
			copyWords (quo, value);
			}
		return rem;
		}

	/**
	 * Divide this big integer by the given big integer and assign the quotient
	 * to this big integer. The remainder is assigned to <TT>remainder</TT>.
	 *
	 * @param  divisor    Big integer divisor.
	 * @param  remainder  Big integer remainder.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>divisor</TT> is null. Thrown if
	 *     <TT>remainder</TT> is null.
	 * @exception  ArithmeticException
	 *     (unchecked exception) Thrown if <TT>divisor</TT> is zero.
	 */
	public BigInteger divRem
		(BigInteger divisor,
		 BigInteger remainder)
		{
		if (eqZero (divisor.value))
			throw new ArithmeticException
				("BigInteger.divRem(): Divide by zero");
		int cmp = compare (divisor.value, this.value);
		if (cmp > 0)
			{
			remainder.assign (this);
			assignZero();
			}
		else if (cmp == 0)
			{
			remainder.assignZero();
			assign (1);
			}
		else
			{
			allocateTmp();
			allocateQuo();
			copyWords (divisor.value, this.tmp);
			doDivide (this.value, this.tmp, this.quo);
			copyWords (this.value, remainder.value);
			copyWords (this.quo, this.value);
			}
		return this;
		}

	private static void doDivide
		(int[] remainder, // Dividend (input), remainder (output)
		 int[] divisor,   // Divisor (input), garbage (output)
		 int[] quotient)  // Quotient (output)
		{
		clearWords (quotient, 0);
		int shift = leftmostOne (remainder) - leftmostOne (divisor);
		if (shift < 0) return;
		leftShift (divisor, shift);
		while (shift >= 0)
			{
			leftShift (quotient, 1);
			if (compare (remainder, divisor) >= 0)
				{
				addComplement (remainder, divisor, 1L);
				setBit (quotient, 0);
				}
			rightShift (divisor, 1);
			-- shift;
			}
		}

	/**
	 * Divide this big integer by the given unsigned integer and assign the
	 * remainder to this big integer. The quotient is discarded.
	 *
	 * @param  divisor  Integer divisor.
	 *
	 * @return  This big integer.
	 *
	 * @exception  ArithmeticException
	 *     (unchecked exception) Thrown if <TT>divisor</TT> is zero.
	 */
	public BigInteger rem
		(int divisor)
		{
		if (divisor == 0)
			throw new ArithmeticException
				("BigInteger.rem(): Divide by zero");
		copyWord (divisor, tmp);
		int cmp = compare (tmp, value);
		if (cmp > 0)
			return this;
		else if (cmp == 0)
			return assignZero();
		else
			{
			allocateTmp();
			doRemainder (value, tmp);
			return this;
			}
		}

	/**
	 * Divide this big integer by the given unsigned long integer and assign the
	 * remainder to this big integer. The quotient is discarded.
	 *
	 * @param  divisor  Long integer divisor.
	 *
	 * @return  This big integer.
	 *
	 * @exception  ArithmeticException
	 *     (unchecked exception) Thrown if <TT>divisor</TT> is zero.
	 */
	public BigInteger rem
		(long divisor)
		{
		if (divisor == 0L)
			throw new ArithmeticException
				("BigInteger.rem(): Divide by zero");
		copyWords (divisor, tmp);
		int cmp = compare (tmp, value);
		if (cmp > 0)
			return this;
		else if (cmp == 0)
			return assignZero();
		else
			{
			allocateTmp();
			doRemainder (value, tmp);
			return this;
			}
		}

	/**
	 * Divide this big integer by the given big integer and assign the remainder
	 * to this big integer. The quotient is discarded.
	 *
	 * @param  divisor  Big integer divisor.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>divisor</TT> is null.
	 * @exception  ArithmeticException
	 *     (unchecked exception) Thrown if <TT>divisor</TT> is zero.
	 */
	public BigInteger rem
		(BigInteger divisor)
		{
		if (eqZero (divisor.value))
			throw new ArithmeticException
				("BigInteger.div(): Divide by zero");
		int cmp = divisor.compareTo (this);
		if (cmp > 0)
			return this;
		else if (cmp == 0)
			return assignZero();
		else
			{
			allocateTmp();
			copyWords (divisor.value, this.tmp);
			doRemainder (this.value, this.tmp);
			return this;
			}
		}

	private static void doRemainder
		(int[] remainder, // Dividend (input), remainder (output)
		 int[] divisor)   // Divisor (input), garbage (output)
		{
		int shift = leftmostOne (remainder) - leftmostOne (divisor);
		if (shift < 0) return;
		leftShift (divisor, shift);
		while (shift >= 0)
			{
			if (compare (remainder, divisor) >= 0)
				addComplement (remainder, divisor, 1L);
			rightShift (divisor, 1);
			-- shift;
			}
		}

	/**
	 * Assign to this big integer the greatest common denominator (GCD) of this
	 * big integer and the given big integer. If this big integer is zero or
	 * <TT>bigint</TT> is zero, this big integer is set to zero.
	 *
	 * @param  bigint  Big integer.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>bigint</TT> is null.
	 */
	public BigInteger gcd
		(BigInteger bigint)
		{
		if (eqZero (this.value))
			return this;
		else if (eqZero (bigint.value))
			return assignZero();
		else
			{
			allocateTmp();
			allocateQuo();
			copyWords (bigint.value, this.tmp);
			do
				{
				copyWords (this.tmp, this.quo);
				doRemainder (this.value, this.tmp);
				copyWords (this.value, this.tmp);
				copyWords (this.quo, this.value);
				}
			while (! eqZero (this.tmp));
			return this;
			}
		}

	/**
	 * Assign to this big integer the smaller of this big integer and the given
	 * big integer.
	 *
	 * @param  bigint  Big integer to compare.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>bigint</TT> is null.
	 */
	public BigInteger min
		(BigInteger bigint)
		{
		int alen = this.value.length;
		int blen = bigint.value.length;
		int i = Math.max (alen, blen) - 1;
		long aw, bw;
		while (i >= 0)
			{
			aw = i < alen ? this.value[i] & MASK_32 : 0L;
			bw = i < blen ? bigint.value[i] & MASK_32 : 0L;
			if (aw < bw)
				i = -1;
			else if (aw > bw)
				while (i >= 0)
					{
					this.value[i] = bigint.value[i];
					-- i;
					}
			else
				-- i;
			}
		return this;
		}

	/**
	 * Assign to this big integer the larger of this big integer and the given
	 * big integer.
	 *
	 * @param  bigint  Big integer to compare.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>bigint</TT> is null.
	 */
	public BigInteger max
		(BigInteger bigint)
		{
		int alen = this.value.length;
		int blen = bigint.value.length;
		int i = Math.max (alen, blen) - 1;
		long aw, bw;
		while (i >= 0)
			{
			aw = i < alen ? this.value[i] & MASK_32 : 0L;
			bw = i < blen ? bigint.value[i] & MASK_32 : 0L;
			if (aw > bw)
				i = -1;
			else if (aw < bw)
				while (i >= 0)
					{
					this.value[i] = bigint.value[i];
					-- i;
					}
			else
				-- i;
			}
		return this;
		}

	/**
	 * Set this big integer to its complement.
	 *
	 * @return  This big integer.
	 */
	public BigInteger not()
		{
		for (int i = 0; i < value.length; ++ i)
			value[i] = ~ value[i];
		clearUnusedBits();
		return this;
		}

	/**
	 * Bitwise-exclusive-or this big integer with the given unsigned integer and
	 * assign the result to this big integer.
	 *
	 * @param  n  Integer to bitwise-exclusive-or.
	 *
	 * @return  This big integer.
	 */
	public BigInteger xor
		(int n)
		{
		value[0] ^= n;
		clearUnusedBits();
		return this;
		}

	/**
	 * Bitwise-exclusive-or this big integer with the given unsigned long
	 * integer and assign the result to this big integer.
	 *
	 * @param  n  Long integer to bitwise-exclusive-or.
	 *
	 * @return  This big integer.
	 */
	public BigInteger xor
		(long n)
		{
		value[0] ^= (int)n;
		value[1] ^= (int)(n >> 32);
		clearUnusedBits();
		return this;
		}

	/**
	 * Bitwise-exclusive-or this big integer with the given big integer and
	 * assign the result to this big integer.
	 *
	 * @param  bigint  Big integer to bitwise-exclusive-or.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>bigint</TT> is null.
	 */
	public BigInteger xor
		(BigInteger bigint)
		{
		int n = Math.min (bigint.value.length, this.value.length);
		for (int i = 0; i < n; ++ i)
			this.value[i] ^= bigint.value[i];
		clearUnusedBits();
		return this;
		}

	/**
	 * Bitwise-or this big integer with the given unsigned integer and assign
	 * the result to this big integer.
	 *
	 * @param  n  Integer to bitwise-or.
	 *
	 * @return  This big integer.
	 */
	public BigInteger or
		(int n)
		{
		value[0] |= n;
		clearUnusedBits();
		return this;
		}

	/**
	 * Bitwise-or this big integer with the given unsigned long integer and
	 * assign the result to this big integer.
	 *
	 * @param  n  Long integer to bitwise-or.
	 *
	 * @return  This big integer.
	 */
	public BigInteger or
		(long n)
		{
		value[0] |= (int)n;
		value[1] |= (int)(n >> 32);
		clearUnusedBits();
		return this;
		}

	/**
	 * Bitwise-or this big integer with the given big integer and assign the
	 * result to this big integer.
	 *
	 * @param  bigint  Big integer to bitwise-or.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>bigint</TT> is null.
	 */
	public BigInteger or
		(BigInteger bigint)
		{
		bitwiseOr (this.value, bigint.value);
		clearUnusedBits();
		return this;
		}

	/**
	 * Bitwise-and this big integer with the given unsigned integer and assign
	 * the result to this big integer.
	 *
	 * @param  n  Integer to bitwise-and.
	 *
	 * @return  This big integer.
	 */
	public BigInteger and
		(int n)
		{
		value[0] &= n;
		clearWords (value, 1);
		clearUnusedBits();
		return this;
		}

	/**
	 * Bitwise-and this big integer with the given unsigned long integer and
	 * assign the result to this big integer.
	 *
	 * @param  n  Long integer to bitwise-and.
	 *
	 * @return  This big integer.
	 */
	public BigInteger and
		(long n)
		{
		value[0] &= (int)n;
		value[1] &= (int)(n >> 32);
		clearWords (value, 2);
		clearUnusedBits();
		return this;
		}

	/**
	 * Bitwise-and this big integer with the given big integer and assign the
	 * result to this big integer.
	 *
	 * @param  bigint  Big integer to bitwise-and.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>bigint</TT> is null.
	 */
	public BigInteger and
		(BigInteger bigint)
		{
		int n = Math.min (bigint.value.length, this.value.length);
		for (int i = 0; i < n; ++ i)
			this.value[i] &= bigint.value[i];
		clearWords (this.value, n);
		clearUnusedBits();
		return this;
		}

	/**
	 * Bitwise-and this big integer with the complement of the given unsigned
	 * integer and assign the result to this big integer.
	 *
	 * @param  n  Integer to bitwise-and.
	 *
	 * @return  This big integer.
	 */
	public BigInteger andNot
		(int n)
		{
		value[0] &= ~n;
		clearUnusedBits();
		return this;
		}

	/**
	 * Bitwise-and this big integer with the complement of the given unsigned
	 * long integer and assign the result to this big integer.
	 *
	 * @param  n  Long integer to bitwise-and.
	 *
	 * @return  This big integer.
	 */
	public BigInteger andNot
		(long n)
		{
		value[0] &= ~((int)n);
		value[1] &= ~((int)(n >> 32));
		clearUnusedBits();
		return this;
		}

	/**
	 * Bitwise-and this big integer with the complement of the given big integer
	 * and assign the result to this big integer.
	 *
	 * @param  bigint  Big integer to bitwise-and.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>bigint</TT> is null.
	 */
	public BigInteger andNot
		(BigInteger bigint)
		{
		int n = Math.min (bigint.value.length, this.value.length);
		for (int i = 0; i < n; ++ i)
			this.value[i] &= ~bigint.value[i];
		clearUnusedBits();
		return this;
		}

	/**
	 * Extract the given bits from the given big integer and assign them to this
	 * big integer. The bit at position <TT>pos[0]</TT> of <TT>bigint</TT> is
	 * assigned to bit 0 of this big integer, the bit at position
	 * <TT>pos[1]</TT> of <TT>bigint</TT> is assigned to bit 1 of this big
	 * integer, and so on. Any remaining unassigned bits of this big integer are
	 * set to 0. The bit size of this big integer must be greater than or equal
	 * to the length of the <TT>pos</TT> array. Each bit position in the
	 * <TT>pos</TT> array must be in the range 0 .. {@link #bitSize()
	 * bitSize()}&minus;1.
	 *
	 * @param  bigint  Big integer from which to extract bits.
	 * @param  pos     Array of bit positions to extract.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>bigint</TT> is null. Thrown if
	 *     <TT>pos</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if {@link #bitSize() bitSize()} &lt;
	 *     <TT>pos.length</TT>. Thrown if any bit position in <TT>pos</TT> is
	 *     out of bounds.
	 */
	public BigInteger extractBits
		(BigInteger bigint,
		 int[] pos)
		{
		int n = pos.length;
		if (bitSize < n)
			throw new IndexOutOfBoundsException (String.format
				("BigInteger.extractBits(): bitSize = %d too small", bitSize));
		allocateTmp();
		clearWords (tmp, 0);
		for (int i = 0; i < n; ++ i)
			{
			int j = pos[i];
			if (0 > j || j >= bigint.bitSize)
				throw new IndexOutOfBoundsException (String.format
					("BigInteger.extractBits(): pos[%d] = %d out of bounds",
					 i, j));
			putBit (tmp, i, getBit (bigint.value, j));
			}
		copyWords (tmp, value);
		return this;
		}

	/**
	 * Extract the given bits from the given big integer and assign them to this
	 * big integer. The bit at position <TT>bits.bit(0)</TT> of <TT>bigint</TT>
	 * is assigned to bit 0 of this big integer, the bit at position
	 * <TT>bits.bit(1)</TT> of <TT>bigint</TT> is assigned to bit 1 of this big
	 * integer, and so on. Any remaining unassigned bits of this big integer are
	 * set to 0. The bit size of this big integer must be greater than or equal
	 * to <TT>bits.size()</TT>. Each bit position in <TT>bits</TT> must be in
	 * the range 0 .. {@link #bitSize() bitSize()}&minus;1.
	 *
	 * @param  bigint  Big integer from which to extract bits.
	 * @param  bits    Object specifying bit positions to extract.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>bigint</TT> is null. Thrown if
	 *     <TT>bits</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if {@link #bitSize() bitSize()} &lt;
	 *     <TT>bits.size()</TT>. Thrown if any bit position in <TT>bits</TT> is
	 *     out of bounds.
	 */
	public BigInteger extractBits
		(BigInteger bigint,
		 Bits bits)
		{
		int n = bits.size();
		if (bitSize < n)
			throw new IndexOutOfBoundsException (String.format
				("BigInteger.extractBits(): bitSize = %d too small", bitSize));
		allocateTmp();
		clearWords (tmp, 0);
		for (int i = 0; i < n; ++ i)
			{
			int j = bits.bit(i);
			if (0 > j || j >= bigint.bitSize)
				throw new IndexOutOfBoundsException (String.format
					("BigInteger.extractBits(): bits.bit(%d) = %d out of bounds",
					 i, j));
			putBit (tmp, i, getBit (bigint.value, j));
			}
		copyWords (tmp, value);
		return this;
		}

	/**
	 * Returns this big integer's population count. The population count is the
	 * number of 1 bits in this big integer.
	 *
	 * @return  Population count.
	 */
	public int popCount()
		{
		int rv = 0;
		for (int i = 0; i < value.length; ++ i)
			rv += Integer.bitCount (value[i]);
		return rv;
		}

	/**
	 * Returns the position of the leftmost 1 bit in this big integer.
	 *
	 * @return  Position of the leftmost 1 bit in the range 0 ..
	 *          {@link #bitSize() bitSize()}&minus;1, or &minus;1 if this big
	 *          integer is zero.
	 */
	public int leftmostOne()
		{
		return leftmostOne (value);
		}

	/**
	 * Returns the position of the rightmost 1 bit in this big integer.
	 *
	 * @return  Position of the rightmost 1 bit in the range 0 ..
	 *          {@link #bitSize() bitSize()}&minus;1, or &minus;1 if this big
	 *          integer is zero.
	 */
	public int rightmostOne()
		{
		return rightmostOne (value);
		}

	/**
	 * Determine if this big integer's value is equal to 0.
	 *
	 * @return  True if value = 0, false if value &ne; 0.
	 */
	public boolean eqZero()
		{
		return eqZero (value);
		}

	/**
	 * Compare this big integer to the given unsigned integer.
	 *
	 * @param  n  Unsigned integer to compare.
	 *
	 * @return  A value less than, equal to, or greater than 0 if this big
	 *          integer is less than, equal to, or greater than <TT>n</TT>,
	 *          respectively.
	 */
	public int compareTo
		(int n)
		{
		allocateTmp();
		copyWord (n, tmp);
		return compare (value, tmp);
		}

	/**
	 * Compare this big integer to the given unsigned long integer.
	 *
	 * @param  n  Unsigned long integer to compare.
	 *
	 * @return  A value less than, equal to, or greater than 0 if this big
	 *          integer is less than, equal to, or greater than <TT>n</TT>,
	 *          respectively.
	 */
	public int compareTo
		(long n)
		{
		allocateTmp();
		copyWords (n, tmp);
		return compare (value, tmp);
		}

	/**
	 * Compare this big integer to the given big integer.
	 *
	 * @param  bigint  Big integer to compare.
	 *
	 * @return  A value less than, equal to, or greater than 0 if this big
	 *          integer is less than, equal to, or greater than <TT>bigint</TT>,
	 *          respectively.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>bigint</TT> is null.
	 */
	public int compareTo
		(BigInteger bigint)
		{
		return compare (this.value, bigint.value);
		}

	/**
	 * Pack the value obtained from the given byte array into this big integer
	 * in little-endian order. Bits 0 through 7 of this big integer are set to
	 * <TT>v[0]</TT>, bits 8 through 15 of this big integer are set to
	 * <TT>v[1]</TT>, and so on. If there are not enough bytes in <TT>v</TT> to
	 * fill this big integer, the unfilled most significant bits of this big
	 * integer are set to 0. If there are more bytes in <TT>v</TT> than needed
	 * to fill this big integer, the extra bytes in <TT>v</TT> are ignored.
	 *
	 * @param  v  Value array.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 */
	public BigInteger pack
		(byte[] v)
		{
		int src = 0;
		int dst = 0;
		while (src < v.length && dst < value.length)
			{
			value[dst] = v[src++] & 0xff;
			if (src < v.length) value[dst] |= (v[src++] & 0xff) << 8;
			if (src < v.length) value[dst] |= (v[src++] & 0xff) << 16;
			if (src < v.length) value[dst] |= (v[src++] & 0xff) << 24;
			++ dst;
			}
		clearWords (value, dst);
		clearUnusedBits();
		return this;
		}

	/**
	 * Pack the value obtained from the given short array into this big integer
	 * in little-endian order. Bits 0 through 15 of this big integer are set to
	 * <TT>v[0]</TT>, bits 15 through 31 of this big integer are set to
	 * <TT>v[1]</TT>, and so on. If there are not enough shorts in <TT>v</TT> to
	 * fill this big integer, the unfilled most significant bits of this big
	 * integer are set to 0. If there are more shorts in <TT>v</TT> than needed
	 * to fill this big integer, the extra shorts in <TT>v</TT> are ignored.
	 *
	 * @param  v  Value array.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 */
	public BigInteger pack
		(short[] v)
		{
		int src = 0;
		int dst = 0;
		while (src < v.length && dst < value.length)
			{
			value[dst] = v[src++] & 0xffff;
			if (src < v.length) value[dst] |= (v[src++] & 0xffff) << 16;
			++ dst;
			}
		clearWords (value, dst);
		clearUnusedBits();
		return this;
		}

	/**
	 * Pack the value obtained from the given integer array into this big
	 * integer in little-endian order. Bits 0 through 31 of this big integer are
	 * set to <TT>v[0]</TT>, bits 32 through 63 of this big integer are set to
	 * <TT>v[1]</TT>, and so on. If there are not enough integers in <TT>v</TT>
	 * to fill this big integer, the unfilled most significant bits of this big
	 * integer are set to 0. If there are more integers in <TT>v</TT> than
	 * needed to fill this big integer, the extra integers in <TT>v</TT> are
	 * ignored.
	 *
	 * @param  v  Value array.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 */
	public BigInteger pack
		(int[] v)
		{
		copyWords (v, this.value);
		clearUnusedBits();
		return this;
		}

	/**
	 * Pack the value obtained from the given long array into this big integer
	 * in little-endian order. Bits 0 through 63 of this big integer are set to
	 * <TT>v[0]</TT>, bits 64 through 127 of this big integer are set to
	 * <TT>v[1]</TT>, and so on. If there are not enough longs in <TT>v</TT> to
	 * fill this big integer, the unfilled most significant bits of this big
	 * integer are set to 0. If there are more longs in <TT>v</TT> than needed
	 * to fill this big integer, the extra longs in <TT>v</TT> are ignored.
	 *
	 * @param  v  Value array.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 */
	public BigInteger pack
		(long[] v)
		{
		int src = 0;
		int dst = 0;
		while (src < v.length && dst < value.length)
			{
			value[dst++] = (int)v[src];
			if (dst < value.length) value[dst++] = (int)(v[src] >>> 32);
			++ src;
			}
		clearWords (value, dst);
		clearUnusedBits();
		return this;
		}

	/**
	 * Pack the value obtained from the given byte array into this big integer
	 * in big-endian order. Let <TT>L</TT> = <TT>v.length</TT>. Bits 0 through 7
	 * of this big integer are set to <TT>v[L-1]</TT>, bits 8 through 15 of this
	 * big integer are set to <TT>v[L-2]</TT>, and so on. If there are not
	 * enough bytes in <TT>v</TT> to fill this big integer, the unfilled most
	 * significant bits of this big integer are set to 0. If there are more
	 * bytes in <TT>v</TT> than needed to fill this big integer, the extra bytes
	 * in <TT>v</TT> are ignored.
	 *
	 * @param  v  Value array.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 */
	public BigInteger packBigEndian
		(byte[] v)
		{
		int src = v.length - 1;
		int dst = 0;
		while (src >= 0 && dst < value.length)
			{
			value[dst] = v[src--] & 0xff;
			if (src >= 0) value[dst] |= (v[src--] & 0xff) << 8;
			if (src >= 0) value[dst] |= (v[src--] & 0xff) << 16;
			if (src >= 0) value[dst] |= (v[src--] & 0xff) << 24;
			++ dst;
			}
		clearWords (value, dst);
		clearUnusedBits();
		return this;
		}

	/**
	 * Pack the value obtained from the given short array into this big integer
	 * in big-endian order. Let <TT>L</TT> = <TT>v.length</TT>. Bits 0 through
	 * 15 of this big integer are set to <TT>v[L-1]</TT>, bits 15 through 31 of
	 * this big integer are set to <TT>v[L-2]</TT>, and so on. If there are not
	 * enough shorts in <TT>v</TT> to fill this big integer, the unfilled most
	 * significant bits of this big integer are set to 0. If there are more
	 * shorts in <TT>v</TT> than needed to fill this big integer, the extra
	 * shorts in <TT>v</TT> are ignored.
	 *
	 * @param  v  Value array.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 */
	public BigInteger packBigEndian
		(short[] v)
		{
		int src = v.length - 1;
		int dst = 0;
		while (src >= 0 && dst < value.length)
			{
			value[dst] = v[src--] & 0xffff;
			if (src >= 0) value[dst] |= (v[src--] & 0xffff) << 16;
			++ dst;
			}
		clearWords (value, dst);
		clearUnusedBits();
		return this;
		}

	/**
	 * Pack the value obtained from the given integer array into this big
	 * integer in big-endian order. Let <TT>L</TT> = <TT>v.length</TT>. Bits 0
	 * through 31 of this big integer are set to <TT>v[L-1]</TT>, bits 32
	 * through 63 of this big integer are set to <TT>v[L-2]</TT>, and so on. If
	 * there are not enough integers in <TT>v</TT> to fill this big integer, the
	 * unfilled most significant bits of this big integer are set to 0. If there
	 * are more integers in <TT>v</TT> than needed to fill this big integer, the
	 * extra integers in <TT>v</TT> are ignored.
	 *
	 * @param  v  Value array.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 */
	public BigInteger packBigEndian
		(int[] v)
		{
		int src = v.length - 1;
		int dst = 0;
		while (src >= 0 && dst < value.length)
			value[dst++] = v[src--];
		clearWords (value, dst);
		clearUnusedBits();
		return this;
		}

	/**
	 * Pack the value obtained from the given long array into this big integer
	 * in big-endian order. Let <TT>L</TT> = <TT>v.length</TT>. Bits 0 through
	 * 63 of this big integer are set to <TT>v[L-1]</TT>, bits 64 through 127 of
	 * this big integer are set to <TT>v[L-2]</TT>, and so on. If there are not
	 * enough longs in <TT>v</TT> to fill this big integer, the unfilled most
	 * significant bits of this big integer are set to 0. If there are more
	 * longs in <TT>v</TT> than needed to fill this big integer, the extra longs
	 * in <TT>v</TT> are ignored.
	 *
	 * @param  v  Value array.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 */
	public BigInteger packBigEndian
		(long[] v)
		{
		int src = v.length - 1;
		int dst = 0;
		while (src >= 0 && dst < value.length)
			{
			value[dst++] = (int)v[src];
			if (dst < value.length) value[dst++] = (int)(v[src] >>> 32);
			-- src;
			}
		clearWords (value, dst);
		clearUnusedBits();
		return this;
		}

	/**
	 * Unpack this big integer's value into the given byte array in
	 * little-endian order. <TT>v[0]</TT> is set to bits 0&ndash;7 of this big
	 * integer, <TT>v[1]</TT> is set to bits 8&ndash;15 of this big integer, and
	 * so on. If there are not enough bytes in this big integer's value to fill
	 * <TT>v</TT>, the unfilled portion of <TT>v</TT> is set to 0. If there are
	 * more bytes in this big integer's value than needed to fill <TT>v</TT>,
	 * the extra bytes in this big integer's value are ignored.
	 *
	 * @param  v  Value array.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 */
	public BigInteger unpack
		(byte[] v)
		{
		int src = 0;
		int dst = 0;
		while (src < value.length && dst < v.length)
			{
			v[dst++] = (byte)value[src];
			if (dst < v.length) v[dst++] = (byte)(value[src] >>> 8);
			if (dst < v.length) v[dst++] = (byte)(value[src] >>> 16);
			if (dst < v.length) v[dst++] = (byte)(value[src] >>> 24);
			++ src;
			}
		while (dst < v.length)
			v[dst++] = (byte)0;
		return this;
		}

	/**
	 * Unpack this big integer's value into the given short array in
	 * little-endian order. <TT>v[0]</TT> is set to bits 0&ndash;15 of this big
	 * integer, <TT>v[1]</TT> is set to bits 16&ndash;31 of this big integer,
	 * and so on. If there are not enough shorts in this big integer's value to
	 * fill <TT>v</TT>, the unfilled portion of <TT>v</TT> is set to 0. If there
	 * are more shorts in this big integer's value than needed to fill
	 * <TT>v</TT>, the extra shorts in this big integer's value are ignored.
	 *
	 * @param  v  Value array.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 */
	public BigInteger unpack
		(short[] v)
		{
		int src = 0;
		int dst = 0;
		while (src < value.length && dst < v.length)
			{
			v[dst++] = (short)value[src];
			if (dst < v.length) v[dst++] = (short)(value[src] >>> 16);
			++ src;
			}
		while (dst < v.length)
			v[dst++] = (short)0;
		return this;
		}

	/**
	 * Unpack this big integer's value into the given integer array in
	 * little-endian order. <TT>v[0]</TT> is set to bits 0&ndash;31 of this big
	 * integer, <TT>v[1]</TT> is set to bits 32&ndash;63 of this big integer,
	 * and so on. If there are not enough integers in this big integer's value
	 * to fill <TT>v</TT>, the unfilled portion of <TT>v</TT> is set to 0. If
	 * there are more integers in this big integer's value than needed to fill
	 * <TT>v</TT>, the extra integers in this big integer's value are ignored.
	 *
	 * @param  v  Value array.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 */
	public BigInteger unpack
		(int[] v)
		{
		copyWords (this.value, v);
		return this;
		}

	/**
	 * Unpack this big integer's value into the given long array in
	 * little-endian order. <TT>v[0]</TT> is set to bits 0&ndash;63 of this big
	 * integer, <TT>v[1]</TT> is set to bits 64&ndash;127 of this big integer,
	 * and so on. If there are not enough longs in this big integer's value to
	 * fill <TT>v</TT>, the unfilled portion of <TT>v</TT> is set to 0. If there
	 * are more longs in this big integer's value than needed to fill
	 * <TT>v</TT>, the extra longs in this big integer's value are ignored.
	 *
	 * @param  v  Value array.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 */
	public BigInteger unpack
		(long[] v)
		{
		int src = 0;
		int dst = 0;
		while (src < value.length && dst < v.length)
			{
			v[dst] = value[src++] & MASK_32;
			if (src < value.length) v[dst] |= (value[src++] & MASK_32) << 32;
			++ dst;
			}
		while (dst < v.length)
			v[dst++] = 0L;
		return this;
		}

	/**
	 * Unpack this big integer's value into the given byte array in big-endian
	 * order. Let <TT>L</TT> = <TT>v.length</TT>. <TT>v[L-1]</TT> is set to bits
	 * 0&ndash;7 of this big integer, <TT>v[L-2]</TT> is set to bits 8&ndash;15
	 * of this big integer, and so on. If there are not enough bytes in this big
	 * integer's value to fill <TT>v</TT>, the unfilled portion of <TT>v</TT> is
	 * set to 0. If there are more bytes in this big integer's value than needed
	 * to fill <TT>v</TT>, the extra bytes in this big integer's value are
	 * ignored.
	 *
	 * @param  v  Value array.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 */
	public BigInteger unpackBigEndian
		(byte[] v)
		{
		int src = 0;
		int dst = v.length - 1;
		while (src < value.length && dst >= 0)
			{
			v[dst--] = (byte)value[src];
			if (dst >= 0) v[dst--] = (byte)(value[src] >>> 8);
			if (dst >= 0) v[dst--] = (byte)(value[src] >>> 16);
			if (dst >= 0) v[dst--] = (byte)(value[src] >>> 24);
			++ src;
			}
		while (dst >= 0)
			v[dst--] = (byte)0;
		return this;
		}

	/**
	 * Unpack this big integer's value into the given short array in big-endian
	 * order. Let <TT>L</TT> = <TT>v.length</TT>. <TT>v[L-1]</TT> is set to bits
	 * 0&ndash;15 of this big integer, <TT>v[L-2]</TT> is set to bits
	 * 16&ndash;31 of this big integer, and so on. If there are not enough
	 * shorts in this big integer's value to fill <TT>v</TT>, the unfilled
	 * portion of <TT>v</TT> is set to 0. If there are more shorts in this big
	 * integer's value than needed to fill <TT>v</TT>, the extra shorts in this
	 * big integer's value are ignored.
	 *
	 * @param  v  Value array.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 */
	public BigInteger unpackBigEndian
		(short[] v)
		{
		int src = 0;
		int dst = v.length - 1;
		while (src < value.length && dst >= 0)
			{
			v[dst--] = (short)value[src];
			if (dst >= 0) v[dst--] = (short)(value[src] >>> 16);
			++ src;
			}
		while (dst >= 0)
			v[dst--] = (short)0;
		return this;
		}

	/**
	 * Unpack this big integer's value into the given integer array in
	 * big-endian order. Let <TT>L</TT> = <TT>v.length</TT>. <TT>v[L-1]</TT> is
	 * set to bits 0&ndash;31 of this big integer, <TT>v[L-2]</TT> is set to
	 * bits 32&ndash;63 of this big integer, and so on. If there are not enough
	 * integers in this big integer's value to fill <TT>v</TT>, the unfilled
	 * portion of <TT>v</TT> is set to 0. If there are more integers in this big
	 * integer's value than needed to fill <TT>v</TT>, the extra integers in
	 * this big integer's value are ignored.
	 *
	 * @param  v  Value array.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 */
	public BigInteger unpackBigEndian
		(int[] v)
		{
		int src = 0;
		int dst = v.length - 1;
		while (src < value.length && dst >= 0)
			v[dst--] = value[src++];
		while (dst >= 0)
			v[dst--] = 0;
		return this;
		}

	/**
	 * Unpack this big integer's value into the given long array in big-endian
	 * order. Let <TT>L</TT> = <TT>v.length</TT>. <TT>v[L-1]</TT> is set to bits
	 * 0&ndash;63 of this big integer, <TT>v[L-2]</TT> is set to bits
	 * 64&ndash;127 of this big integer, and so on. If there are not enough
	 * longs in this big integer's value to fill <TT>v</TT>, the unfilled
	 * portion of <TT>v</TT> is set to 0. If there are more longs in this big
	 * integer's value than needed to fill <TT>v</TT>, the extra longs in this
	 * big integer's value are ignored.
	 *
	 * @param  v  Value array.
	 *
	 * @return  This big integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 */
	public BigInteger unpackBigEndian
		(long[] v)
		{
		int src = 0;
		int dst = v.length - 1;
		while (src < value.length && dst >= 0)
			{
			v[dst] = value[src++] & MASK_32;
			if (src < value.length) v[dst] |= (value[src++] & MASK_32) << 32;
			-- dst;
			}
		while (dst >= 0)
			v[dst--] = 0L;
		return this;
		}

	/**
	 * Determine if this big integer is equal to the given object.
	 *
	 * @param  obj  Object to test.
	 *
	 * @return  True if this big integer is equal to <TT>obj</TT>, false
	 *          otherwise.
	 */
	public boolean equals
		(Object obj)
		{
		return (obj instanceof BigInteger) &&
			(compareTo ((BigInteger)obj) == 0);
		}

	/**
	 * Returns a hash code for this big integer.
	 *
	 * @return  Hash code.
	 */
	public int hashCode()
		{
		long hash = 0L;
		for (int i = 0; i < value.length; ++ i)
			hash = hash*31L + (value[i] & MASK_32);
		return (int)hash;
		}

	/**
	 * Returns a string version of this big integer. The string is this big
	 * integer's value in decimal.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		StringBuilder b = new StringBuilder();
		BigInteger bigint = new BigInteger (this);
		while (! bigint.eqZero())
			b.insert (0, (char)(bigint.divRem (10) + '0'));
		return b.length() == 0 ? "0" : b.toString();
		}

	/**
	 * Returns a string version of this big integer. The string is this big
	 * integer's value in hexadecimal.
	 *
	 * @return  String version.
	 */
	public String toStringHex()
		{
		StringBuilder b = new StringBuilder();
		for (int i = 0; i < bitSize; i += 4)
			b.insert (0, hexToChar ((value[i>>>5] >>> (i & 31)) & 15));
		return b.toString();
		}

	private static char hexToChar
		(int d)
		{
		return d <= 9 ? (char)(d + '0') : (char)(d + 'a' - 10);
		}

	/**
	 * Write this big integer to the given out stream.
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
		out.writeUnsignedInt (bitSize);

		// Count number of least significant nonzero bytes.
		int lsnzb = value.length*4;
		int i = value.length - 1;
		int j = 24;
		while (lsnzb > 0 && ((value[i] >>> j) & 0xff) == 0)
			{
			-- lsnzb;
			j -= 8;
			if (j < 0)
				{
				-- i;
				j = 24;
				}
			}

		// Write out the least significant nonzero bytes.
		out.writeUnsignedInt (lsnzb);
		i = 0;
		j = 0;
		while (lsnzb > 0)
			{
			out.writeByte ((byte)(value[i] >>> j));
			-- lsnzb;
			j += 8;
			if (j == 32)
				{
				++ i;
				j = 0;
				}
			}
		}

	/**
	 * Read this big integer from the given in stream.
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
		bitSize = in.readUnsignedInt();
		int ws = wordSize (bitSize);
		if (value != null && value.length == ws)
			clearWords (value, 0);
		else
			value = new int [ws];
		tmp = null;
		quo = null;

		// Read in the least significant nonzero bytes.
		int lsnzb = in.readUnsignedInt();
		int i = 0;
		int j = 0;
		while (lsnzb > 0)
			{
			value[i] |= (in.readByte() & 0xff) << j;
			-- lsnzb;
			j += 8;
			if (j == 32)
				{
				++ i;
				j = 0;
				}
			}
		}

// Hidden operations.

	/**
	 * Copy one word from the source to the destination. Any remaining words in
	 * the destination are filled with 0s.
	 */
	private static void copyWord
		(int src,
		 int[] dst)
		{
		dst[0] = src;
		clearWords (dst, 1);
		}

	/**
	 * Copy two words from the source to the destination. Any remaining words in
	 * the destination are filled with 0s.
	 */
	private static void copyWords
		(long src,
		 int[] dst)
		{
		dst[0] = (int)src;
		dst[1] = (int)(src >>> 32);
		clearWords (dst, 2);
		}

	/**
	 * Copy words from the source to the destination. Words are copied until the
	 * end of the source or the end of the destination, whichever comes first.
	 * Any remaining words in the destination are filled with 0s.
	 */
	private static void copyWords
		(int[] src,
		 int[] dst)
		{
		int n = Math.min (src.length, dst.length);
		System.arraycopy (src, 0, dst, 0, n);
		clearWords (dst, n);
		}

	/**
	 * Clear words in the destination. Words are cleared starting at index off
	 * through the end of the destination.
	 */
	private static void clearWords
		(int[] dst,
		 int off)
		{
		for (int i = off; i < dst.length; ++ i)
			dst[i] = 0;
		}

	/**
	 * Clear the unused high-order bits in this big integer.
	 */
	private void clearUnusedBits()
		{
		int i = bitSize >>> 5;
		if (i < value.length)
			{
			value[i++] &= (1 << (bitSize & 31)) - 1;
			while (i < value.length)
				value[i++] = 0;
			}
		}

	/**
	 * Get the given bit from the given array.
	 *
	 * @param  a  Array.
	 * @param  i  Bit position.
	 *
	 * @return  Bit value at position <TT>i</TT>, either 0 or 1.
	 */
	private static int getBit
		(int[] a,
		 int i)
		{
		return (a[i >>> 5] >>> (i & 31)) & 1;
		}

	/**
	 * Clear the given bit in the given array.
	 *
	 * @param  a  Array.
	 * @param  i  Bit position.
	 */
	private static void clearBit
		(int[] a,
		 int i)
		{
		a[i >>> 5] &= ~(1 << (i & 31));
		}

	/**
	 * Set the given bit in the given array.
	 *
	 * @param  a  Array.
	 * @param  i  Bit position.
	 */
	private static void setBit
		(int[] a,
		 int i)
		{
		a[i >>> 5] |= 1 << (i & 31);
		}

	/**
	 * Flip the given bit in the given array.
	 *
	 * @param  a  Array.
	 * @param  i  Bit position.
	 */
	private static void flipBit
		(int[] a,
		 int i)
		{
		a[i >>> 5] ^= 1 << (i & 31);
		}

	/**
	 * Put the given value into the given bit in the given array.
	 *
	 * @param  a  Array.
	 * @param  i  Bit position.
	 * @param  v  Bit value, either 0 or 1.
	 */
	private static void putBit
		(int[] a,
		 int i,
		 int v)
		{
		int off = i >>> 5;
		int shift = i & 31;
		a[off] &= ~ (1 << shift);
		a[off] |= v << shift;
		}

	/**
	 * Left-shift the given array by the given number of bits.
	 */
	private static void leftShift
		(int[] a,
		 int n)
		{
		int off = n >>> 5; // off = number of words to left-shift
		n &= 31;           // n = number of bits within word to left-shift
		int m = 32 - n;    // m = number of bits within word to right-shift

		int d = a.length - 1;
		int s = d - off;
		if (n == 0)
			{
			// Shift entire words only.
			while (s >= 0)
				{
				a[d] = a[s];
				-- d;
				-- s;
				}
			}
		else
			{
			// Shift words and bits within words.
			while (s > 0)
				{
				a[d] = (a[s] << n) | (a[s-1] >>> m);
				-- d;
				-- s;
				}
			a[d] = (a[s] << n);
			-- d;
			}

		// Clear remaining least significant words.
		while (d >= 0)
			{
			a[d] = 0;
			-- d;
			}
		}

	/**
	 * Right-shift the given array by the given number of bits.
	 */
	private static void rightShift
		(int[] a,
		 int n)
		{
		int off = n >>> 5; // off = number of words to right-shift
		n &= 31;           // n = number of bits within word to right-shift
		int m = 32 - n;    // m = number of bits within word to left-shift
		int alenm1 = a.length - 1;

		int d = 0;
		int s = off;
		if (n == 0)
			{
			// Shift entire words only.
			while (s <= alenm1)
				{
				a[d] = a[s];
				++ d;
				++ s;
				}
			}
		else
			{
			// Shift words and bits within words.
			while (s < alenm1)
				{
				a[d] = (a[s] >>> n) | (a[s+1] << m);
				++ d;
				++ s;
				}
			a[d] = (a[s] >>> n);
			++ d;
			}

		// Clear remaining most significant words.
		while (d <= alenm1)
			{
			a[d] = 0;
			++ d;
			}
		}

	/**
	 * Add array b to array a.
	 */
	private static void add
		(int[] a,
		 int[] b,
		 long sum)
		{
		int len = Math.min (a.length, b.length);
		for (int i = 0; i < len; ++ i)
			{
			sum += (a[i] & MASK_32) + (b[i] & MASK_32);
			a[i] = (int)sum;
			sum >>>= 32;
			}
		for (int i = len; i < a.length; ++ i)
			{
			sum += (a[i] & MASK_32);
			a[i] = (int)sum;
			sum >>>= 32;
			}
		}

	/**
	 * Add the complement of array b to array a.
	 */
	private static void addComplement
		(int[] a,
		 int[] b,
		 long sum)
		{
		int len = Math.min (a.length, b.length);
		for (int i = 0; i < len; ++ i)
			{
			sum += (a[i] & MASK_32) + ((~ b[i]) & MASK_32);
			a[i] = (int)sum;
			sum >>>= 32;
			}
		for (int i = len; i < a.length; ++ i)
			{
			sum += (a[i] & MASK_32);
			a[i] = (int)sum;
			sum >>>= 32;
			}
		}

	/**
	 * Bitwise-or array b into array a.
	 */
	private static void bitwiseOr
		(int[] a,
		 int[] b)
		{
		int n = Math.min (a.length, b.length);
		for (int i = 0; i < n; ++ i)
			a[i] |= b[i];
		}

	/**
	 * Returns true if array a contains all zeroes.
	 */
	private static boolean eqZero
		(int[] a)
		{
		for (int i = 0; i < a.length; ++ i)
			if (a[i] != 0)
				return false;
		return true;
		}

	/**
	 * Compare array a with array b. Returns 1 if a &gt; b, -1 if a &lt; b, 0 if
	 * a == b.
	 */
	private static int compare
		(int[] a,
		 int[] b)
		{
		int alen = a.length;
		int blen = b.length;
		long aw, bw;
		for (int i = Math.max (alen, blen) - 1; i >= 0; -- i)
			{
			aw = i < alen ? a[i] & MASK_32 : 0L;
			bw = i < blen ? b[i] & MASK_32 : 0L;
			if (aw < bw) return -1;
			else if (aw > bw) return 1;
			}
		return 0;
		}

	/**
	 * Returns the position of the leftmost 1 bit in array a, or -1 if array a
	 * is zero.
	 */
	private static int leftmostOne
		(int[] a)
		{
		int i = a.length - 1;
		while (i >= 0 && a[i] == 0) -- i;
		if (i < 0) return -1;
		int j = 31;
		while ((a[i] & (1 << j)) == 0) -- j;
		return 32*i + j;
		}

	/**
	 * Returns the position of the rightmost 1 bit in array a, or -1 if array a
	 * is zero.
	 */
	private static int rightmostOne
		(int[] a)
		{
		int i = 0;
		while (i < a.length && a[i] == 0) ++ i;
		if (i == a.length) return -1;
		int j = 0;
		while ((a[i] & (1 << j)) == 0) ++ j;
		return 32*i + j;
		}

	/**
	 * Allocate temporary storage if necessary.
	 */
	private void allocateTmp()
		{
		if (tmp == null)
			tmp = new int [value.length];
		}

	/**
	 * Allocate quotient storage if necessary.
	 */
	private void allocateQuo()
		{
		if (quo == null)
			quo = new int [value.length];
		}

// Unit test main program.

//	/**
//	 * Unit test main program.
//	 */
//	public static void main
//		(String[] args)
//		{
//		if (args.length != 3) usage();
//		int n = Integer.parseInt (args[0]);
//		BigInteger a = new BigInteger (n) .assign (args[1]);
//		BigInteger b = new BigInteger (n) .assign (args[2]);
//		BigInteger c = new BigInteger (n);
//		BigInteger d = new BigInteger (n);
//
//		System.out.printf ("a      = %s%n", a);
//		System.out.printf ("int    = %d%n", a.intValue());
//		System.out.printf ("long   = %d%n", a.longValue());
//		System.out.printf ("float  = %.6g%n", a.floatValue());
//		System.out.printf ("double = %.15g%n", a.doubleValue());
//		System.out.println();
//
//		System.out.printf ("b      = %s%n", b);
//		System.out.printf ("int    = %d%n", b.intValue());
//		System.out.printf ("long   = %d%n", b.longValue());
//		System.out.printf ("float  = %.6g%n", b.floatValue());
//		System.out.printf ("double = %.15g%n", b.doubleValue());
//		System.out.println();
//
//		System.out.printf ("a   = %s = %s%n", a.toStringHex(), a);
//		System.out.printf ("b   = %s = %s%n", b.toStringHex(), b);
//		c.assign (a) .neg();
//		System.out.printf ("-a  = %s = %s%n", c.toStringHex(), c);
//		c.assign (b) .neg();
//		System.out.printf ("-b  = %s = %s%n", c.toStringHex(), c);
//		c.assign (a) .add (b);
//		System.out.printf ("a+b = %s = %s%n", c.toStringHex(), c);
//		c.assign (b) .add (a);
//		System.out.printf ("b+a = %s = %s%n", c.toStringHex(), c);
//		c.assign (a) .sub (b);
//		System.out.printf ("a-b = %s = %s%n", c.toStringHex(), c);
//		c.assign (b) .sub (a);
//		System.out.printf ("b-a = %s = %s%n", c.toStringHex(), c);
//		c.assign (a) .mul (b);
//		System.out.printf ("a*b = %s = %s%n", c.toStringHex(), c);
//		c.assign (b) .mul (a);
//		System.out.printf ("b*a = %s = %s%n", c.toStringHex(), c);
//		try
//			{
//			c.assign (a) .div (b);
//			System.out.printf ("a/b = %s = %s%n", c.toStringHex(), c);
//			}
//		catch (ArithmeticException exc)
//			{
//			System.out.printf ("a/b = %s%n", exc);
//			}
//		try
//			{
//			c.assign (b) .div (a);
//			System.out.printf ("b/a = %s = %s%n", c.toStringHex(), c);
//			}
//		catch (ArithmeticException exc)
//			{
//			System.out.printf ("b/a = %s%n", exc);
//			}
//		try
//			{
//			c.assign (a) .rem (b);
//			System.out.printf ("a%%b = %s = %s%n", c.toStringHex(), c);
//			}
//		catch (ArithmeticException exc)
//			{
//			System.out.printf ("a%%b = %s%n", exc);
//			}
//		try
//			{
//			c.assign (b) .rem (a);
//			System.out.printf ("b%%a = %s = %s%n", c.toStringHex(), c);
//			}
//		catch (ArithmeticException exc)
//			{
//			System.out.printf ("b%%a = %s%n", exc);
//			}
//		System.out.println();
//
//		System.out.printf ("a         = %s = %s%n", a.toStringHex(), a);
//		c.assign (a) .leftShift (36);
//		System.out.printf ("a << 36   = %s = %s%n", c.toStringHex(), c);
//		c.assign (a) .rightShift (36);
//		System.out.printf ("a >> 36   = %s = %s%n", c.toStringHex(), c);
//		c.assign (a) .leftRotate (36);
//		System.out.printf ("a rotl 36 = %s = %s%n", c.toStringHex(), c);
//		c.assign (a) .rightRotate (36);
//		System.out.printf ("a rotr 36 = %s = %s%n", c.toStringHex(), c);
//		System.out.printf ("a.leftmostOne()  = %d%n", a.leftmostOne());
//		System.out.printf ("a.rightmostOne() = %d%n", a.rightmostOne());
//		System.out.println();
//
//		System.out.printf ("b         = %s = %s%n", b.toStringHex(), b);
//		c.assign (b) .leftShift (36);
//		System.out.printf ("b << 36   = %s = %s%n", c.toStringHex(), c);
//		c.assign (b) .rightShift (36);
//		System.out.printf ("b >> 36   = %s = %s%n", c.toStringHex(), c);
//		c.assign (b) .leftRotate (36);
//		System.out.printf ("b rotl 36 = %s = %s%n", c.toStringHex(), c);
//		c.assign (b) .rightRotate (36);
//		System.out.printf ("b rotr 36 = %s = %s%n", c.toStringHex(), c);
//		System.out.printf ("b.leftmostOne()  = %d%n", b.leftmostOne());
//		System.out.printf ("b.rightmostOne() = %d%n", b.rightmostOne());
//		System.out.println();
//
//		System.out.printf ("a        = %s = %s%n", a.toStringHex(), a);
//		System.out.printf ("b        = %s = %s%n", b.toStringHex(), b);
//		c.assign (a) .gcd (b);
//		System.out.printf ("a.gcd(b) = %s = %s%n", c.toStringHex(), c);
//		c.assign (b) .gcd (a);
//		System.out.printf ("b.gcd(a) = %s = %s%n", c.toStringHex(), c);
//		c.assign (a) .min (b);
//		System.out.printf ("a.min(b) = %s = %s%n", c.toStringHex(), c);
//		c.assign (b) .min (a);
//		System.out.printf ("b.min(a) = %s = %s%n", c.toStringHex(), c);
//		c.assign (a) .max (b);
//		System.out.printf ("a.max(b) = %s = %s%n", c.toStringHex(), c);
//		c.assign (b) .max (a);
//		System.out.printf ("b.max(a) = %s = %s%n", c.toStringHex(), c);
//		System.out.printf ("a.compareTo(b) = %d%n", a.compareTo (b));
//		System.out.printf ("b.compareTo(a) = %d%n", b.compareTo (a));
//		System.out.println();
//
//		System.out.printf ("a    = %s = %s%n", a.toStringHex(), a);
//		System.out.printf ("b    = %s = %s%n", b.toStringHex(), b);
//		c.assign (a) .not();
//		System.out.printf ("~a   = %s = %s%n", c.toStringHex(), c);
//		c.assign (b) .not();
//		System.out.printf ("~b   = %s = %s%n", c.toStringHex(), c);
//		c.assign (a) .xor (b);
//		System.out.printf ("a^b  = %s = %s%n", c.toStringHex(), c);
//		c.assign (b) .xor (a);
//		System.out.printf ("b^a  = %s = %s%n", c.toStringHex(), c);
//		c.assign (a) .or (b);
//		System.out.printf ("a|b  = %s = %s%n", c.toStringHex(), c);
//		c.assign (b) .or (a);
//		System.out.printf ("b|a  = %s = %s%n", c.toStringHex(), c);
//		c.assign (a) .and (b);
//		System.out.printf ("a&b  = %s = %s%n", c.toStringHex(), c);
//		c.assign (b) .and (a);
//		System.out.printf ("b&a  = %s = %s%n", c.toStringHex(), c);
//		c.assign (a) .andNot (b);
//		System.out.printf ("a&~b = %s = %s%n", c.toStringHex(), c);
//		c.assign (b) .andNot (a);
//		System.out.printf ("b&~a = %s = %s%n", c.toStringHex(), c);
//		}
//
//	/**
//	 * Print a usage message and exit.
//	 */
//	private static void usage()
//		{
//		System.err.println ("Usage: java BigInteger <n> <a> <b>");
//		System.exit (1);
//		}

//	/**
//	 * Unit test main program.
//	 */
//	public static void main
//		(String[] args)
//		throws Exception
//		{
//		if (args.length != 3) usage();
//		File file = new File (args[0]);
//		int n = Integer.parseInt (args[1]);
//		BigInteger a = new BigInteger (n) .assign (args[2]);
//
//		System.out.printf ("a = %s = %s%n", a.toStringHex(), a);
//
//		OutStream out = new OutStream (new FileOutputStream (file));
//		out.writeObject (a);
//		out.close();
//
//		InStream in = new InStream (new FileInputStream (file));
//		BigInteger b = (BigInteger) in.readObject();
//		in.close();
//
//		System.out.printf ("b = %s = %s%n", b.toStringHex(), b);
//		System.out.printf ("a == b ? %b%n", a.equals (b));
//		}
//
//	/**
//	 * Print a usage message and exit.
//	 */
//	private static void usage()
//		{
//		System.err.println ("Usage: java BigInteger <file> <n> <a>");
//		System.exit (1);
//		}

	}
