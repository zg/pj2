//******************************************************************************
//
// File:    Prng.java
// Package: edu.rit.gpu.example
// Unit:    Class edu.rit.gpu.example.Prng
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

package edu.rit.gpu.example;

import edu.rit.gpu.Struct;
import java.nio.ByteBuffer;

/**
 * Class Prng provides a pseudorandom number generator (PRNG). It is the Java
 * version of the CUDA <TT>prng_t</TT> structure in file Prng.cu. Class Prng can
 * be used to mirror PRNG objects between the CPU and the GPU.
 *
 * @author  Alan Kaminsky
 * @version 10-Feb-2016
 */
public class Prng
	extends Struct
	{

// Hidden constants.

	private static final long A = 3935559000370003845L;
	private static final long B = 2691343689449507681L;
	private static final long C = 4768777513237032717L;

	// 2^{-64}
	private static final double TWO_SUP_MINUS_64 =
		1.0/18446744073709551616.0;

// Hidden data members.

	private long counter;

// Hidden operations.

	/**
	 * Return the hash of the given value.
	 */
	private static long hash
		(long x)
		{
		x = A*x + B;
		x = x ^ (x >>> 21);
		x = x ^ (x << 37);
		x = x ^ (x >>> 4);
		x = C*x;
		x = x ^ (x << 20);
		x = x ^ (x >>> 41);
		x = x ^ (x << 5);
		return x;
		}

	/**
	 * Return the next 64-bit value in this PRNG's sequence.
	 */
	private long next()
		{
		return hash (++ counter);
		}

// Exported constructors.

	/**
	 * Construct a new PRNG.
	 */
	public Prng()
		{
		}

// Exported operations.

	/**
	 * Set this PRNG's seed.
	 *
	 * @param  seed  Seed.
	 */
	public void setSeed
		(long seed)
		{
		counter = hash (seed);
		}

	/**
	 * Skip this PRNG the given number of positions ahead in this PRNG's
	 * sequence.
	 *
	 * @param  n  Number of positions to skip.
	 */
	public void skip
		(long n)
		{
		counter += n;
		}

	/**
	 * Return the Boolean value from the next pseudorandom value in this PRNG's
	 * sequence. With a probability of 0.5 true is returned, with a probability
	 * of 0.5 false is returned.
	 *
	 * @return  Boolean value.
	 */
	public boolean nextBoolean()
		{
		// Use the high-order bit of the 64-bit random value.
		return (next() >>> 63) != 0;
		}

	/**
	 * Return the byte value from the next pseudorandom value in this PRNG's
	 * sequence. Each value in the range -128 through 127 is returned with a
	 * probability of 1/2<SUP>8</SUP>.
	 *
	 * @return  Byte value.
	 */
	public byte nextByte()
		{
		return (byte) next();
		}

	/**
	 * Return the unsigned byte value from the next pseudorandom value in this
	 * PRNG's sequence. Each value in the range 0 through 255 is returned with a
	 * probability of 1/2<SUP>8</SUP>.
	 *
	 * @return  Unsigned byte value.
	 */
	public int nextUnsignedByte()
		{
		return (int) next() & 255;
		}

	/**
	 * Return the short value from the next pseudorandom value in this PRNG's
	 * sequence. Each value in the range -32768 through 32767 is returned with a
	 * probability of 1/2<SUP>16</SUP>.
	 *
	 * @return  Short value.
	 */
	public short nextShort()
		{
		return (short) next();
		}

	/**
	 * Return the unsigned short value from the next pseudorandom value in this
	 * PRNG's sequence. Each value in the range 0 through 65535 is returned with
	 * a probability of 1/2<SUP>16</SUP>.
	 *
	 * @return  Unsigned short value.
	 */
	public int nextUnsignedShort()
		{
		return (int) next() & 65535;
		}

	/**
	 * Return the integer value from the next pseudorandom value in this PRNG's
	 * sequence. Each value in the range -2147483648 through 2147483647 is
	 * returned with a probability of 1/2<SUP>32</SUP>.
	 *
	 * @return  Integer value.
	 */
	public int nextInteger()
		{
		return (int) next();
		}

	/**
	 * Return the unsigned integer value from the next pseudorandom value in
	 * this PRNG's sequence. Each value in the range 0 through 4294967296 is
	 * returned with a probability of 1/2<SUP>32</SUP>.
	 *
	 * @return  Unsigned integer value.
	 */
	public int nextUnsignedInteger()
		{
		return (int) next();
		}

	/**
	 * Return the long value from the next pseudorandom value in this PRNG's
	 * sequence. Each value in the range -9223372036854775808 through
	 * 9223372036854775807 is returned with a probability of 1/2<SUP>64</SUP>.
	 *
	 * @return  Long value.
	 */
	public long nextLong()
		{
		return next();
		}

	/**
	 * Return the unsigned long value from the next pseudorandom value in this
	 * PRNG's sequence. Each value in the range 0 through 18446744073709551615
	 * is returned with a probability of 1/2<SUP>64</SUP>.
	 *
	 * @return  Unsigned long value.
	 */
	public long nextUnsignedLong()
		{
		return next();
		}

	/**
	 * Return the double precision floating point value from the next
	 * pseudorandom value in this PRNG's sequence. The returned numbers have a
	 * uniform distribution in the range 0.0 (inclusive) to 1.0 (exclusive).
	 *
	 * @return  Double value.
	 */
	public double nextDouble()
		{
		return next()*TWO_SUP_MINUS_64 + 0.5;
		}

	/**
	 * Return the single precision floating point value from the next
	 * pseudorandom value in this PRNG's sequence. The returned numbers have a
	 * uniform distribution in the range 0.0 (inclusive) to 1.0 (exclusive).
	 *
	 * @return  Float value.
	 */
	public float nextFloat()
		{
		return (float) nextDouble();
		}

	/**
	 * Return the integer value in the given range from the next pseudorandom
	 * value in this PRNG's sequence. Each value in the range 0 through
	 * <TT>n</TT>&minus;1 is returned with a probability of 1/<TT>n</TT>.
	 *
	 * @param  n  Range of values to return.
	 *
	 * @return  Integer value in the range 0 through <TT>n</TT>&minus;1
	 *          inclusive.
	 */
	public int nextInt
		(int n)
		{
		return (int) (nextDouble() * n);
		}

	/**
	 * Returns the size in bytes of the C struct. The size must include any
	 * internal padding bytes needed to align the fields of the C struct. The
	 * size must include any padding bytes at the end needed to align a series
	 * of C structs in an array.
	 *
	 * @return  Size of C struct (bytes).
	 */
	public static long sizeof()
		{
		return 8;
		}

	/**
	 * Write this Java object to the given byte buffer in the form of a C
	 * struct. The byte buffer's byte order is little endian. The byte buffer is
	 * positioned at the first byte of the C struct. The <TT>toStruct()</TT>
	 * method must write this object's fields into the byte buffer exactly as
	 * the C struct is laid out in GPU memory.
	 *
	 * @param  buf  Byte buffer to write.
	 */
	public void toStruct
		(ByteBuffer buf)
		{
		buf.putLong (counter);
		}

	/**
	 * Read this Java object from the given byte buffer in the form of a C
	 * struct. The byte buffer's byte order is little endian. The byte buffer is
	 * positioned at the first byte of the C struct. The <TT>fromStruct()</TT>
	 * method must read this object's fields from the byte buffer exactly as the
	 * C struct is laid out in GPU memory.
	 *
	 * @param  buf  Byte buffer to read.
	 */
	public void fromStruct
		(ByteBuffer buf)
		{
		counter = buf.getLong();
		}

	}
