//******************************************************************************
//
// File:    WV.java
// Package: edu.rit.gpu.example
// Unit:    Class edu.rit.gpu.example.WV
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
 * Class WV encapsulates the weight and value of an item in a knapsack problem.
 * It is also used to hold the total weight and total value of all the items in
 * the knapsack. It is the Java version of the CUDA <TT>wv_t</TT> structure in
 * files KnapsackExhGpu.cu and WalkSackGpu.cu. Class WV can be used to mirror
 * weight/value objects between the CPU and the GPU.
 *
 * @author  Alan Kaminsky
 * @version 10-Mar-2016
 */
public class WV
	extends Struct
	implements Cloneable
	{

// Exported data members.

	/**
	 * Weight.
	 */
	public long weight;

	/**
	 * Value.
	 */
	public long value;

// Exported constructors.

	/**
	 * Construct a new weight/value object. The weight is 0. The value is 0.
	 */
	public WV()
		{
		}

	/**
	 * Construct a new weight/value object with the given weight and value.
	 *
	 * @param  weight  Weight.
	 * @param  value   Value.
	 */
	public WV
		(long weight,
		 long value)
		{
		this.weight = weight;
		this.value = value;
		}

	/**
	 * Construct a new weight/value object that is a copy of the given
	 * weight/value object.
	 *
	 * @param  wv  Weight/value object to copy.
	 */
	public WV
		(WV wv)
		{
		copy (wv);
		}

// Exported operations.

	/**
	 * Set this weight/value object to a copy of the given weight/value object.
	 *
	 * @param  wv  Weight/value object to copy.
	 */
	public void copy
		(WV wv)
		{
		this.weight = wv.weight;
		this.value = wv.value;
		}

	/**
	 * Create a clone of this weight/value object.
	 *
	 * @return  Clone.
	 */
	public Object clone()
		{
		try
			{
			return super.clone();
			}
		catch (CloneNotSupportedException exc)
			{
			// Shouldn't happen.
			throw new IllegalStateException (exc);
			}
		}

	/**
	 * Determine if this weight/value object is better than the given
	 * weight/value object with respect to the given capacity.
	 *
	 * @param  wv  Weight/value object to compare with.
	 * @param  C   Knapsack capacity.
	 *
	 * @return  True if this weight/value object is better than <TT>wv</TT>,
	 *          false otherwise.
	 */
	public boolean isBetterThan
		(WV wv,
		 long C)
		{
		if (this.weight > C && wv.weight > C)
			return this.weight < wv.weight;
		else if (this.weight > C)
			return false;
		else if (wv.weight > C)
			return true;
		else
			return this.value > wv.value;
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
		return 16;
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
		buf.putLong (weight);
		buf.putLong (value);
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
		weight = buf.getLong();
		value = buf.getLong();
		}

	/**
	 * Returns a string version of this weight/value object.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		return String.format ("(weight=%d,value=%d)", weight, value);
		}

	}
