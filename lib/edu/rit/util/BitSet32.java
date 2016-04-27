//******************************************************************************
//
// File:    BitSet32.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.BitSet32
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

package edu.rit.util;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import java.io.IOException;

/**
 * Class BitSet32 provides a set of integers from 0 to 31. The set elements are
 * stored in a bitmap representation.
 * <P>
 * The bitmap representation is a value of type <TT>int</TT>. Bit 0 of the
 * bitmap (the least significant bit) corresponds to set element 0, bit 1 of the
 * bitmap (the next least significant bit) corresponds to set element 1, and so
 * on. A bit of the bitmap is 1 if the set contains the corresponding element; a
 * bit of the bitmap is 0 if the set does not contain the corresponding element.
 * <P>
 * <I>Note:</I> Class BitSet32 is not multiple thread safe.
 *
 * @author  Alan Kaminsky
 * @version 24-Mar-2015
 */
public class BitSet32
	implements Streamable
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

// Hidden data members.

	private int bitmap;

// Exported constructors.

	/**
	 * Construct a new empty set.
	 */
	public BitSet32()
		{
		}

	/**
	 * Construct a new set with the elements in the given bitmap.
	 *
	 * @param  bitmap  Bitmap of set elements.
	 */
	public BitSet32
		(int bitmap)
		{
		this.bitmap = bitmap;
		}

	/**
	 * Construct a new set that is a copy of the given set.
	 *
	 * @param  set  Set to copy.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>set</TT> is null.
	 */
	public BitSet32
		(BitSet32 set)
		{
		copy (set);
		}

// Exported operations.

	/**
	 * Determine if this set is empty.
	 *
	 * @return  True if this set is empty, false otherwise.
	 */
	public boolean isEmpty()
		{
		return bitmap == 0;
		}

	/**
	 * Clear this set.
	 *
	 * @return  This set.
	 */
	public BitSet32 clear()
		{
		bitmap = 0;
		return this;
		}

	/**
	 * Change this set to be a copy of the given set.
	 *
	 * @param  set  Set to copy.
	 *
	 * @return  This set.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>set</TT> is null.
	 */
	public BitSet32 copy
		(BitSet32 set)
		{
		this.bitmap = set.bitmap;
		return this;
		}

	/**
	 * Returns the number of elements in this set.
	 *
	 * @return  Number of elements.
	 */
	public int size()
		{
		return Integer.bitCount (bitmap);
		}

	/**
	 * Determine if this set contains the given element. If <TT>elem</TT> is not
	 * in the range 0 .. 31, false is returned.
	 *
	 * @param  elem  Element.
	 *
	 * @return  True if this set contains <TT>elem</TT>, false otherwise.
	 */
	public boolean contains
		(int elem)
		{
		return (bitmap & maskForElement (elem)) != 0;
		}

	/**
	 * Add the given element to this set. If <TT>elem</TT> is not in the range 0
	 * .. 31, this set is unchanged.
	 *
	 * @param  elem  Element.
	 *
	 * @return  This set.
	 */
	public BitSet32 add
		(int elem)
		{
		bitmap |= maskForElement (elem);
		return this;
		}

	/**
	 * Add all elements in the given range to this set. All elements from
	 * <TT>lb</TT> through <TT>ub</TT>&minus;1, inclusive, are added to this
	 * set. If any element is not in the range 0 .. 31, that element is not
	 * added. If <TT>lb</TT> &ge; <TT>ub</TT>, this set is unchanged.
	 *
	 * @param  lb  Lower bound element (inclusive).
	 * @param  ub  Upper bound element (exclusive).
	 *
	 * @return  This set.
	 */
	public BitSet32 add
		(int lb,
		 int ub)
		{
		bitmap |= maskForRange (lb, ub);
		return this;
		}

	/**
	 * Remove the given element from this set. If <TT>elem</TT> is not in the
	 * range 0 .. 31, this set is unchanged.
	 *
	 * @param  elem  Element.
	 *
	 * @return  This set.
	 */
	public BitSet32 remove
		(int elem)
		{
		bitmap &= ~maskForElement (elem);
		return this;
		}

	/**
	 * Remove all elements in the given range from this set. All elements from
	 * <TT>lb</TT> through <TT>ub</TT>&minus;1, inclusive, are removed from this
	 * set. If any element is not in the range 0 .. 31, that element is not
	 * removed. If <TT>lb</TT> &ge; <TT>ub</TT>, this set is unchanged.
	 *
	 * @param  lb  Lower bound element (inclusive).
	 * @param  ub  Upper bound element (exclusive).
	 *
	 * @return  This set.
	 */
	public BitSet32 remove
		(int lb,
		 int ub)
		{
		bitmap &= ~maskForRange (lb, ub);
		return this;
		}

	/**
	 * Flip the given element. If this set contains <TT>elem</TT>, it is
	 * removed; if this set does not contain <TT>elem</TT>, it is added. If
	 * <TT>elem</TT> is not in the range 0 .. 31, this set is unchanged.
	 *
	 * @param  elem  Element.
	 *
	 * @return  This set.
	 */
	public BitSet32 flip
		(int elem)
		{
		bitmap ^= maskForElement (elem);
		return this;
		}

	/**
	 * Flip all elements in the given range. All elements from <TT>lb</TT>
	 * through <TT>ub</TT>&minus;1, inclusive, are flipped. If this set contains
	 * such an element, it is removed; if this set does not contain such an
	 * element, it is added. If any element is not in the range 0 .. 31, that
	 * element is not flipped. If <TT>lb</TT> &ge; <TT>ub</TT>, this set is
	 * unchanged.
	 *
	 * @param  lb  Lower bound element (inclusive).
	 * @param  ub  Upper bound element (exclusive).
	 *
	 * @return  This set.
	 */
	public BitSet32 flip
		(int lb,
		 int ub)
		{
		bitmap ^= maskForRange (lb, ub);
		return this;
		}

	/**
	 * Determine if this set is a subset of the given set. This is so if every
	 * element of this set is also an element of the given set.
	 *
	 * @param  set  Set.
	 *
	 * @return  True if this set is a subset of the given set, false otherwise.
	 */
	public boolean isSubsetOf
		(BitSet32 set)
		{
		return (this.bitmap & set.bitmap) == this.bitmap;
		}

	/**
	 * Change this set to be the union of itself and the given set. The union
	 * consists of all elements that appear in this set or the given set or
	 * both.
	 *
	 * @param  set  Set.
	 *
	 * @return  This set.
	 */
	public BitSet32 union
		(BitSet32 set)
		{
		this.bitmap |= set.bitmap;
		return this;
		}

	/**
	 * Change this set to be the intersection of itself and the given set. The
	 * intersection consists of all elements that appear in this set and the
	 * given set.
	 *
	 * @param  set  Set.
	 *
	 * @return  This set.
	 */
	public BitSet32 intersection
		(BitSet32 set)
		{
		this.bitmap &= set.bitmap;
		return this;
		}

	/**
	 * Change this set to be the difference of itself and the given set. The
	 * difference consists of all elements that appear in this set and not in
	 * the given set.
	 *
	 * @param  set  Set.
	 *
	 * @return  This set.
	 */
	public BitSet32 difference
		(BitSet32 set)
		{
		this.bitmap &= ~set.bitmap;
		return this;
		}

	/**
	 * Change this set to be the symmetric difference of itself and the given
	 * set. The symmetric difference consists of all elements that appear in
	 * this set or the given set, but not both.
	 *
	 * @param  set  Set.
	 *
	 * @return  This set.
	 */
	public BitSet32 symmetricDifference
		(BitSet32 set)
		{
		this.bitmap ^= set.bitmap;
		return this;
		}

	/**
	 * Perform the given action on each element in this set. For each element in
	 * this set from smallest to largest, the given <TT>action</TT>'s
	 * <TT>run()</TT> method is called, passing in the element.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds elements to or removes
	 * elements from the set, the <TT>forEachItemDo()</TT> method's behavior is
	 * unspecified.
	 *
	 * @param  action  Action.
	 */
	public void forEachItemDo
		(IntAction action)
		{
		int b = bitmap;
		for (int i = 0; b != 0; ++ i, b >>>= 1)
			if ((b & 1) != 0)
				action.run (i);
		}

	/**
	 * Perform the given action on each element in this set and return a result.
	 * For each element in this set from smallest to largest, the given
	 * <TT>action</TT>'s <TT>run()</TT> method is called, passing in the
	 * element. After all the elements have been processed, the given
	 * <TT>action</TT>'s <TT>result()</TT> method is called, and its result is
	 * returned.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds elements to or removes
	 * elements from the set, the <TT>forEachItemDo()</TT> method's behavior is
	 * unspecified.
	 *
	 * @param  <R>     Result data type.
	 * @param  action  Action.
	 *
	 * @return  Result of processing all the set elements.
	 */
	public <R> R forEachItemDo
		(IntActionResult<R> action)
		{
		int b = bitmap;
		for (int i = 0; b != 0; ++ i, b >>>= 1)
			if ((b & 1) != 0)
				action.run (i);
		return action.result();
		}

	/**
	 * Evaluate the given predicate on, and possibly remove, each element in
	 * this set. For each element in this set from smallest to largest, the
	 * given <TT>predicate</TT>'s <TT>test()</TT> method is called, passing in
	 * the element. If the <TT>test()</TT> method returns true, the element is
	 * removed from this set.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>predicate</TT> adds elements to or
	 * removes elements from the set, other than by returning true, the
	 * <TT>removeEachItemIf()</TT> method's behavior is not specified.
	 *
	 * @param  predicate  Predicate.
	 */
	public void removeEachItemIf
		(IntPredicate predicate)
		{
		int b = bitmap;
		for (int i = 0; b != 0; ++ i, b >>>= 1)
			if ((b & 1) != 0 && predicate.test (i))
				remove (i);
		}

	/**
	 * Obtain this set's elements in the form of a bitmap.
	 *
	 * @return  Bitmap of set elements.
	 */
	public int bitmap()
		{
		return bitmap;
		}

	/**
	 * Change this set's elements to be those in the given bitmap.
	 *
	 * @param  bitmap  Bitmap of set elements.
	 *
	 * @return  This set.
	 */
	public BitSet32 bitmap
		(int bitmap)
		{
		this.bitmap = bitmap;
		return this;
		}

	/**
	 * Return an element in this set chosen uniformly at random
	 *
	 * @param  prng  Pseudorandom number generator.
	 *
	 * @return  Random element, or &minus;1 if this set is empty.
	 */
	public int randomElement
		(Random prng)
		{
		int n = Integer.bitCount (bitmap);
		if (n == 0) return -1;
		int i = prng.nextInt (n);
		int e = 0;
		for (;;)
			{
			if ((bitmap & (1 << e)) != 0)
				{
				if (i == 0) break;
				-- i;
				}
			++ e;
			}
		return e;
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
		out.writeUnsignedInt (bitmap);
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
		bitmap = in.readUnsignedInt();
		}

// Hidden operations.

	/**
	 * Returns a mask that has a 1 bit at position <TT>elem</TT> and 0 bits
	 * elsewhere.
	 */
	private static int maskForElement
		(int elem)
		{
		return 0 <= elem && elem <= 31 ? 1 << elem : 0;
		}

	/**
	 * Returns a mask that has 1 bits at positions <TT>lb</TT> through
	 * <TT>ub</TT>&minus;1 inclusive and 0 bits elsewhere.
	 */
	private static int maskForRange
		(int lb,
		 int ub)
		{
		lb = Math.max (lb, 0);
		ub = Math.min (ub, 32);
		if (lb >= ub) return 0;
		int lbfill = lb == 32 ? 0xffffffff : (1 << lb) - 1;
		int ubfill = ub == 32 ? 0xffffffff : (1 << ub) - 1;
		return ubfill & ~lbfill;
		}

	}
