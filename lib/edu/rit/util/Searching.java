//******************************************************************************
//
// File:    Searching.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.Searching
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

import java.util.Comparator;

/**
 * Class Searching provides static methods for searching arrays of primitive
 * types and object types.
 * <P>
 * <I>Note:</I> The operations in class Searching are not multiple thread safe.
 *
 * @author  Alan Kaminsky
 * @version 27-Jan-2015
 */
public class Searching
	{

// Prevent construction.

	private Searching()
		{
		}

// Exported helper classes.

	/**
	 * Class Searching.Byte is the base class for a helper object used to
	 * search an array of type <TT>byte[]</TT>.
	 *
	 * @author  Alan Kaminsky
	 * @version 23-Mar-2015
	 */
	public static class Byte
		{
		/**
		 * An instance of the Searching.Byte base class.
		 */
		public static final Searching.Byte DEFAULT = new Searching.Byte();

		/**
		 * Compare two elements according to the desired ordering criterion.
		 * <P>
		 * The default implementation compares <TT>a</TT> and <TT>b</TT> using
		 * ascending order. A subclass can override this method to obtain a
		 * different ordering criterion; for example, descending order.
		 *
		 * @param  a  First element being compared.
		 * @param  b  Second element being compared.
		 *
		 * @return  A number less than, equal to, or greater than 0 if
		 *          <TT>a</TT> comes before, is the same as, or comes after
		 *          <TT>b</TT>, respectively.
		 */
		public int compare
			(byte a,
			 byte b)
			{
			return a - b;
			}
		}

	/**
	 * Class Searching.Character is the base class for a helper object used to
	 * search an array of type <TT>char[]</TT>.
	 *
	 * @author  Alan Kaminsky
	 * @version 23-Mar-2015
	 */
	public static class Character
		{
		/**
		 * An instance of the Searching.Character base class.
		 */
		public static final Searching.Character DEFAULT =
			new Searching.Character();

		/**
		 * Compare two elements according to the desired ordering criterion.
		 * <P>
		 * The default implementation compares <TT>a</TT> and <TT>b</TT> using
		 * ascending order. A subclass can override this method to obtain a
		 * different ordering criterion; for example, descending order.
		 *
		 * @param  a  First element being compared.
		 * @param  b  Second element being compared.
		 *
		 * @return  A number less than, equal to, or greater than 0 if
		 *          <TT>a</TT> comes before, is the same as, or comes after
		 *          <TT>b</TT>, respectively.
		 */
		public int compare
			(char a,
			 char b)
			{
			return a - b;
			}
		}

	/**
	 * Class Searching.Short is the base class for a helper object used to
	 * search an array of type <TT>short[]</TT>.
	 *
	 * @author  Alan Kaminsky
	 * @version 23-Mar-2015
	 */
	public static class Short
		{
		/**
		 * An instance of the Searching.Short base class.
		 */
		public static final Searching.Short DEFAULT = new Searching.Short();

		/**
		 * Compare two elements according to the desired ordering criterion.
		 * <P>
		 * The default implementation compares <TT>a</TT> and <TT>b</TT> using
		 * ascending order. A subclass can override this method to obtain a
		 * different ordering criterion; for example, descending order.
		 *
		 * @param  a  First element being compared.
		 * @param  b  Second element being compared.
		 *
		 * @return  A number less than, equal to, or greater than 0 if
		 *          <TT>a</TT> comes before, is the same as, or comes after
		 *          <TT>b</TT>, respectively.
		 */
		public int compare
			(short a,
			 short b)
			{
			return a - b;
			}
		}

	/**
	 * Class Searching.Integer is the base class for a helper object used to
	 * search an array of type <TT>int[]</TT>.
	 *
	 * @author  Alan Kaminsky
	 * @version 23-Mar-2015
	 */
	public static class Integer
		{
		/**
		 * An instance of the Searching.Integer base class.
		 */
		public static final Searching.Integer DEFAULT = new Searching.Integer();

		/**
		 * Compare two elements according to the desired ordering criterion.
		 * <P>
		 * The default implementation compares <TT>a</TT> and <TT>b</TT> using
		 * ascending order. A subclass can override this method to obtain a
		 * different ordering criterion; for example, descending order.
		 *
		 * @param  a  First element being compared.
		 * @param  b  Second element being compared.
		 *
		 * @return  A number less than, equal to, or greater than 0 if
		 *          <TT>a</TT> comes before, is the same as, or comes after
		 *          <TT>b</TT>, respectively.
		 */
		public int compare
			(int a,
			 int b)
			{
			return a - b;
			}
		}

	/**
	 * Class Searching.Long is the base class for a helper object used to
	 * search an array of type <TT>long[]</TT>.
	 *
	 * @author  Alan Kaminsky
	 * @version 23-Mar-2015
	 */
	public static class Long
		{
		/**
		 * An instance of the Searching.Long base class.
		 */
		public static final Searching.Long DEFAULT = new Searching.Long();

		/**
		 * Compare two elements according to the desired ordering criterion.
		 * <P>
		 * The default implementation compares <TT>a</TT> and <TT>b</TT> using
		 * ascending order. A subclass can override this method to obtain a
		 * different ordering criterion; for example, descending order.
		 *
		 * @param  a  First element being compared.
		 * @param  b  Second element being compared.
		 *
		 * @return  A number less than, equal to, or greater than 0 if
		 *          <TT>a</TT> comes before, is the same as, or comes after
		 *          <TT>b</TT>, respectively.
		 */
		public int compare
			(long a,
			 long b)
			{
			long d = a - b;
			return d < 0L ? -1 : d > 0L ? 1 : 0;
			}
		}

	/**
	 * Class Searching.Float is the base class for a helper object used to
	 * search an array of type <TT>float[]</TT>.
	 *
	 * @author  Alan Kaminsky
	 * @version 23-Mar-2015
	 */
	public static class Float
		{
		/**
		 * An instance of the Searching.Float base class.
		 */
		public static final Searching.Float DEFAULT = new Searching.Float();

		/**
		 * Compare two elements according to the desired ordering criterion.
		 * <P>
		 * The default implementation compares <TT>a</TT> and <TT>b</TT> using
		 * ascending order. A subclass can override this method to obtain a
		 * different ordering criterion; for example, descending order.
		 *
		 * @param  a  First element being compared.
		 * @param  b  Second element being compared.
		 *
		 * @return  A number less than, equal to, or greater than 0 if
		 *          <TT>a</TT> comes before, is the same as, or comes after
		 *          <TT>b</TT>, respectively.
		 */
		public int compare
			(float a,
			 float b)
			{
			float d = a - b;
			return d < 0.0f ? -1 : d > 0.0f ? 1 : 0;
			}
		}

	/**
	 * Class Searching.Double is the base class for a helper object used to
	 * search an array of type <TT>double[]</TT>.
	 *
	 * @author  Alan Kaminsky
	 * @version 23-Mar-2015
	 */
	public static class Double
		{
		/**
		 * An instance of the Searching.Double base class.
		 */
		public static final Searching.Double DEFAULT = new Searching.Double();

		/**
		 * Compare two elements according to the desired ordering criterion.
		 * <P>
		 * The default implementation compares <TT>a</TT> and <TT>b</TT> using
		 * ascending order. A subclass can override this method to obtain a
		 * different ordering criterion; for example, descending order.
		 *
		 * @param  a  First element being compared.
		 * @param  b  Second element being compared.
		 *
		 * @return  A number less than, equal to, or greater than 0 if
		 *          <TT>a</TT> comes before, is the same as, or comes after
		 *          <TT>b</TT>, respectively.
		 */
		public int compare
			(double a,
			 double b)
			{
			double d = a - b;
			return d < 0.0 ? -1 : d > 0.0 ? 1 : 0;
			}
		}

	/**
	 * Class Searching.Object is the base class for a helper object used to
	 * search an array of type <TT>T[]</TT>.
	 *
	 * @param  <T>  Array element data type.
	 *
	 * @author  Alan Kaminsky
	 * @version 22-Nov-2011
	 */
	public static abstract class Object<T>
		{
		/**
		 * Compare two elements according to the desired ordering criterion.
		 *
		 * @param  a  First element being compared.
		 * @param  b  Second element being compared.
		 *
		 * @return  A number less than, equal to, or greater than 0 if
		 *          <TT>a</TT> comes before, is the same as, or comes after
		 *          <TT>b</TT>, respectively.
		 */
		public abstract int compare
			(T a,
			 T b);
		}

	/**
	 * Class Searching.Comparable is the base class for a helper object used to
	 * search an array of type {@linkplain java.lang.Comparable
	 * Comparable}<TT>&lt;T&gt;[]</TT>.
	 *
	 * @param  <T>  Array element data type.
	 *
	 * @author  Alan Kaminsky
	 * @version 23-Mar-2015
	 */
	public static class Comparable<T extends java.lang.Comparable<T>>
		extends Searching.Object<T>
		{
		/**
		 * Compare two elements according to the desired ordering criterion.
		 * <P>
		 * The default implementation returns <TT>a.compareTo(b)</TT>.
		 *
		 * @param  a  First element being compared.
		 * @param  b  Second element being compared.
		 *
		 * @return  A number less than, equal to, or greater than 0 if
		 *          <TT>a</TT> comes before, is the same as, or comes after
		 *          <TT>b</TT>, respectively.
		 */
		public int compare
			(T a,
			 T b)
			{
			return a.compareTo (b);
			}
		}

// Exported operations.

	/**
	 * Search the given unordered array of type <TT>byte[]</TT> for the given
	 * element. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(byte[] x,
		 byte a)
		{
		return searchUnsorted (x, 0, x.length, a, Searching.Byte.DEFAULT);
		}

	/**
	 * Search a portion of the given unordered array of type <TT>byte[]</TT> for
	 * the given element. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchUnsorted
		(byte[] x,
		 int off,
		 int len,
		 byte a)
		{
		return searchUnsorted (x, off, len, a, Searching.Byte.DEFAULT);
		}

	/**
	 * Search the given unordered array of type <TT>byte[]</TT> for the given
	 * element. The given helper object is used to compare elements for
	 * equality only. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(byte[] x,
		 byte a,
		 Searching.Byte helper)
		{
		return searchUnsorted (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given unordered array of type <TT>byte[]</TT> for
	 * the given element. The given helper object is used to compare elements
	 * for equality only. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchUnsorted
		(byte[] x,
		 int off,
		 int len,
		 byte a,
		 Searching.Byte helper)
		{
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		for (int i = 0; i < len; ++ i)
			{
			if (helper.compare (x[i+off], a) == 0) return i;
			}
		return -1;
		}

	/**
	 * Search the given ordered array of type <TT>byte[]</TT> for the given
	 * element. It is assumed that the array is sorted in ascending order;
	 * otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(byte[] x,
		 byte a)
		{
		return searchSorted (x, 0, x.length, a, Searching.Byte.DEFAULT);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>byte[]</TT> for
	 * the given element. It is assumed that the array is sorted in ascending
	 * order; otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchSorted
		(byte[] x,
		 int off,
		 int len,
		 byte a)
		{
		return searchSorted (x, off, len, a, Searching.Byte.DEFAULT);
		}

	/**
	 * Search the given ordered array of type <TT>byte[]</TT> for the given
	 * element. The given helper object is used to compare elements for order
	 * and equality. It is assumed that the array is sorted in the order
	 * determined by the helper object; otherwise, the <TT>searchSorted()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(byte[] x,
		 byte a,
		 Searching.Byte helper)
		{
		return searchSorted (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>byte[]</TT> for
	 * the given element. The given helper object is used to compare elements
	 * for order and equality. It is assumed that the array is sorted in the
	 * order determined by the helper object; otherwise, the
	 * <TT>searchSorted()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchSorted
		(byte[] x,
		 int off,
		 int len,
		 byte a,
		 Searching.Byte helper)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (x.length == 0) return -1;

		int lo = off;
		int locomp = helper.compare (x[lo], a);
		if (locomp == 0) return lo;
		else if (locomp > 0) return -1;

		int hi = off + len - 1;
		int hicomp = helper.compare (x[hi], a);
		if (hicomp == 0) return hi;
		else if (hicomp < 0) return -1;

		// Loop invariant: x[lo] comes before a; x[hi] comes after a.
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			int midcomp = helper.compare (x[mid], a);
			if (midcomp == 0)
				{
				return mid;
				}
			else if (midcomp < 0)
				{
				lo = mid;
				locomp = midcomp;
				}
			else
				{
				hi = mid;
				hicomp = midcomp;
				}
			}

		return locomp == 0 ? lo : hicomp == 0 ? hi : -1;
		}

	/**
	 * Search the given ordered array of type <TT>byte[]</TT> for an interval
	 * containing the given element. It is assumed that the array is sorted in
	 * ascending order; otherwise, the <TT>searchInterval()</TT> method's
	 * behavior is not specified. An <I>O</I>(log <I>n</I>) binary search
	 * algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[0]</TT>, then <TT>i</TT> = 0
	 *     is returned. If <TT>x[x.length-1]</TT> &le; <TT>a</TT>, then
	 *     <TT>i</TT> = <TT>x.length</TT> is returned. If <TT>x</TT> is zero
	 *     length, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(byte[] x,
		 byte a)
		{
		return searchInterval (x, 0, x.length, a, Searching.Byte.DEFAULT);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>byte[]</TT> for
	 * an interval containing the given element. It is assumed that the array is
	 * sorted in ascending order; otherwise, the <TT>searchInterval()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[off]</TT>, then <TT>i</TT> =
	 *     <TT>off</TT> is returned. If <TT>x[off+len-1]</TT> &le; <TT>a</TT>,
	 *     then <TT>i</TT> = <TT>off+len</TT> is returned. If <TT>len</TT> = 0,
	 *     then <TT>i</TT> = &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchInterval
		(byte[] x,
		 int off,
		 int len,
		 byte a)
		{
		return searchInterval (x, off, len, a, Searching.Byte.DEFAULT);
		}

	/**
	 * Search the given ordered array of type <TT>byte[]</TT> for an interval
	 * containing the given element. The given helper object is used to compare
	 * elements for order and equality. It is assumed that the array is sorted
	 * in the order determined by the helper object; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[0]</TT>, then <TT>i</TT> = 0
	 *     is returned. If <TT>x[x.length-1]</TT> &le; <TT>a</TT>, then
	 *     <TT>i</TT> = <TT>x.length</TT> is returned. If <TT>x</TT> is zero
	 *     length, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(byte[] x,
		 byte a,
		 Searching.Byte helper)
		{
		return searchInterval (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>byte[]</TT> for
	 * an interval containing the given element. The given helper object is used
	 * to compare elements for order and equality. It is assumed that the array
	 * is sorted in the order determined by the helper object; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[off]</TT>, then <TT>i</TT> =
	 *     <TT>off</TT> is returned. If <TT>x[off+len-1]</TT> &le; <TT>a</TT>,
	 *     then <TT>i</TT> = <TT>off+len</TT> is returned. If <TT>len</TT> = 0,
	 *     then <TT>i</TT> = &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchInterval
		(byte[] x,
		 int off,
		 int len,
		 byte a,
		 Searching.Byte helper)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (len == 0) return -1;

		int lo = off;
		if (helper.compare (x[lo], a) > 0) return off;

		int hi = off + len - 1;
		if (helper.compare (x[hi], a) <= 0) return off + len;

		// Loop invariant: x[lo] <= a and a < x[hi].
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			if (helper.compare (x[mid], a) <= 0)
				lo = mid;
			else
				hi = mid;
			}

		return hi;
		}

	/**
	 * Search the given unordered array of type <TT>char[]</TT> for the given
	 * element. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(char[] x,
		 char a)
		{
		return searchUnsorted (x, 0, x.length, a, Searching.Character.DEFAULT);
		}

	/**
	 * Search a portion of the given unordered array of type <TT>char[]</TT> for
	 * the given element. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchUnsorted
		(char[] x,
		 int off,
		 int len,
		 char a)
		{
		return searchUnsorted (x, off, len, a, Searching.Character.DEFAULT);
		}

	/**
	 * Search the given unordered array of type <TT>char[]</TT> for the given
	 * element. The given helper object is used to compare elements for
	 * equality only. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(char[] x,
		 char a,
		 Searching.Character helper)
		{
		return searchUnsorted (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given unordered array of type <TT>char[]</TT> for
	 * the given element. The given helper object is used to compare elements
	 * for equality only. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchUnsorted
		(char[] x,
		 int off,
		 int len,
		 char a,
		 Searching.Character helper)
		{
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		for (int i = 0; i < len; ++ i)
			{
			if (helper.compare (x[i+off], a) == 0) return i;
			}
		return -1;
		}

	/**
	 * Search the given ordered array of type <TT>char[]</TT> for the given
	 * element. It is assumed that the array is sorted in ascending order;
	 * otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(char[] x,
		 char a)
		{
		return searchSorted (x, 0, x.length, a, Searching.Character.DEFAULT);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>char[]</TT> for
	 * the given element. It is assumed that the array is sorted in ascending
	 * order; otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchSorted
		(char[] x,
		 int off,
		 int len,
		 char a)
		{
		return searchSorted (x, off, len, a, Searching.Character.DEFAULT);
		}

	/**
	 * Search the given ordered array of type <TT>char[]</TT> for the given
	 * element. The given helper object is used to compare elements for order
	 * and equality. It is assumed that the array is sorted in the order
	 * determined by the helper object; otherwise, the <TT>searchSorted()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(char[] x,
		 char a,
		 Searching.Character helper)
		{
		return searchSorted (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>char[]</TT> for
	 * the given element. The given helper object is used to compare elements
	 * for order and equality. It is assumed that the array is sorted in the
	 * order determined by the helper object; otherwise, the
	 * <TT>searchSorted()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchSorted
		(char[] x,
		 int off,
		 int len,
		 char a,
		 Searching.Character helper)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (x.length == 0) return -1;

		int lo = off;
		int locomp = helper.compare (x[lo], a);
		if (locomp == 0) return lo;
		else if (locomp > 0) return -1;

		int hi = off + len - 1;
		int hicomp = helper.compare (x[hi], a);
		if (hicomp == 0) return hi;
		else if (hicomp < 0) return -1;

		// Loop invariant: x[lo] comes before a; x[hi] comes after a.
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			int midcomp = helper.compare (x[mid], a);
			if (midcomp == 0)
				{
				return mid;
				}
			else if (midcomp < 0)
				{
				lo = mid;
				locomp = midcomp;
				}
			else
				{
				hi = mid;
				hicomp = midcomp;
				}
			}

		return locomp == 0 ? lo : hicomp == 0 ? hi : -1;
		}

	/**
	 * Search the given ordered array of type <TT>char[]</TT> for an interval
	 * containing the given element. It is assumed that the array is sorted in
	 * ascending order; otherwise, the <TT>searchInterval()</TT> method's
	 * behavior is not specified. An <I>O</I>(log <I>n</I>) binary search
	 * algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[0]</TT>, then <TT>i</TT> = 0
	 *     is returned. If <TT>x[x.length-1]</TT> &le; <TT>a</TT>, then
	 *     <TT>i</TT> = <TT>x.length</TT> is returned. If <TT>x</TT> is zero
	 *     length, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(char[] x,
		 char a)
		{
		return searchInterval (x, 0, x.length, a, Searching.Character.DEFAULT);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>char[]</TT> for
	 * an interval containing the given element. It is assumed that the array is
	 * sorted in ascending order; otherwise, the <TT>searchInterval()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[off]</TT>, then <TT>i</TT> =
	 *     <TT>off</TT> is returned. If <TT>x[off+len-1]</TT> &le; <TT>a</TT>,
	 *     then <TT>i</TT> = <TT>off+len</TT> is returned. If <TT>len</TT> = 0,
	 *     then <TT>i</TT> = &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchInterval
		(char[] x,
		 int off,
		 int len,
		 char a)
		{
		return searchInterval (x, off, len, a, Searching.Character.DEFAULT);
		}

	/**
	 * Search the given ordered array of type <TT>char[]</TT> for an interval
	 * containing the given element. The given helper object is used to compare
	 * elements for order and equality. It is assumed that the array is sorted
	 * in the order determined by the helper object; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[0]</TT>, then <TT>i</TT> = 0
	 *     is returned. If <TT>x[x.length-1]</TT> &le; <TT>a</TT>, then
	 *     <TT>i</TT> = <TT>x.length</TT> is returned. If <TT>x</TT> is zero
	 *     length, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(char[] x,
		 char a,
		 Searching.Character helper)
		{
		return searchInterval (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>char[]</TT> for
	 * an interval containing the given element. The given helper object is used
	 * to compare elements for order and equality. It is assumed that the array
	 * is sorted in the order determined by the helper object; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[off]</TT>, then <TT>i</TT> =
	 *     <TT>off</TT> is returned. If <TT>x[off+len-1]</TT> &le; <TT>a</TT>,
	 *     then <TT>i</TT> = <TT>off+len</TT> is returned. If <TT>len</TT> = 0,
	 *     then <TT>i</TT> = &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchInterval
		(char[] x,
		 int off,
		 int len,
		 char a,
		 Searching.Character helper)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (len == 0) return -1;

		int lo = off;
		if (helper.compare (x[lo], a) > 0) return off;

		int hi = off + len - 1;
		if (helper.compare (x[hi], a) <= 0) return off + len;

		// Loop invariant: x[lo] <= a and a < x[hi].
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			if (helper.compare (x[mid], a) <= 0)
				lo = mid;
			else
				hi = mid;
			}

		return hi;
		}

	/**
	 * Search the given unordered array of type <TT>short[]</TT> for the given
	 * element. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(short[] x,
		 short a)
		{
		return searchUnsorted (x, 0, x.length, a, Searching.Short.DEFAULT);
		}

	/**
	 * Search a portion of the given unordered array of type <TT>short[]</TT>
	 * for the given element. An <I>O</I>(<I>n</I>) linear search algorithm is
	 * used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchUnsorted
		(short[] x,
		 int off,
		 int len,
		 short a)
		{
		return searchUnsorted (x, off, len, a, Searching.Short.DEFAULT);
		}

	/**
	 * Search the given unordered array of type <TT>short[]</TT> for the given
	 * element. The given helper object is used to compare elements for
	 * equality only. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(short[] x,
		 short a,
		 Searching.Short helper)
		{
		return searchUnsorted (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given unordered array of type <TT>short[]</TT>
	 * for the given element. The given helper object is used to compare
	 * elements for equality only. An <I>O</I>(<I>n</I>) linear search algorithm
	 * is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchUnsorted
		(short[] x,
		 int off,
		 int len,
		 short a,
		 Searching.Short helper)
		{
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		for (int i = 0; i < len; ++ i)
			{
			if (helper.compare (x[i+off], a) == 0) return i;
			}
		return -1;
		}

	/**
	 * Search the given ordered array of type <TT>short[]</TT> for the given
	 * element. It is assumed that the array is sorted in ascending order;
	 * otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(short[] x,
		 short a)
		{
		return searchSorted (x, 0, x.length, a, Searching.Short.DEFAULT);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>short[]</TT> for
	 * the given element. It is assumed that the array is sorted in ascending
	 * order; otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchSorted
		(short[] x,
		 int off,
		 int len,
		 short a)
		{
		return searchSorted (x, off, len, a, Searching.Short.DEFAULT);
		}

	/**
	 * Search the given ordered array of type <TT>short[]</TT> for the given
	 * element. The given helper object is used to compare elements for order
	 * and equality. It is assumed that the array is sorted in the order
	 * determined by the helper object; otherwise, the <TT>searchSorted()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(short[] x,
		 short a,
		 Searching.Short helper)
		{
		return searchSorted (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>short[]</TT> for
	 * the given element. The given helper object is used to compare elements
	 * for order and equality. It is assumed that the array is sorted in the
	 * order determined by the helper object; otherwise, the
	 * <TT>searchSorted()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchSorted
		(short[] x,
		 int off,
		 int len,
		 short a,
		 Searching.Short helper)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (x.length == 0) return -1;

		int lo = off;
		int locomp = helper.compare (x[lo], a);
		if (locomp == 0) return lo;
		else if (locomp > 0) return -1;

		int hi = off + len - 1;
		int hicomp = helper.compare (x[hi], a);
		if (hicomp == 0) return hi;
		else if (hicomp < 0) return -1;

		// Loop invariant: x[lo] comes before a; x[hi] comes after a.
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			int midcomp = helper.compare (x[mid], a);
			if (midcomp == 0)
				{
				return mid;
				}
			else if (midcomp < 0)
				{
				lo = mid;
				locomp = midcomp;
				}
			else
				{
				hi = mid;
				hicomp = midcomp;
				}
			}

		return locomp == 0 ? lo : hicomp == 0 ? hi : -1;
		}

	/**
	 * Search the given ordered array of type <TT>short[]</TT> for an interval
	 * containing the given element. It is assumed that the array is sorted in
	 * ascending order; otherwise, the <TT>searchInterval()</TT> method's
	 * behavior is not specified. An <I>O</I>(log <I>n</I>) binary search
	 * algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[0]</TT>, then <TT>i</TT> = 0
	 *     is returned. If <TT>x[x.length-1]</TT> &le; <TT>a</TT>, then
	 *     <TT>i</TT> = <TT>x.length</TT> is returned. If <TT>x</TT> is zero
	 *     length, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(short[] x,
		 short a)
		{
		return searchInterval (x, 0, x.length, a, Searching.Short.DEFAULT);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>short[]</TT> for
	 * an interval containing the given element. It is assumed that the array is
	 * sorted in ascending order; otherwise, the <TT>searchInterval()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[off]</TT>, then <TT>i</TT> =
	 *     <TT>off</TT> is returned. If <TT>x[off+len-1]</TT> &le; <TT>a</TT>,
	 *     then <TT>i</TT> = <TT>off+len</TT> is returned. If <TT>len</TT> = 0,
	 *     then <TT>i</TT> = &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchInterval
		(short[] x,
		 int off,
		 int len,
		 short a)
		{
		return searchInterval (x, off, len, a, Searching.Short.DEFAULT);
		}

	/**
	 * Search the given ordered array of type <TT>short[]</TT> for an interval
	 * containing the given element. The given helper object is used to compare
	 * elements for order and equality. It is assumed that the array is sorted
	 * in the order determined by the helper object; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[0]</TT>, then <TT>i</TT> = 0
	 *     is returned. If <TT>x[x.length-1]</TT> &le; <TT>a</TT>, then
	 *     <TT>i</TT> = <TT>x.length</TT> is returned. If <TT>x</TT> is zero
	 *     length, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(short[] x,
		 short a,
		 Searching.Short helper)
		{
		return searchInterval (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>short[]</TT> for
	 * an interval containing the given element. The given helper object is used
	 * to compare elements for order and equality. It is assumed that the array
	 * is sorted in the order determined by the helper object; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[off]</TT>, then <TT>i</TT> =
	 *     <TT>off</TT> is returned. If <TT>x[off+len-1]</TT> &le; <TT>a</TT>,
	 *     then <TT>i</TT> = <TT>off+len</TT> is returned. If <TT>len</TT> = 0,
	 *     then <TT>i</TT> = &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchInterval
		(short[] x,
		 int off,
		 int len,
		 short a,
		 Searching.Short helper)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (len == 0) return -1;

		int lo = off;
		if (helper.compare (x[lo], a) > 0) return off;

		int hi = off + len - 1;
		if (helper.compare (x[hi], a) <= 0) return off + len;

		// Loop invariant: x[lo] <= a and a < x[hi].
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			if (helper.compare (x[mid], a) <= 0)
				lo = mid;
			else
				hi = mid;
			}

		return hi;
		}

	/**
	 * Search the given unordered array of type <TT>int[]</TT> for the given
	 * element. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(int[] x,
		 int a)
		{
		return searchUnsorted (x, 0, x.length, a, Searching.Integer.DEFAULT);
		}

	/**
	 * Search a portion of the given unordered array of type <TT>int[]</TT> for
	 * the given element. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchUnsorted
		(int[] x,
		 int off,
		 int len,
		 int a)
		{
		return searchUnsorted (x, off, len, a, Searching.Integer.DEFAULT);
		}

	/**
	 * Search the given unordered integer list for the given element. An
	 * <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x  List to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(IntList x,
		 int a)
		{
		return searchUnsorted (x.item, 0, x.size, a, Searching.Integer.DEFAULT);
		}

	/**
	 * Search the given unordered array of type <TT>int[]</TT> for the given
	 * element. The given helper object is used to compare elements for
	 * equality only. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(int[] x,
		 int a,
		 Searching.Integer helper)
		{
		return searchUnsorted (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given unordered array of type <TT>int[]</TT> for
	 * the given element. The given helper object is used to compare elements
	 * for equality only. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchUnsorted
		(int[] x,
		 int off,
		 int len,
		 int a,
		 Searching.Integer helper)
		{
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		for (int i = 0; i < len; ++ i)
			{
			if (helper.compare (x[i+off], a) == 0) return i;
			}
		return -1;
		}

	/**
	 * Search the given unordered integer list for the given element. The given
	 * helper object is used to compare elements for equality only. An
	 * <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x       List to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(IntList x,
		 int a,
		 Searching.Integer helper)
		{
		return searchUnsorted (x.item, 0, x.size, a, helper);
		}

	/**
	 * Search the given ordered array of type <TT>int[]</TT> for the given
	 * element. It is assumed that the array is sorted in ascending order;
	 * otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(int[] x,
		 int a)
		{
		return searchSorted (x, 0, x.length, a, Searching.Integer.DEFAULT);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>int[]</TT> for
	 * the given element. It is assumed that the array is sorted in ascending
	 * order; otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchSorted
		(int[] x,
		 int off,
		 int len,
		 int a)
		{
		return searchSorted (x, off, len, a, Searching.Integer.DEFAULT);
		}

	/**
	 * Search the given ordered integer list for the given element. It is
	 * assumed that the list is sorted in ascending order; otherwise, the
	 * <TT>searchSorted()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x  List to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(IntList x,
		 int a)
		{
		return searchSorted (x.item, 0, x.size, a, Searching.Integer.DEFAULT);
		}

	/**
	 * Search the given ordered array of type <TT>int[]</TT> for the given
	 * element. The given helper object is used to compare elements for order
	 * and equality. It is assumed that the array is sorted in the order
	 * determined by the helper object; otherwise, the <TT>searchSorted()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(int[] x,
		 int a,
		 Searching.Integer helper)
		{
		return searchSorted (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>int[]</TT> for
	 * the given element. The given helper object is used to compare elements
	 * for order and equality. It is assumed that the array is sorted in the
	 * order determined by the helper object; otherwise, the
	 * <TT>searchSorted()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchSorted
		(int[] x,
		 int off,
		 int len,
		 int a,
		 Searching.Integer helper)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (x.length == 0) return -1;

		int lo = off;
		int locomp = helper.compare (x[lo], a);
		if (locomp == 0) return lo;
		else if (locomp > 0) return -1;

		int hi = off + len - 1;
		int hicomp = helper.compare (x[hi], a);
		if (hicomp == 0) return hi;
		else if (hicomp < 0) return -1;

		// Loop invariant: x[lo] comes before a; x[hi] comes after a.
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			int midcomp = helper.compare (x[mid], a);
			if (midcomp == 0)
				{
				return mid;
				}
			else if (midcomp < 0)
				{
				lo = mid;
				locomp = midcomp;
				}
			else
				{
				hi = mid;
				hicomp = midcomp;
				}
			}

		return locomp == 0 ? lo : hicomp == 0 ? hi : -1;
		}

	/**
	 * Search the given ordered integer list for the given element. The given
	 * helper object is used to compare elements for order and equality. It is
	 * assumed that the list is sorted in the order determined by the helper
	 * object; otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       List to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(IntList x,
		 int a,
		 Searching.Integer helper)
		{
		return searchSorted (x.item, 0, x.size, a, helper);
		}

	/**
	 * Search the given ordered array of type <TT>int[]</TT> for an interval
	 * containing the given element. It is assumed that the array is sorted in
	 * ascending order; otherwise, the <TT>searchInterval()</TT> method's
	 * behavior is not specified. An <I>O</I>(log <I>n</I>) binary search
	 * algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[0]</TT>, then <TT>i</TT> = 0
	 *     is returned. If <TT>x[x.length-1]</TT> &le; <TT>a</TT>, then
	 *     <TT>i</TT> = <TT>x.length</TT> is returned. If <TT>x</TT> is zero
	 *     length, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(int[] x,
		 int a)
		{
		return searchInterval (x, 0, x.length, a, Searching.Integer.DEFAULT);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>int[]</TT> for
	 * an interval containing the given element. It is assumed that the array is
	 * sorted in ascending order; otherwise, the <TT>searchInterval()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[off]</TT>, then <TT>i</TT> =
	 *     <TT>off</TT> is returned. If <TT>x[off+len-1]</TT> &le; <TT>a</TT>,
	 *     then <TT>i</TT> = <TT>off+len</TT> is returned. If <TT>len</TT> = 0,
	 *     then <TT>i</TT> = &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchInterval
		(int[] x,
		 int off,
		 int len,
		 int a)
		{
		return searchInterval (x, off, len, a, Searching.Integer.DEFAULT);
		}

	/**
	 * Search the given ordered integer list for an interval containing the
	 * given element. It is assumed that the list is sorted in ascending order;
	 * otherwise, the <TT>searchInterval()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x  List to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x.get(i-1)</TT> &le; <TT>a</TT>
	 *     &lt; <TT>x.get(i)</TT>. If <TT>a</TT> &lt; <TT>x.get(0)</TT>, then
	 *     <TT>i</TT> = 0 is returned. If <TT>x.get(x.size()-1)</TT> &le;
	 *     <TT>a</TT>, then <TT>i</TT> = <TT>x.size()</TT> is returned. If
	 *     <TT>x</TT> is empty, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(IntList x,
		 int a)
		{
		return searchInterval (x.item, 0, x.size, a, Searching.Integer.DEFAULT);
		}

	/**
	 * Search the given ordered array of type <TT>int[]</TT> for an interval
	 * containing the given element. The given helper object is used to compare
	 * elements for order and equality. It is assumed that the array is sorted
	 * in the order determined by the helper object; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[0]</TT>, then <TT>i</TT> = 0
	 *     is returned. If <TT>x[x.length-1]</TT> &le; <TT>a</TT>, then
	 *     <TT>i</TT> = <TT>x.length</TT> is returned. If <TT>x</TT> is zero
	 *     length, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(int[] x,
		 int a,
		 Searching.Integer helper)
		{
		return searchInterval (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>int[]</TT> for
	 * an interval containing the given element. The given helper object is used
	 * to compare elements for order and equality. It is assumed that the array
	 * is sorted in the order determined by the helper object; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[off]</TT>, then <TT>i</TT> =
	 *     <TT>off</TT> is returned. If <TT>x[off+len-1]</TT> &le; <TT>a</TT>,
	 *     then <TT>i</TT> = <TT>off+len</TT> is returned. If <TT>len</TT> = 0,
	 *     then <TT>i</TT> = &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchInterval
		(int[] x,
		 int off,
		 int len,
		 int a,
		 Searching.Integer helper)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (len == 0) return -1;

		int lo = off;
		if (helper.compare (x[lo], a) > 0) return off;

		int hi = off + len - 1;
		if (helper.compare (x[hi], a) <= 0) return off + len;

		// Loop invariant: x[lo] <= a and a < x[hi].
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			if (helper.compare (x[mid], a) <= 0)
				lo = mid;
			else
				hi = mid;
			}

		return hi;
		}

	/**
	 * Search the given ordered integer list for an interval containing the
	 * given element. The given helper object is used to compare elements for
	 * order and equality. It is assumed that the array is sorted in the order
	 * determined by the helper object; otherwise, the <TT>searchInterval()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x.get(i-1)</TT> &le; <TT>a</TT>
	 *     &lt; <TT>x.get(i)</TT>. If <TT>a</TT> &lt; <TT>x.get(0)</TT>, then
	 *     <TT>i</TT> = 0 is returned. If <TT>x.get(x.size()-1)</TT> &le;
	 *     <TT>a</TT>, then <TT>i</TT> = <TT>x.size()</TT> is returned. If
	 *     <TT>x</TT> is empty, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(IntList x,
		 int a,
		 Searching.Integer helper)
		{
		return searchInterval (x.item, 0, x.size, a, helper);
		}

	/**
	 * Search the given unordered array of type <TT>long[]</TT> for the given
	 * element. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(long[] x,
		 long a)
		{
		return searchUnsorted (x, 0, x.length, a, Searching.Long.DEFAULT);
		}

	/**
	 * Search a portion of the given unordered array of type <TT>long[]</TT> for
	 * the given element. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchUnsorted
		(long[] x,
		 int off,
		 int len,
		 long a)
		{
		return searchUnsorted (x, off, len, a, Searching.Long.DEFAULT);
		}

	/**
	 * Search the given unordered long integer list for the given element. An
	 * <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x  List to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(LongList x,
		 long a)
		{
		return searchUnsorted (x.item, 0, x.size, a, Searching.Long.DEFAULT);
		}

	/**
	 * Search the given unordered array of type <TT>long[]</TT> for the given
	 * element. The given helper object is used to compare elements for
	 * equality only. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(long[] x,
		 long a,
		 Searching.Long helper)
		{
		return searchUnsorted (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given unordered array of type <TT>long[]</TT> for
	 * the given element. The given helper object is used to compare elements
	 * for equality only. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchUnsorted
		(long[] x,
		 int off,
		 int len,
		 long a,
		 Searching.Long helper)
		{
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		for (int i = 0; i < len; ++ i)
			{
			if (helper.compare (x[i+off], a) == 0) return i;
			}
		return -1;
		}

	/**
	 * Search the given unordered long integer list for the given element. The
	 * given helper object is used to compare elements for equality only. An
	 * <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x       List to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(LongList x,
		 long a,
		 Searching.Long helper)
		{
		return searchUnsorted (x.item, 0, x.size, a, helper);
		}

	/**
	 * Search the given ordered array of type <TT>long[]</TT> for the given
	 * element. It is assumed that the array is sorted in ascending order;
	 * otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(long[] x,
		 long a)
		{
		return searchSorted (x, 0, x.length, a, Searching.Long.DEFAULT);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>long[]</TT> for
	 * the given element. It is assumed that the array is sorted in ascending
	 * order; otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchSorted
		(long[] x,
		 int off,
		 int len,
		 long a)
		{
		return searchSorted (x, off, len, a, Searching.Long.DEFAULT);
		}

	/**
	 * Search the given ordered long integer list for the given element. It is
	 * assumed that the list is sorted in ascending order; otherwise, the
	 * <TT>searchSorted()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x  List to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(LongList x,
		 long a)
		{
		return searchSorted (x.item, 0, x.size, a, Searching.Long.DEFAULT);
		}

	/**
	 * Search the given ordered array of type <TT>long[]</TT> for the given
	 * element. The given helper object is used to compare elements for order
	 * and equality. It is assumed that the array is sorted in the order
	 * determined by the helper object; otherwise, the <TT>searchSorted()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(long[] x,
		 long a,
		 Searching.Long helper)
		{
		return searchSorted (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>long[]</TT> for
	 * the given element. The given helper object is used to compare elements
	 * for order and equality. It is assumed that the array is sorted in the
	 * order determined by the helper object; otherwise, the
	 * <TT>searchSorted()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchSorted
		(long[] x,
		 int off,
		 int len,
		 long a,
		 Searching.Long helper)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (x.length == 0) return -1;

		int lo = off;
		int locomp = helper.compare (x[lo], a);
		if (locomp == 0) return lo;
		else if (locomp > 0) return -1;

		int hi = off + len - 1;
		int hicomp = helper.compare (x[hi], a);
		if (hicomp == 0) return hi;
		else if (hicomp < 0) return -1;

		// Loop invariant: x[lo] comes before a; x[hi] comes after a.
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			int midcomp = helper.compare (x[mid], a);
			if (midcomp == 0)
				{
				return mid;
				}
			else if (midcomp < 0)
				{
				lo = mid;
				locomp = midcomp;
				}
			else
				{
				hi = mid;
				hicomp = midcomp;
				}
			}

		return locomp == 0 ? lo : hicomp == 0 ? hi : -1;
		}

	/**
	 * Search the given ordered long integer list for the given element. The
	 * given helper object is used to compare elements for order and equality.
	 * It is assumed that the list is sorted in the order determined by the
	 * helper object; otherwise, the <TT>searchSorted()</TT> method's behavior
	 * is not specified. An <I>O</I>(log <I>n</I>) binary search algorithm is
	 * used.
	 *
	 * @param  x       List to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(LongList x,
		 long a,
		 Searching.Long helper)
		{
		return searchSorted (x.item, 0, x.size, a, helper);
		}

	/**
	 * Search the given ordered array of type <TT>long[]</TT> for an interval
	 * containing the given element. It is assumed that the array is sorted in
	 * ascending order; otherwise, the <TT>searchInterval()</TT> method's
	 * behavior is not specified. An <I>O</I>(log <I>n</I>) binary search
	 * algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[0]</TT>, then <TT>i</TT> = 0
	 *     is returned. If <TT>x[x.length-1]</TT> &le; <TT>a</TT>, then
	 *     <TT>i</TT> = <TT>x.length</TT> is returned. If <TT>x</TT> is zero
	 *     length, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(long[] x,
		 long a)
		{
		return searchInterval (x, 0, x.length, a, Searching.Long.DEFAULT);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>long[]</TT> for
	 * an interval containing the given element. It is assumed that the array is
	 * sorted in ascending order; otherwise, the <TT>searchInterval()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[off]</TT>, then <TT>i</TT> =
	 *     <TT>off</TT> is returned. If <TT>x[off+len-1]</TT> &le; <TT>a</TT>,
	 *     then <TT>i</TT> = <TT>off+len</TT> is returned. If <TT>len</TT> = 0,
	 *     then <TT>i</TT> = &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchInterval
		(long[] x,
		 int off,
		 int len,
		 long a)
		{
		return searchInterval (x, off, len, a, Searching.Long.DEFAULT);
		}

	/**
	 * Search the given ordered long integer list for an interval containing the
	 * given element. It is assumed that the list is sorted in ascending order;
	 * otherwise, the <TT>searchInterval()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x  List to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x.get(i-1)</TT> &le; <TT>a</TT>
	 *     &lt; <TT>x.get(i)</TT>. If <TT>a</TT> &lt; <TT>x.get(0)</TT>, then
	 *     <TT>i</TT> = 0 is returned. If <TT>x.get(x.size()-1)</TT> &le;
	 *     <TT>a</TT>, then <TT>i</TT> = <TT>x.size()</TT> is returned. If
	 *     <TT>x</TT> is empty, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(LongList x,
		 long a)
		{
		return searchInterval (x.item, 0, x.size, a, Searching.Long.DEFAULT);
		}

	/**
	 * Search the given ordered array of type <TT>long[]</TT> for an interval
	 * containing the given element. The given helper object is used to compare
	 * elements for order and equality. It is assumed that the array is sorted
	 * in the order determined by the helper object; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[0]</TT>, then <TT>i</TT> = 0
	 *     is returned. If <TT>x[x.length-1]</TT> &le; <TT>a</TT>, then
	 *     <TT>i</TT> = <TT>x.length</TT> is returned. If <TT>x</TT> is zero
	 *     length, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(long[] x,
		 long a,
		 Searching.Long helper)
		{
		return searchInterval (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>long[]</TT> for
	 * an interval containing the given element. The given helper object is used
	 * to compare elements for order and equality. It is assumed that the array
	 * is sorted in the order determined by the helper object; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[off]</TT>, then <TT>i</TT> =
	 *     <TT>off</TT> is returned. If <TT>x[off+len-1]</TT> &le; <TT>a</TT>,
	 *     then <TT>i</TT> = <TT>off+len</TT> is returned. If <TT>len</TT> = 0,
	 *     then <TT>i</TT> = &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchInterval
		(long[] x,
		 int off,
		 int len,
		 long a,
		 Searching.Long helper)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (len == 0) return -1;

		int lo = off;
		if (helper.compare (x[lo], a) > 0) return off;

		int hi = off + len - 1;
		if (helper.compare (x[hi], a) <= 0) return off + len;

		// Loop invariant: x[lo] <= a and a < x[hi].
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			if (helper.compare (x[mid], a) <= 0)
				lo = mid;
			else
				hi = mid;
			}

		return hi;
		}

	/**
	 * Search the given ordered long integer list for an interval containing the
	 * given element. The given helper object is used to compare elements for
	 * order and equality. It is assumed that the array is sorted in the order
	 * determined by the helper object; otherwise, the <TT>searchInterval()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x.get(i-1)</TT> &le; <TT>a</TT>
	 *     &lt; <TT>x.get(i)</TT>. If <TT>a</TT> &lt; <TT>x.get(0)</TT>, then
	 *     <TT>i</TT> = 0 is returned. If <TT>x.get(x.size()-1)</TT> &le;
	 *     <TT>a</TT>, then <TT>i</TT> = <TT>x.size()</TT> is returned. If
	 *     <TT>x</TT> is empty, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(LongList x,
		 long a,
		 Searching.Long helper)
		{
		return searchInterval (x.item, 0, x.size, a, helper);
		}

	/**
	 * Search the given unordered array of type <TT>float[]</TT> for the given
	 * element. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(float[] x,
		 float a)
		{
		return searchUnsorted (x, 0, x.length, a, Searching.Float.DEFAULT);
		}

	/**
	 * Search a portion of the given unordered array of type <TT>float[]</TT>
	 * for the given element. An <I>O</I>(<I>n</I>) linear search algorithm is
	 * used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchUnsorted
		(float[] x,
		 int off,
		 int len,
		 float a)
		{
		return searchUnsorted (x, off, len, a, Searching.Float.DEFAULT);
		}

	/**
	 * Search the given unordered float list for the given element. An
	 * <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x  List to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(FloatList x,
		 float a)
		{
		return searchUnsorted (x.item, 0, x.size, a, Searching.Float.DEFAULT);
		}

	/**
	 * Search the given unordered array of type <TT>float[]</TT> for the given
	 * element. The given helper object is used to compare elements for
	 * equality only. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(float[] x,
		 float a,
		 Searching.Float helper)
		{
		return searchUnsorted (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given unordered array of type <TT>float[]</TT>
	 * for the given element. The given helper object is used to compare
	 * elements for equality only. An <I>O</I>(<I>n</I>) linear search algorithm
	 * is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchUnsorted
		(float[] x,
		 int off,
		 int len,
		 float a,
		 Searching.Float helper)
		{
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		for (int i = 0; i < len; ++ i)
			{
			if (helper.compare (x[i+off], a) == 0) return i;
			}
		return -1;
		}

	/**
	 * Search the given unordered float list for the given element. The given
	 * helper object is used to compare elements for equality only. An
	 * <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x       List to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(FloatList x,
		 float a,
		 Searching.Float helper)
		{
		return searchUnsorted (x.item, 0, x.size, a, helper);
		}

	/**
	 * Search the given ordered array of type <TT>float[]</TT> for the given
	 * element. It is assumed that the array is sorted in ascending order;
	 * otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(float[] x,
		 float a)
		{
		return searchSorted (x, 0, x.length, a, Searching.Float.DEFAULT);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>float[]</TT> for
	 * the given element. It is assumed that the array is sorted in ascending
	 * order; otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchSorted
		(float[] x,
		 int off,
		 int len,
		 float a)
		{
		return searchSorted (x, off, len, a, Searching.Float.DEFAULT);
		}

	/**
	 * Search the given ordered float list for the given element. It is assumed
	 * that the list is sorted in ascending order; otherwise, the
	 * <TT>searchSorted()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x  List to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(FloatList x,
		 float a)
		{
		return searchSorted (x.item, 0, x.size, a, Searching.Float.DEFAULT);
		}

	/**
	 * Search the given ordered array of type <TT>float[]</TT> for the given
	 * element. The given helper object is used to compare elements for order
	 * and equality. It is assumed that the array is sorted in the order
	 * determined by the helper object; otherwise, the <TT>searchSorted()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(float[] x,
		 float a,
		 Searching.Float helper)
		{
		return searchSorted (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>float[]</TT> for
	 * the given element. The given helper object is used to compare elements
	 * for order and equality. It is assumed that the array is sorted in the
	 * order determined by the helper object; otherwise, the
	 * <TT>searchSorted()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchSorted
		(float[] x,
		 int off,
		 int len,
		 float a,
		 Searching.Float helper)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (x.length == 0) return -1;

		int lo = off;
		int locomp = helper.compare (x[lo], a);
		if (locomp == 0) return lo;
		else if (locomp > 0) return -1;

		int hi = off + len - 1;
		int hicomp = helper.compare (x[hi], a);
		if (hicomp == 0) return hi;
		else if (hicomp < 0) return -1;

		// Loop invariant: x[lo] comes before a; x[hi] comes after a.
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			int midcomp = helper.compare (x[mid], a);
			if (midcomp == 0)
				{
				return mid;
				}
			else if (midcomp < 0)
				{
				lo = mid;
				locomp = midcomp;
				}
			else
				{
				hi = mid;
				hicomp = midcomp;
				}
			}

		return locomp == 0 ? lo : hicomp == 0 ? hi : -1;
		}

	/**
	 * Search the given ordered float list for the given element. The given
	 * helper object is used to compare elements for order and equality. It is
	 * assumed that the list is sorted in the order determined by the helper
	 * object; otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       List to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(FloatList x,
		 float a,
		 Searching.Float helper)
		{
		return searchSorted (x.item, 0, x.size, a, helper);
		}

	/**
	 * Search the given ordered array of type <TT>float[]</TT> for an interval
	 * containing the given element. It is assumed that the array is sorted in
	 * ascending order; otherwise, the <TT>searchInterval()</TT> method's
	 * behavior is not specified. An <I>O</I>(log <I>n</I>) binary search
	 * algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[0]</TT>, then <TT>i</TT> = 0
	 *     is returned. If <TT>x[x.length-1]</TT> &le; <TT>a</TT>, then
	 *     <TT>i</TT> = <TT>x.length</TT> is returned. If <TT>x</TT> is zero
	 *     length, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(float[] x,
		 float a)
		{
		return searchInterval (x, 0, x.length, a, Searching.Float.DEFAULT);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>float[]</TT> for
	 * an interval containing the given element. It is assumed that the array is
	 * sorted in ascending order; otherwise, the <TT>searchInterval()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[off]</TT>, then <TT>i</TT> =
	 *     <TT>off</TT> is returned. If <TT>x[off+len-1]</TT> &le; <TT>a</TT>,
	 *     then <TT>i</TT> = <TT>off+len</TT> is returned. If <TT>len</TT> = 0,
	 *     then <TT>i</TT> = &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchInterval
		(float[] x,
		 int off,
		 int len,
		 float a)
		{
		return searchInterval (x, off, len, a, Searching.Float.DEFAULT);
		}

	/**
	 * Search the given ordered float list for an interval containing the
	 * given element. It is assumed that the list is sorted in ascending order;
	 * otherwise, the <TT>searchInterval()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x  List to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x.get(i-1)</TT> &le; <TT>a</TT>
	 *     &lt; <TT>x.get(i)</TT>. If <TT>a</TT> &lt; <TT>x.get(0)</TT>, then
	 *     <TT>i</TT> = 0 is returned. If <TT>x.get(x.size()-1)</TT> &le;
	 *     <TT>a</TT>, then <TT>i</TT> = <TT>x.size()</TT> is returned. If
	 *     <TT>x</TT> is empty, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(FloatList x,
		 float a)
		{
		return searchInterval (x.item, 0, x.size, a, Searching.Float.DEFAULT);
		}

	/**
	 * Search the given ordered array of type <TT>float[]</TT> for an interval
	 * containing the given element. The given helper object is used to compare
	 * elements for order and equality. It is assumed that the array is sorted
	 * in the order determined by the helper object; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[0]</TT>, then <TT>i</TT> = 0
	 *     is returned. If <TT>x[x.length-1]</TT> &le; <TT>a</TT>, then
	 *     <TT>i</TT> = <TT>x.length</TT> is returned. If <TT>x</TT> is zero
	 *     length, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(float[] x,
		 float a,
		 Searching.Float helper)
		{
		return searchInterval (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>float[]</TT> for
	 * an interval containing the given element. The given helper object is used
	 * to compare elements for order and equality. It is assumed that the array
	 * is sorted in the order determined by the helper object; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[off]</TT>, then <TT>i</TT> =
	 *     <TT>off</TT> is returned. If <TT>x[off+len-1]</TT> &le; <TT>a</TT>,
	 *     then <TT>i</TT> = <TT>off+len</TT> is returned. If <TT>len</TT> = 0,
	 *     then <TT>i</TT> = &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchInterval
		(float[] x,
		 int off,
		 int len,
		 float a,
		 Searching.Float helper)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (len == 0) return -1;

		int lo = off;
		if (helper.compare (x[lo], a) > 0) return off;

		int hi = off + len - 1;
		if (helper.compare (x[hi], a) <= 0) return off + len;

		// Loop invariant: x[lo] <= a and a < x[hi].
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			if (helper.compare (x[mid], a) <= 0)
				lo = mid;
			else
				hi = mid;
			}

		return hi;
		}

	/**
	 * Search the given ordered float list for an interval containing the
	 * given element. The given helper object is used to compare elements for
	 * order and equality. It is assumed that the array is sorted in the order
	 * determined by the helper object; otherwise, the <TT>searchInterval()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x.get(i-1)</TT> &le; <TT>a</TT>
	 *     &lt; <TT>x.get(i)</TT>. If <TT>a</TT> &lt; <TT>x.get(0)</TT>, then
	 *     <TT>i</TT> = 0 is returned. If <TT>x.get(x.size()-1)</TT> &le;
	 *     <TT>a</TT>, then <TT>i</TT> = <TT>x.size()</TT> is returned. If
	 *     <TT>x</TT> is empty, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(FloatList x,
		 float a,
		 Searching.Float helper)
		{
		return searchInterval (x.item, 0, x.size, a, helper);
		}

	/**
	 * Search the given unordered array of type <TT>double[]</TT> for the given
	 * element. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(double[] x,
		 double a)
		{
		return searchUnsorted (x, 0, x.length, a, Searching.Double.DEFAULT);
		}

	/**
	 * Search a portion of the given unordered array of type <TT>double[]</TT>
	 * for the given element. An <I>O</I>(<I>n</I>) linear search algorithm is
	 * used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchUnsorted
		(double[] x,
		 int off,
		 int len,
		 double a)
		{
		return searchUnsorted (x, off, len, a, Searching.Double.DEFAULT);
		}

	/**
	 * Search the given unordered double list for the given element. An
	 * <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x  List to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(DoubleList x,
		 double a)
		{
		return searchUnsorted (x.item, 0, x.size, a, Searching.Double.DEFAULT);
		}

	/**
	 * Search the given unordered array of type <TT>double[]</TT> for the given
	 * element. The given helper object is used to compare elements for
	 * equality only. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(double[] x,
		 double a,
		 Searching.Double helper)
		{
		return searchUnsorted (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given unordered array of type <TT>double[]</TT>
	 * for the given element. The given helper object is used to compare
	 * elements for equality only. An <I>O</I>(<I>n</I>) linear search algorithm
	 * is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchUnsorted
		(double[] x,
		 int off,
		 int len,
		 double a,
		 Searching.Double helper)
		{
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		for (int i = 0; i < len; ++ i)
			{
			if (helper.compare (x[i+off], a) == 0) return i;
			}
		return -1;
		}

	/**
	 * Search the given unordered double list for the given element. The given
	 * helper object is used to compare elements for equality only. An
	 * <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  x       List to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchUnsorted
		(DoubleList x,
		 double a,
		 Searching.Double helper)
		{
		return searchUnsorted (x.item, 0, x.size, a, helper);
		}

	/**
	 * Search the given ordered array of type <TT>double[]</TT> for the given
	 * element. It is assumed that the array is sorted in ascending order;
	 * otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(double[] x,
		 double a)
		{
		return searchSorted (x, 0, x.length, a, Searching.Double.DEFAULT);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>double[]</TT> for
	 * the given element. It is assumed that the array is sorted in ascending
	 * order; otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchSorted
		(double[] x,
		 int off,
		 int len,
		 double a)
		{
		return searchSorted (x, off, len, a, Searching.Double.DEFAULT);
		}

	/**
	 * Search the given ordered double list for the given element. It is
	 * assumed that the list is sorted in ascending order; otherwise, the
	 * <TT>searchSorted()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x  List to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(DoubleList x,
		 double a)
		{
		return searchSorted (x.item, 0, x.size, a, Searching.Double.DEFAULT);
		}

	/**
	 * Search the given ordered array of type <TT>double[]</TT> for the given
	 * element. The given helper object is used to compare elements for order
	 * and equality. It is assumed that the array is sorted in the order
	 * determined by the helper object; otherwise, the <TT>searchSorted()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(double[] x,
		 double a,
		 Searching.Double helper)
		{
		return searchSorted (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>double[]</TT> for
	 * the given element. The given helper object is used to compare elements
	 * for order and equality. It is assumed that the array is sorted in the
	 * order determined by the helper object; otherwise, the
	 * <TT>searchSorted()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchSorted
		(double[] x,
		 int off,
		 int len,
		 double a,
		 Searching.Double helper)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (x.length == 0) return -1;

		int lo = off;
		int locomp = helper.compare (x[lo], a);
		if (locomp == 0) return lo;
		else if (locomp > 0) return -1;

		int hi = off + len - 1;
		int hicomp = helper.compare (x[hi], a);
		if (hicomp == 0) return hi;
		else if (hicomp < 0) return -1;

		// Loop invariant: x[lo] comes before a; x[hi] comes after a.
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			int midcomp = helper.compare (x[mid], a);
			if (midcomp == 0)
				{
				return mid;
				}
			else if (midcomp < 0)
				{
				lo = mid;
				locomp = midcomp;
				}
			else
				{
				hi = mid;
				hicomp = midcomp;
				}
			}

		return locomp == 0 ? lo : hicomp == 0 ? hi : -1;
		}

	/**
	 * Search the given ordered double list for the given element. The given
	 * helper object is used to compare elements for order and equality. It is
	 * assumed that the list is sorted in the order determined by the helper
	 * object; otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       List to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static int searchSorted
		(DoubleList x,
		 double a,
		 Searching.Double helper)
		{
		return searchSorted (x.item, 0, x.size, a, helper);
		}

	/**
	 * Search the given ordered array of type <TT>double[]</TT> for an interval
	 * containing the given element. It is assumed that the array is sorted in
	 * ascending order; otherwise, the <TT>searchInterval()</TT> method's
	 * behavior is not specified. An <I>O</I>(log <I>n</I>) binary search
	 * algorithm is used.
	 *
	 * @param  x  Array to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[0]</TT>, then <TT>i</TT> = 0
	 *     is returned. If <TT>x[x.length-1]</TT> &le; <TT>a</TT>, then
	 *     <TT>i</TT> = <TT>x.length</TT> is returned. If <TT>x</TT> is zero
	 *     length, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(double[] x,
		 double a)
		{
		return searchInterval (x, 0, x.length, a, Searching.Double.DEFAULT);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>double[]</TT> for
	 * an interval containing the given element. It is assumed that the array is
	 * sorted in ascending order; otherwise, the <TT>searchInterval()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[off]</TT>, then <TT>i</TT> =
	 *     <TT>off</TT> is returned. If <TT>x[off+len-1]</TT> &le; <TT>a</TT>,
	 *     then <TT>i</TT> = <TT>off+len</TT> is returned. If <TT>len</TT> = 0,
	 *     then <TT>i</TT> = &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchInterval
		(double[] x,
		 int off,
		 int len,
		 double a)
		{
		return searchInterval (x, off, len, a, Searching.Double.DEFAULT);
		}

	/**
	 * Search the given ordered double list for an interval containing the
	 * given element. It is assumed that the list is sorted in ascending order;
	 * otherwise, the <TT>searchInterval()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x  List to be searched.
	 * @param  a  Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x.get(i-1)</TT> &le; <TT>a</TT>
	 *     &lt; <TT>x.get(i)</TT>. If <TT>a</TT> &lt; <TT>x.get(0)</TT>, then
	 *     <TT>i</TT> = 0 is returned. If <TT>x.get(x.size()-1)</TT> &le;
	 *     <TT>a</TT>, then <TT>i</TT> = <TT>x.size()</TT> is returned. If
	 *     <TT>x</TT> is empty, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(DoubleList x,
		 double a)
		{
		return searchInterval (x.item, 0, x.size, a, Searching.Double.DEFAULT);
		}

	/**
	 * Search the given ordered array of type <TT>double[]</TT> for an interval
	 * containing the given element. The given helper object is used to compare
	 * elements for order and equality. It is assumed that the array is sorted
	 * in the order determined by the helper object; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[0]</TT>, then <TT>i</TT> = 0
	 *     is returned. If <TT>x[x.length-1]</TT> &le; <TT>a</TT>, then
	 *     <TT>i</TT> = <TT>x.length</TT> is returned. If <TT>x</TT> is zero
	 *     length, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(double[] x,
		 double a,
		 Searching.Double helper)
		{
		return searchInterval (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>double[]</TT> for
	 * an interval containing the given element. The given helper object is used
	 * to compare elements for order and equality. It is assumed that the array
	 * is sorted in the order determined by the helper object; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[off]</TT>, then <TT>i</TT> =
	 *     <TT>off</TT> is returned. If <TT>x[off+len-1]</TT> &le; <TT>a</TT>,
	 *     then <TT>i</TT> = <TT>off+len</TT> is returned. If <TT>len</TT> = 0,
	 *     then <TT>i</TT> = &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int searchInterval
		(double[] x,
		 int off,
		 int len,
		 double a,
		 Searching.Double helper)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (len == 0) return -1;

		int lo = off;
		if (helper.compare (x[lo], a) > 0) return off;

		int hi = off + len - 1;
		if (helper.compare (x[hi], a) <= 0) return off + len;

		// Loop invariant: x[lo] <= a and a < x[hi].
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			if (helper.compare (x[mid], a) <= 0)
				lo = mid;
			else
				hi = mid;
			}

		return hi;
		}

	/**
	 * Search the given ordered double list for an interval containing the
	 * given element. The given helper object is used to compare elements for
	 * order and equality. It is assumed that the array is sorted in the order
	 * determined by the helper object; otherwise, the <TT>searchInterval()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x.get(i-1)</TT> &le; <TT>a</TT>
	 *     &lt; <TT>x.get(i)</TT>. If <TT>a</TT> &lt; <TT>x.get(0)</TT>, then
	 *     <TT>i</TT> = 0 is returned. If <TT>x.get(x.size()-1)</TT> &le;
	 *     <TT>a</TT>, then <TT>i</TT> = <TT>x.size()</TT> is returned. If
	 *     <TT>x</TT> is empty, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static int searchInterval
		(DoubleList x,
		 double a,
		 Searching.Double helper)
		{
		return searchInterval (x.item, 0, x.size, a, helper);
		}

	/**
	 * Search the given unordered array of type {@linkplain java.lang.Comparable
	 * Comparable}<TT>&lt;T&gt;[]</TT> for the given element. The element's
	 * <TT>compareTo()</TT> method is used to compare elements for equality
	 * only. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  <T>  Array element data type.
	 * @param  x    Array to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 */
	public static <T extends java.lang.Comparable<T>> int searchUnsorted
		(T[] x,
		 T a)
		{
		return searchUnsorted (x, 0, x.length, a,
			new Searching.Comparable<T>());
		}

	/**
	 * Search a portion of the given unordered array of type {@linkplain
	 * java.lang.Comparable Comparable}<TT>&lt;T&gt;[]</TT> for the given
	 * element. The element's <TT>compareTo()</TT> method is used to compare
	 * elements for equality only. An <I>O</I>(<I>n</I>) linear search algorithm
	 * is used.
	 *
	 * @param  <T>  Array element data type.
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static <T extends java.lang.Comparable<T>> int searchUnsorted
		(T[] x,
		 int off,
		 int len,
		 T a)
		{
		return searchUnsorted (x, off, len, a, new Searching.Comparable<T>());
		}

	/**
	 * Search the given unordered comparable object list for the given element.
	 * The element's <TT>compareTo()</TT> method is used to compare elements for
	 * equality only. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  <T>  List element data type.
	 * @param  x    List to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>,
	 *          then the index of that element is returned. Otherwise, &minus;1
	 *          is returned.
	 */
	public static <T extends java.lang.Comparable<T>> int searchUnsorted
		(AList<T> x,
		 T a)
		{
		return searchUnsorted (x.item, 0, x.size, a,
			new Searching.Comparable<T>());
		}

	/**
	 * Search the given unordered array of type <TT>T[]</TT> for the given
	 * element. The given helper object is used to compare elements for
	 * equality only. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  <T>     Array element data type.
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static <T> int searchUnsorted
		(T[] x,
		 T a,
		 Searching.Object<T> helper)
		{
		return searchUnsorted (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given unordered array of type <TT>T[]</TT> for
	 * the given element. The given helper object is used to compare elements
	 * for equality only. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  <T>     Array element data type.
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static <T> int searchUnsorted
		(T[] x,
		 int off,
		 int len,
		 T a,
		 Searching.Object<T> helper)
		{
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		for (int i = 0; i < len; ++ i)
			{
			if (helper.compare (x[off+i], a) == 0) return i;
			}
		return -1;
		}

	/**
	 * Search the given unordered object list for the given element. The given
	 * helper object is used to compare elements for equality only. An
	 * <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  <T>     List element data type.
	 * @param  x       List to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static <T> int searchUnsorted
		(AList<T> x,
		 T a,
		 Searching.Object<T> helper)
		{
		return searchUnsorted (x.item, 0, x.size, a, helper);
		}

	/**
	 * Search the given unordered array of type <TT>T[]</TT> for the given
	 * element. The given comparator is used to compare elements for equality
	 * only. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  <T>   Array element data type.
	 * @param  x     Array to be searched.
	 * @param  a     Element to be searched for.
	 * @param  comp  Comparator.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static <T> int searchUnsorted
		(T[] x,
		 T a,
		 Comparator<T> comp)
		{
		return searchUnsorted (x, 0, x.length, a, comp);
		}

	/**
	 * Search a portion of the given unordered array of type <TT>T[]</TT> for
	 * the given element. The given comparator is used to compare elements for
	 * equality only. An <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  <T>   Array element data type.
	 * @param  x     Array to be searched.
	 * @param  off   Index of first element to be searched.
	 * @param  len   Number of elements to be searched.
	 * @param  a     Element to be searched for.
	 * @param  comp  Comparator.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static <T> int searchUnsorted
		(T[] x,
		 int off,
		 int len,
		 T a,
		 Comparator<T> comp)
		{
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		for (int i = 0; i < len; ++ i)
			{
			if (comp.compare (x[off+i], a) == 0) return i;
			}
		return -1;
		}

	/**
	 * Search the given unordered object list for the given element. The given
	 * comparator is used to compare elements for equality only. An
	 * <I>O</I>(<I>n</I>) linear search algorithm is used.
	 *
	 * @param  <T>   List element data type.
	 * @param  x     List to be searched.
	 * @param  a     Element to be searched for.
	 * @param  comp  Comparator.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static <T> int searchUnsorted
		(AList<T> x,
		 T a,
		 Comparator<T> comp)
		{
		return searchUnsorted (x.item, 0, x.size, a, comp);
		}

	/**
	 * Search the given ordered array of type {@linkplain java.lang.Comparable
	 * Comparable}<TT>&lt;T&gt;[]</TT> for the given element. The element's
	 * <TT>compareTo()</TT> method is used to compare elements for equality and
	 * ordering. It is assumed that the array is sorted in the order determined
	 * by the element's <TT>compareTo()</TT> method; otherwise, the
	 * <TT>searchSorted()</TT> method's behavior is not specified. An
	 * <I>O</I>(log&nbsp;<I>n</I>) binary search algorithm is used.
	 *
	 * @param  <T>  Array element data type.
	 * @param  x    Array to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static <T extends java.lang.Comparable<T>> int searchSorted
		(T[] x,
		 T a)
		{
		return searchSorted (x, 0, x.length, a, new Searching.Comparable<T>());
		}

	/**
	 * Search a portion of the given ordered array of type {@linkplain
	 * java.lang.Comparable Comparable}<TT>&lt;T&gt;[]</TT> for the given
	 * element. The element's <TT>compareTo()</TT> method is used to compare
	 * elements for equality and ordering. It is assumed that the array is
	 * sorted in the order determined by the element's <TT>compareTo()</TT>
	 * method; otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log&nbsp;<I>n</I>) binary search algorithm is
	 * used.
	 *
	 * @param  <T>  Array element data type.
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static <T extends java.lang.Comparable<T>> int searchSorted
		(T[] x,
		 int off,
		 int len,
		 T a)
		{
		return searchSorted (x, off, len, a, new Searching.Comparable<T>());
		}

	/**
	 * Search the given ordered comparable object list for the given element.
	 * The element's <TT>compareTo()</TT> method is used to compare elements for
	 * equality and ordering. It is assumed that the list is sorted in the order
	 * determined by the element's <TT>compareTo()</TT> method; otherwise, the
	 * <TT>searchSorted()</TT> method's behavior is not specified. An
	 * <I>O</I>(log&nbsp;<I>n</I>) binary search algorithm is used.
	 *
	 * @param  <T>  List element data type.
	 * @param  x    List to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static <T extends java.lang.Comparable<T>> int searchSorted
		(AList<T> x,
		 T a)
		{
		return searchSorted (x.item, 0, x.size, a,
			new Searching.Comparable<T>());
		}

	/**
	 * Search the given ordered array of type <TT>T[]</TT> for the given
	 * element. The given helper object is used to compare elements for equality
	 * and ordering. It is assumed that the array is sorted in the order
	 * determined by the helper object; otherwise, the <TT>searchSorted()</TT>
	 * method's behavior is not specified. An <I>O</I>(log&nbsp;<I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  <T>     Array element data type.
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static <T> int searchSorted
		(T[] x,
		 T a,
		 Searching.Object<T> helper)
		{
		return searchSorted (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>T[]</TT> for the
	 * given element. The given helper object is used to compare elements for
	 * equality and ordering. It is assumed that the array is sorted in the
	 * order determined by the helper object; otherwise, the
	 * <TT>searchSorted()</TT> method's behavior is not specified. An
	 * <I>O</I>(log&nbsp;<I>n</I>) binary search algorithm is used.
	 *
	 * @param  <T>     Array element data type.
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static <T> int searchSorted
		(T[] x,
		 int off,
		 int len,
		 T a,
		 Searching.Object<T> helper)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (len == 0) return -1;

		int lo = off;
		int locomp = helper.compare (x[lo], a);
		if (locomp == 0) return lo;
		else if (locomp > 0) return -1;

		int hi = off + len - 1;
		int hicomp = helper.compare (x[hi], a);
		if (hicomp == 0) return hi;
		else if (hicomp < 0) return -1;

		// Loop invariant: x[lo] comes before a; x[hi] comes after a.
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			int midcomp = helper.compare (x[mid], a);
			if (midcomp == 0)
				{
				return mid;
				}
			else if (midcomp < 0)
				{
				lo = mid;
				locomp = midcomp;
				}
			else
				{
				hi = mid;
				hicomp = midcomp;
				}
			}

		return locomp == 0 ? lo : hicomp == 0 ? hi : -1;
		}

	/**
	 * Search the given ordered object list for the given element. The given
	 * helper object is used to compare elements for equality and ordering. It
	 * is assumed that the list is sorted in the order determined by the helper
	 * object; otherwise, the <TT>searchSorted()</TT> method's behavior is not
	 * specified. An <I>O</I>(log&nbsp;<I>n</I>) binary search algorithm is
	 * used.
	 *
	 * @param  <T>     List element data type.
	 * @param  x       List to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static <T> int searchSorted
		(AList<T> x,
		 T a,
		 Searching.Object<T> helper)
		{
		return searchSorted (x.item, 0, x.size, a, helper);
		}

	/**
	 * Search the given ordered array of type <TT>T[]</TT> for the given
	 * element. The given comparator is used to compare elements for equality
	 * and ordering. It is assumed that the array is sorted in the order
	 * determined by the comparator; otherwise, the <TT>searchSorted()</TT>
	 * method's behavior is not specified. An <I>O</I>(log&nbsp;<I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  <T>   Array element data type.
	 * @param  x     Array to be searched.
	 * @param  a     Element to be searched for.
	 * @param  comp  Comparator.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static <T> int searchSorted
		(T[] x,
		 T a,
		 Comparator<T> comp)
		{
		return searchSorted (x, 0, x.length, a, comp);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>T[]</TT> for the
	 * given element. The given comparator is used to compare elements for
	 * equality and ordering. It is assumed that the array is sorted in the
	 * order determined by the comparator; otherwise, the
	 * <TT>searchSorted()</TT> method's behavior is not specified. An
	 * <I>O</I>(log&nbsp;<I>n</I>) binary search algorithm is used.
	 *
	 * @param  <T>   Array element data type.
	 * @param  x     Array to be searched.
	 * @param  off   Index of first element to be searched.
	 * @param  len   Number of elements to be searched.
	 * @param  a     Element to be searched for.
	 * @param  comp  Comparator.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x[off]</TT>
	 *          through <TT>x[off+len-1]</TT>, then the index of that element is
	 *          returned. Otherwise, &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static <T> int searchSorted
		(T[] x,
		 int off,
		 int len,
		 T a,
		 Comparator<T> comp)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (len == 0) return -1;

		int lo = off;
		int locomp = comp.compare (x[lo], a);
		if (locomp == 0) return lo;
		else if (locomp > 0) return -1;

		int hi = off + len - 1;
		int hicomp = comp.compare (x[hi], a);
		if (hicomp == 0) return hi;
		else if (hicomp < 0) return -1;

		// Loop invariant: x[lo] comes before a; x[hi] comes after a.
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			int midcomp = comp.compare (x[mid], a);
			if (midcomp == 0)
				{
				return mid;
				}
			else if (midcomp < 0)
				{
				lo = mid;
				locomp = midcomp;
				}
			else
				{
				hi = mid;
				hicomp = midcomp;
				}
			}

		return locomp == 0 ? lo : hicomp == 0 ? hi : -1;
		}

	/**
	 * Search the given ordered object list for the given element. The given
	 * comparator is used to compare elements for equality and ordering. It is
	 * assumed that the list is sorted in the order determined by the
	 * comparator; otherwise, the <TT>searchSorted()</TT> method's behavior is
	 * not specified. An <I>O</I>(log&nbsp;<I>n</I>) binary search algorithm is
	 * used.
	 *
	 * @param  <T>   List element data type.
	 * @param  x     List to be searched.
	 * @param  a     Element to be searched for.
	 * @param  comp  Comparator.
	 *
	 * @return  If an element the same as <TT>a</TT> exists in <TT>x</TT>, then
	 *          the index of that element is returned. Otherwise, &minus;1 is
	 *          returned.
	 */
	public static <T> int searchSorted
		(AList<T> x,
		 T a,
		 Comparator<T> comp)
		{
		return searchSorted (x.item, 0, x.size, a, comp);
		}

	/**
	 * Search the given ordered array of type {@linkplain java.lang.Comparable
	 * Comparable}<TT>&lt;T&gt;[]</TT> for an interval containing the given
	 * element. The element's <TT>compareTo()</TT> method is used to compare
	 * elements for order and equality. It is assumed that the array is sorted
	 * in the order determined by the element's <TT>compareTo()</TT> method;
	 * otherwise, the <TT>searchInterval()</TT> method's behavior is not
	 * specified. An <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  <T>  Array element data type.
	 * @param  x    Array to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[0]</TT>, then <TT>i</TT> = 0
	 *     is returned. If <TT>x[x.length-1]</TT> &le; <TT>a</TT>, then
	 *     <TT>i</TT> = <TT>x.length</TT> is returned. If <TT>x</TT> is zero
	 *     length, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static <T extends java.lang.Comparable<T>> int searchInterval
		(T[] x,
		 T a)
		{
		return searchInterval (x, 0, x.length, a,
			new Searching.Comparable<T>());
		}

	/**
	 * Search a portion of the given ordered array of type {@linkplain
	 * java.lang.Comparable Comparable}<TT>&lt;T&gt;[]</TT> for an interval
	 * containing the given element. The element's <TT>compareTo()</TT> method
	 * is used to compare elements for order and equality. It is assumed that
	 * the array is sorted in the order determined by the element's
	 * <TT>compareTo()</TT> method; otherwise, the <TT>searchInterval()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  <T>  Array element data type.
	 * @param  x    Array to be searched.
	 * @param  off  Index of first element to be searched.
	 * @param  len  Number of elements to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[off]</TT>, then <TT>i</TT> =
	 *     <TT>off</TT> is returned. If <TT>x[off+len-1]</TT> &le; <TT>a</TT>,
	 *     then <TT>i</TT> = <TT>off+len</TT> is returned. If <TT>len</TT> = 0,
	 *     then <TT>i</TT> = &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static <T extends java.lang.Comparable<T>> int searchInterval
		(T[] x,
		 int off,
		 int len,
		 T a)
		{
		return searchInterval (x, off, len, a, new Searching.Comparable<T>());
		}

	/**
	 * Search the given ordered comparable object list for an interval
	 * containing the given element. The element's <TT>compareTo()</TT> method
	 * is used to compare elements for order and equality. It is assumed that
	 * the list is sorted in the order determined by the element's
	 * <TT>compareTo()</TT> method; otherwise, the <TT>searchInterval()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  <T>  List element data type.
	 * @param  x    List to be searched.
	 * @param  a    Element to be searched for.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x.get(i-1)</TT> &le; <TT>a</TT>
	 *     &lt; <TT>x.get(i)</TT>. If <TT>a</TT> &lt; <TT>x.get(0)</TT>, then
	 *     <TT>i</TT> = 0 is returned. If <TT>x.get(x.size()-1)</TT> &le;
	 *     <TT>a</TT>, then <TT>i</TT> = <TT>x.size()</TT> is returned. If
	 *     <TT>x</TT> is empty, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static <T extends java.lang.Comparable<T>> int searchInterval
		(AList<T> x,
		 T a)
		{
		return searchInterval (x.item, 0, x.size, a,
			new Searching.Comparable<T>());
		}

	/**
	 * Search the given ordered array of type <TT>T[]</TT> for an interval
	 * containing the given element. The given helper object is used to compare
	 * elements for order and equality. It is assumed that the array is sorted
	 * in the order determined by the helper object; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  <T>     Array element data type.
	 * @param  x       Array to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[0]</TT>, then <TT>i</TT> = 0
	 *     is returned. If <TT>x[x.length-1]</TT> &le; <TT>a</TT>, then
	 *     <TT>i</TT> = <TT>x.length</TT> is returned. If <TT>x</TT> is zero
	 *     length, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static <T> int searchInterval
		(T[] x,
		 T a,
		 Searching.Object<T> helper)
		{
		return searchInterval (x, 0, x.length, a, helper);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>T[]</TT> for an
	 * interval containing the given element. The given helper object is used to
	 * compare elements for order and equality. It is assumed that the array is
	 * sorted in the order determined by the helper object; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  <T>     Array element data type.
	 * @param  x       Array to be searched.
	 * @param  off     Index of first element to be searched.
	 * @param  len     Number of elements to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[off]</TT>, then <TT>i</TT> =
	 *     <TT>off</TT> is returned. If <TT>x[off+len-1]</TT> &le; <TT>a</TT>,
	 *     then <TT>i</TT> = <TT>off+len</TT> is returned. If <TT>len</TT> = 0,
	 *     then <TT>i</TT> = &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static <T> int searchInterval
		(T[] x,
		 int off,
		 int len,
		 T a,
		 Searching.Object<T> helper)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (len == 0) return -1;

		int lo = off;
		if (helper.compare (x[lo], a) > 0) return off;

		int hi = off + len - 1;
		if (helper.compare (x[hi], a) <= 0) return off + len;

		// Loop invariant: x[lo] <= a and a < x[hi].
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			if (helper.compare (x[mid], a) <= 0)
				lo = mid;
			else
				hi = mid;
			}

		return hi;
		}

	/**
	 * Search the given ordered object list for an interval containing the given
	 * element. The given helper object is used to compare elements for order
	 * and equality. It is assumed that the list is sorted in the order
	 * determined by the helper object; otherwise, the <TT>searchInterval()</TT>
	 * method's behavior is not specified. An <I>O</I>(log <I>n</I>) binary
	 * search algorithm is used.
	 *
	 * @param  <T>     List element data type.
	 * @param  x       List to be searched.
	 * @param  a       Element to be searched for.
	 * @param  helper  Helper object.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x.get(i-1)</TT> &le; <TT>a</TT>
	 *     &lt; <TT>x.get(i)</TT>. If <TT>a</TT> &lt; <TT>x.get(0)</TT>, then
	 *     <TT>i</TT> = 0 is returned. If <TT>x.get(x.size()-1)</TT> &le;
	 *     <TT>a</TT>, then <TT>i</TT> = <TT>x.size()</TT> is returned. If
	 *     <TT>x</TT> is empty, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static <T> int searchInterval
		(AList<T> x,
		 T a,
		 Searching.Object<T> helper)
		{
		return searchInterval (x.item, 0, x.size, a, helper);
		}

	/**
	 * Search the given ordered array of type <TT>T[]</TT> for an interval
	 * containing the given element. The given comparator is used to compare
	 * elements for order and equality. It is assumed that the array is sorted
	 * in the order determined by the comparator; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  <T>   Array element data type.
	 * @param  x     Array to be searched.
	 * @param  a     Element to be searched for.
	 * @param  comp  Comparator.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[0]</TT>, then <TT>i</TT> = 0
	 *     is returned. If <TT>x[x.length-1]</TT> &le; <TT>a</TT>, then
	 *     <TT>i</TT> = <TT>x.length</TT> is returned. If <TT>x</TT> is zero
	 *     length, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static <T> int searchInterval
		(T[] x,
		 T a,
		 Comparator<T> comp)
		{
		return searchInterval (x, 0, x.length, a, comp);
		}

	/**
	 * Search a portion of the given ordered array of type <TT>T[]</TT> for an
	 * interval containing the given element. The given comparator is used to
	 * compare elements for order and equality. It is assumed that the array is
	 * sorted in the order determined by the comparator; otherwise, the
	 * <TT>searchInterval()</TT> method's behavior is not specified. An
	 * <I>O</I>(log <I>n</I>) binary search algorithm is used.
	 *
	 * @param  <T>   Array element data type.
	 * @param  x     Array to be searched.
	 * @param  off   Index of first element to be searched.
	 * @param  len   Number of elements to be searched.
	 * @param  a     Element to be searched for.
	 * @param  comp  Comparator.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x[i-1]</TT> &le; <TT>a</TT> &lt;
	 *     <TT>x[i]</TT>. If <TT>a</TT> &lt; <TT>x[off]</TT>, then <TT>i</TT> =
	 *     <TT>off</TT> is returned. If <TT>x[off+len-1]</TT> &le; <TT>a</TT>,
	 *     then <TT>i</TT> = <TT>off+len</TT> is returned. If <TT>len</TT> = 0,
	 *     then <TT>i</TT> = &minus;1 is returned.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>x.length</TT>.
	 */
	public static <T> int searchInterval
		(T[] x,
		 int off,
		 int len,
		 T a,
		 Comparator<T> comp)
		{
		// Check preconditions.
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();

		// Establish loop invariant.
		if (len == 0) return -1;

		int lo = off;
		if (comp.compare (x[lo], a) > 0) return off;

		int hi = off + len - 1;
		if (comp.compare (x[hi], a) <= 0) return off + len;

		// Loop invariant: x[lo] <= a and a < x[hi].
		while (hi - lo > 1)
			{
			int mid = (hi + lo)/2;
			if (comp.compare (x[mid], a) <= 0)
				lo = mid;
			else
				hi = mid;
			}

		return hi;
		}

	/**
	 * Search the given ordered object list for an interval containing the given
	 * element. The given comparator is used to compare elements for order and
	 * equality. It is assumed that the list is sorted in the order determined
	 * by the comparator; otherwise, the <TT>searchInterval()</TT> method's
	 * behavior is not specified. An <I>O</I>(log <I>n</I>) binary search
	 * algorithm is used.
	 *
	 * @param  <T>   List element data type.
	 * @param  x     List to be searched.
	 * @param  a     Element to be searched for.
	 * @param  comp  Comparator.
	 *
	 * @return
	 *     An index <TT>i</TT> such that <TT>x.get(i-1)</TT> &le; <TT>a</TT>
	 *     &lt; <TT>x.get(i)</TT>. If <TT>a</TT> &lt; <TT>x.get(0)</TT>, then
	 *     <TT>i</TT> = 0 is returned. If <TT>x.get(x.size()-1)</TT> &le;
	 *     <TT>a</TT>, then <TT>i</TT> = <TT>x.size()</TT> is returned. If
	 *     <TT>x</TT> is empty, then <TT>i</TT> = &minus;1 is returned.
	 */
	public static <T> int searchInterval
		(AList<T> x,
		 T a,
		 Comparator<T> comp)
		{
		return searchInterval (x.item, 0, x.size, a, comp);
		}

// Unit test main program.

//	/**
//	 * Unit test main program.
//	 * <P>
//	 * Usage: java edu.rit.util.Searching <I>x_elements</I> <I>off</I>
//	 * <I>len</I> <I>a</I>
//	 * <BR><I>x_elements</I> = Array elements to be searched (int)
//	 * <BR><I>off</I> = Index of first element to be searched
//	 * <BR><I>len</I> = Number of elements to be searched
//	 * <BR><I>a</I> = Element to be searched for (int)
//	 */
//	public static void main
//		(String[] args)
//		{
//		if (args.length < 4) usage();
//		int n = args.length - 3;
//		int[] x = new int [n];
//		for (int i = 0; i < n; ++ i)
//			x[i] = java.lang.Integer.parseInt (args[i]);
//		int off = java.lang.Integer.parseInt (args[n]);
//		int len = java.lang.Integer.parseInt (args[n+1]);
//		int a = java.lang.Integer.parseInt (args[n+2]);
//		System.out.printf ("searchUnsorted(x,%d) returns %d%n",
//			a, Searching.searchUnsorted (x, a));
//		System.out.printf ("searchSorted(x,%d) returns %d%n",
//			a, Searching.searchSorted (x, a));
//		System.out.printf ("searchInterval(x,%d) returns %d%n",
//			a, Searching.searchInterval (x, a));
//		System.out.printf ("searchUnsorted(x,%d,%d,%d) returns %d%n",
//			off, len, a, Searching.searchUnsorted (x, off, len, a));
//		System.out.printf ("searchSorted(x,%d,%d,%d) returns %d%n",
//			off, len, a, Searching.searchSorted (x, off, len, a));
//		System.out.printf ("searchInterval(x,%d,%d,%d) returns %d%n",
//			off, len, a, Searching.searchInterval (x, off, len, a));
//		}
//
//	/**
//	 * Print a usage message and exit.
//	 */
//	private static void usage()
//		{
//		System.err.println ("Usage: java edu.rit.util.Searching <x_elements> <off> <len> <a>");
//		System.err.println ("<x_elements> = Array elements to be searched (int)");
//		System.err.println ("<off> = Index of first element to be searched");
//		System.err.println ("<len> = Number of elements to be searched");
//		System.err.println ("<a> = Element to be searched for (int)");
//		System.exit (1);
//		}

	}
