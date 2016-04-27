//******************************************************************************
//
// File:    Sorting.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.Sorting
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

/**
 * Class Sorting provides static methods for sorting arrays and lists of
 * primitive types and object types.
 * <P>
 * <I>Note:</I> The operations in class Sorting are not multiple thread safe.
 *
 * @author  Alan Kaminsky
 * @version 07-Apr-2015
 */
public class Sorting
	{

// Prevent construction.

	private Sorting()
		{
		}

// Exported helper classes.

	/**
	 * Class Sorting.Byte is the base class for a helper object used to sort an
	 * array of type <TT>byte[]</TT>.
	 *
	 * @author  Alan Kaminsky
	 * @version 13-Jun-2013
	 */
	public static class Byte
		{
		/**
		 * An instance of the Sorting.Byte base class.
		 */
		public static final Sorting.Byte DEFAULT = new Sorting.Byte();

		/**
		 * Compare two elements in the given array. This determines the order of
		 * the elements in the sorted array.
		 * <P>
		 * The default implementation returns true if <TT>x[a] &lt; x[b]</TT>,
		 * which sorts the array into ascending order. A subclass can override
		 * this method to obtain a different ordering criterion; for example,
		 * descending order.
		 *
		 * @param  x  Array being sorted.
		 * @param  a  Index of first array element being compared.
		 * @param  b  Index of second array element being compared.
		 *
		 * @return  True if <TT>x[a]</TT> comes before <TT>x[b]</TT> in the
		 *          desired ordering, false otherwise.
		 */
		public boolean comesBefore
			(byte[] x,
			 int a,
			 int b)
			{
			return x[a] < x[b];
			}

		/**
		 * Swap two elements in the given array.
		 * <P>
		 * The default implementation swaps <TT>x[a]</TT> with <TT>x[b]</TT>. A
		 * subclass can override this method to do something different; for
		 * example, to swap the elements of other arrays in addition to
		 * <TT>x</TT>.
		 *
		 * @param  x  Array being sorted.
		 * @param  a  Index of first array element being swapped.
		 * @param  b  Index of second array element being swapped.
		 */
		public void swap
			(byte[] x,
			 int a,
			 int b)
			{
			byte t = x[a];
			x[a] = x[b];
			x[b] = t;
			}
		}

	/**
	 * Class Sorting.Character is the base class for a helper object used to
	 * sort an array of type <TT>char[]</TT>.
	 *
	 * @author  Alan Kaminsky
	 * @version 13-Jun-2013
	 */
	public static class Character
		{
		/**
		 * An instance of the Sorting.Character base class.
		 */
		public static final Sorting.Character DEFAULT = new Sorting.Character();

		/**
		 * Compare two elements in the given array. This determines the order of
		 * the elements in the sorted array.
		 * <P>
		 * The default implementation returns true if <TT>x[a] &lt; x[b]</TT>,
		 * which sorts the array into ascending order. A subclass can override
		 * this method to obtain a different ordering criterion; for example,
		 * descending order.
		 *
		 * @param  x  Array being sorted.
		 * @param  a  Index of first array element being compared.
		 * @param  b  Index of second array element being compared.
		 *
		 * @return  True if <TT>x[a]</TT> comes before <TT>x[b]</TT> in the
		 *          desired ordering, false otherwise.
		 */
		public boolean comesBefore
			(char[] x,
			 int a,
			 int b)
			{
			return x[a] < x[b];
			}

		/**
		 * Swap two elements in the given array.
		 * <P>
		 * The default implementation swaps <TT>x[a]</TT> with <TT>x[b]</TT>. A
		 * subclass can override this method to do something different; for
		 * example, to swap the elements of other arrays in addition to
		 * <TT>x</TT>.
		 *
		 * @param  x  Array being sorted.
		 * @param  a  Index of first array element being swapped.
		 * @param  b  Index of second array element being swapped.
		 */
		public void swap
			(char[] x,
			 int a,
			 int b)
			{
			char t = x[a];
			x[a] = x[b];
			x[b] = t;
			}
		}

	/**
	 * Class Sorting.Short is the base class for a helper object used to sort an
	 * array of type <TT>short[]</TT>.
	 *
	 * @author  Alan Kaminsky
	 * @version 13-Jun-2013
	 */
	public static class Short
		{
		/**
		 * An instance of the Sorting.Short base class.
		 */
		public static final Sorting.Short DEFAULT = new Sorting.Short();

		/**
		 * Compare two elements in the given array. This determines the order of
		 * the elements in the sorted array.
		 * <P>
		 * The default implementation returns true if <TT>x[a] &lt; x[b]</TT>,
		 * which sorts the array into ascending order. A subclass can override
		 * this method to obtain a different ordering criterion; for example,
		 * descending order.
		 *
		 * @param  x  Array being sorted.
		 * @param  a  Index of first array element being compared.
		 * @param  b  Index of second array element being compared.
		 *
		 * @return  True if <TT>x[a]</TT> comes before <TT>x[b]</TT> in the
		 *          desired ordering, false otherwise.
		 */
		public boolean comesBefore
			(short[] x,
			 int a,
			 int b)
			{
			return x[a] < x[b];
			}

		/**
		 * Swap two elements in the given array.
		 * <P>
		 * The default implementation swaps <TT>x[a]</TT> with <TT>x[b]</TT>. A
		 * subclass can override this method to do something different; for
		 * example, to swap the elements of other arrays in addition to
		 * <TT>x</TT>.
		 *
		 * @param  x  Array being sorted.
		 * @param  a  Index of first array element being swapped.
		 * @param  b  Index of second array element being swapped.
		 */
		public void swap
			(short[] x,
			 int a,
			 int b)
			{
			short t = x[a];
			x[a] = x[b];
			x[b] = t;
			}
		}

	/**
	 * Class Sorting.Integer is the base class for a helper object used to sort
	 * an array of type <TT>int[]</TT>.
	 *
	 * @author  Alan Kaminsky
	 * @version 13-Jun-2013
	 */
	public static class Integer
		{
		/**
		 * An instance of the Sorting.Integer base class.
		 */
		public static final Sorting.Integer DEFAULT = new Sorting.Integer();

		/**
		 * Compare two elements in the given array. This determines the order of
		 * the elements in the sorted array.
		 * <P>
		 * The default implementation returns true if <TT>x[a] &lt; x[b]</TT>,
		 * which sorts the array into ascending order. A subclass can override
		 * this method to obtain a different ordering criterion; for example,
		 * descending order.
		 *
		 * @param  x  Array being sorted.
		 * @param  a  Index of first array element being compared.
		 * @param  b  Index of second array element being compared.
		 *
		 * @return  True if <TT>x[a]</TT> comes before <TT>x[b]</TT> in the
		 *          desired ordering, false otherwise.
		 */
		public boolean comesBefore
			(int[] x,
			 int a,
			 int b)
			{
			return x[a] < x[b];
			}

		/**
		 * Swap two elements in the given array.
		 * <P>
		 * The default implementation swaps <TT>x[a]</TT> with <TT>x[b]</TT>. A
		 * subclass can override this method to do something different; for
		 * example, to swap the elements of other arrays in addition to
		 * <TT>x</TT>.
		 *
		 * @param  x  Array being sorted.
		 * @param  a  Index of first array element being swapped.
		 * @param  b  Index of second array element being swapped.
		 */
		public void swap
			(int[] x,
			 int a,
			 int b)
			{
			int t = x[a];
			x[a] = x[b];
			x[b] = t;
			}
		}

	/**
	 * Class Sorting.Long is the base class for a helper object used to sort an
	 * array of type <TT>long[]</TT>.
	 *
	 * @author  Alan Kaminsky
	 * @version 13-Jun-2013
	 */
	public static class Long
		{
		/**
		 * An instance of the Sorting.Long base class.
		 */
		public static final Sorting.Long DEFAULT = new Sorting.Long();

		/**
		 * Compare two elements in the given array. This determines the order of
		 * the elements in the sorted array.
		 * <P>
		 * The default implementation returns true if <TT>x[a] &lt; x[b]</TT>,
		 * which sorts the array into ascending order. A subclass can override
		 * this method to obtain a different ordering criterion; for example,
		 * descending order.
		 *
		 * @param  x  Array being sorted.
		 * @param  a  Index of first array element being compared.
		 * @param  b  Index of second array element being compared.
		 *
		 * @return  True if <TT>x[a]</TT> comes before <TT>x[b]</TT> in the
		 *          desired ordering, false otherwise.
		 */
		public boolean comesBefore
			(long[] x,
			 int a,
			 int b)
			{
			return x[a] < x[b];
			}

		/**
		 * Swap two elements in the given array.
		 * <P>
		 * The default implementation swaps <TT>x[a]</TT> with <TT>x[b]</TT>. A
		 * subclass can override this method to do something different; for
		 * example, to swap the elements of other arrays in addition to
		 * <TT>x</TT>.
		 *
		 * @param  x  Array being sorted.
		 * @param  a  Index of first array element being swapped.
		 * @param  b  Index of second array element being swapped.
		 */
		public void swap
			(long[] x,
			 int a,
			 int b)
			{
			long t = x[a];
			x[a] = x[b];
			x[b] = t;
			}
		}

	/**
	 * Class Sorting.Float is the base class for a helper object used to sort an
	 * array of type <TT>float[]</TT>.
	 *
	 * @author  Alan Kaminsky
	 * @version 13-Jun-2013
	 */
	public static class Float
		{
		/**
		 * An instance of the Sorting.Float base class.
		 */
		public static final Sorting.Float DEFAULT = new Sorting.Float();

		/**
		 * Compare two elements in the given array. This determines the order of
		 * the elements in the sorted array.
		 * <P>
		 * The default implementation returns true if <TT>x[a] &lt; x[b]</TT>,
		 * which sorts the array into ascending order. A subclass can override
		 * this method to obtain a different ordering criterion; for example,
		 * descending order.
		 *
		 * @param  x  Array being sorted.
		 * @param  a  Index of first array element being compared.
		 * @param  b  Index of second array element being compared.
		 *
		 * @return  True if <TT>x[a]</TT> comes before <TT>x[b]</TT> in the
		 *          desired ordering, false otherwise.
		 */
		public boolean comesBefore
			(float[] x,
			 int a,
			 int b)
			{
			return x[a] < x[b];
			}

		/**
		 * Swap two elements in the given array.
		 * <P>
		 * The default implementation swaps <TT>x[a]</TT> with <TT>x[b]</TT>. A
		 * subclass can override this method to do something different; for
		 * example, to swap the elements of other arrays in addition to
		 * <TT>x</TT>.
		 *
		 * @param  x  Array being sorted.
		 * @param  a  Index of first array element being swapped.
		 * @param  b  Index of second array element being swapped.
		 */
		public void swap
			(float[] x,
			 int a,
			 int b)
			{
			float t = x[a];
			x[a] = x[b];
			x[b] = t;
			}
		}

	/**
	 * Class Sorting.Double is the base class for a helper object used to sort
	 * an array of type <TT>double[]</TT>.
	 *
	 * @author  Alan Kaminsky
	 * @version 13-Jun-2013
	 */
	public static class Double
		{
		/**
		 * An instance of the Sorting.Double base class.
		 */
		public static final Sorting.Double DEFAULT = new Sorting.Double();

		/**
		 * Compare two elements in the given array. This determines the order of
		 * the elements in the sorted array.
		 * <P>
		 * The default implementation returns true if <TT>x[a] &lt; x[b]</TT>,
		 * which sorts the array into ascending order. A subclass can override
		 * this method to obtain a different ordering criterion; for example,
		 * descending order.
		 *
		 * @param  x  Array being sorted.
		 * @param  a  Index of first array element being compared.
		 * @param  b  Index of second array element being compared.
		 *
		 * @return  True if <TT>x[a]</TT> comes before <TT>x[b]</TT> in the
		 *          desired ordering, false otherwise.
		 */
		public boolean comesBefore
			(double[] x,
			 int a,
			 int b)
			{
			return x[a] < x[b];
			}

		/**
		 * Swap two elements in the given array.
		 * <P>
		 * The default implementation swaps <TT>x[a]</TT> with <TT>x[b]</TT>. A
		 * subclass can override this method to do something different; for
		 * example, to swap the elements of other arrays in addition to
		 * <TT>x</TT>.
		 *
		 * @param  x  Array being sorted.
		 * @param  a  Index of first array element being swapped.
		 * @param  b  Index of second array element being swapped.
		 */
		public void swap
			(double[] x,
			 int a,
			 int b)
			{
			double t = x[a];
			x[a] = x[b];
			x[b] = t;
			}
		}

	/**
	 * Class Sorting.Object is the abstract base class for a helper object used
	 * to sort an array of objects of type <TT>T[]</TT>.
	 *
	 * @param  <T>  Data type of the array elements.
	 *
	 * @author  Alan Kaminsky
	 * @version 13-Jun-2013
	 */
	public static abstract class Object<T>
		{
		/**
		 * Compare two elements in the given array. This determines the order of
		 * the elements in the sorted array.
		 *
		 * @param  x  Array being sorted.
		 * @param  a  Index of first array element being compared.
		 * @param  b  Index of second array element being compared.
		 *
		 * @return  True if <TT>x[a]</TT> comes before <TT>x[b]</TT> in the
		 *          desired ordering, false otherwise.
		 */
		public abstract boolean comesBefore
			(T[] x,
			 int a,
			 int b);

		/**
		 * Swap two elements in the given array.
		 * <P>
		 * The default implementation swaps <TT>x[a]</TT> with <TT>x[b]</TT>. A
		 * subclass can override this method to do something different; for
		 * example, to swap the elements of other arrays in addition to
		 * <TT>x</TT>.
		 *
		 * @param  x  Array being sorted.
		 * @param  a  Index of first array element being swapped.
		 * @param  b  Index of second array element being swapped.
		 */
		public void swap
			(T[] x,
			 int a,
			 int b)
			{
			T t = x[a];
			x[a] = x[b];
			x[b] = t;
			}
		}

	/**
	 * Class Sorting.Sortable is the base class for a helper object used to sort
	 * an array of objects of type {@linkplain Comparable
	 * Comparable}<TT>&lt;C&gt;[]</TT>.
	 *
	 * @param  <T>  Data type of the array elements.
	 *
	 * @author  Alan Kaminsky
	 * @version 12-Mar-2015
	 */
	public static class Sortable<T>
		extends Sorting.Object<T>
		{
		/**
		 * Compare two elements in the given array. This determines the order of
		 * the elements in the sorted array.
		 * <P>
		 * The default implementation returns true if <TT>x[a].compareTo(x[b])
		 * &lt; 0</TT>, which sorts the array into the elements' natural
		 * ordering. A subclass can override this method to obtain a different
		 * ordering criterion.
		 *
		 * @param  x  Array being sorted.
		 * @param  a  Index of first array element being compared.
		 * @param  b  Index of second array element being compared.
		 *
		 * @return  True if <TT>x[a]</TT> comes before <TT>x[b]</TT> in the
		 *          desired ordering, false otherwise.
		 */
		public boolean comesBefore
			(T[] x,
			 int a,
			 int b)
			{
			Comparable x_a = (Comparable)(x[a]);
			Comparable x_b = (Comparable)(x[b]);
			return x_a.compareTo (x_b) < 0;
			}
		}

// Exported operations.

	/**
	 * Sort the given array of type <TT>byte[]</TT>. The array is sorted into
	 * ascending order. An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort
	 * algorithm is used.
	 *
	 * @param  x  Array to be sorted.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 */
	public static byte[] sort
		(byte[] x)
		{
		return sort (x, 0, x.length, Sorting.Byte.DEFAULT);
		}

	/**
	 * Sort a portion of the given array of type <TT>byte[]</TT>. Elements
	 * <TT>off</TT> through <TT>off+len-1</TT> are sorted in place. The array
	 * portion is sorted into ascending order. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  off     Index of first element to be sorted.
	 * @param  len     Number of elements to be sorted.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>x.length</TT>.
	 */
	public static byte[] sort
		(byte[] x,
		 int off,
		 int len)
		{
		return sort (x, off, len, Sorting.Byte.DEFAULT);
		}

	/**
	 * Sort the given array of type <TT>byte[]</TT>. The given helper object is
	 * used to determine the desired ordering of the array elements and to swap
	 * the array elements. An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort
	 * algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 */
	public static byte[] sort
		(byte[] x,
		 Sorting.Byte helper)
		{
		return sort (x, 0, x.length, helper);
		}

	/**
	 * Sort a portion of the given array of type <TT>byte[]</TT>. Elements
	 * <TT>off</TT> through <TT>off+len-1</TT> are sorted in place. The given
	 * helper object is used to determine the desired ordering of the array
	 * elements and to swap the array elements. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  off     Index of first element to be sorted.
	 * @param  len     Number of elements to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>x.length</TT>.
	 */
	public static byte[] sort
		(byte[] x,
		 int off,
		 int len,
		 Sorting.Byte helper)
		{
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();
		for (int i = 2; i <= len; ++ i)
			{
			siftUp (x, off - 1, i, helper);
			}
		for (int i = len; i >= 2; -- i)
			{
			helper.swap (x, off, i - 1 + off);
			siftDown (x, off - 1, i - 1, helper);
			}
		return x;
		}

	private static void siftUp
		(byte[] x,
		 int offm1, // Offset - 1
		 int c,     // 1-based index
		 Sorting.Byte helper)
		{
		int p = c >> 1; // 1-based index
		while (p >= 1)
			{
			if (helper.comesBefore (x, p + offm1, c + offm1))
				{
				helper.swap (x, p + offm1, c + offm1);
				}
			else
				{
				break;
				}
			c = p;
			p = c >> 1;
			}
		}

	private static void siftDown
		(byte[] x,
		 int offm1, // Offset - 1
		 int n,     // 1-based index
		 Sorting.Byte helper)
		{
		int p  = 1; // 1-based index
		int ca = 2; // 1-based index
		int cb = 3; // 1-based index
		while (ca <= n)
			{
			if (cb <= n && helper.comesBefore (x, ca + offm1, cb + offm1))
				{
				if (helper.comesBefore (x, p + offm1, cb + offm1))
					{
					helper.swap (x, p + offm1, cb + offm1);
					p = cb;
					}
				else
					{
					break;
					}
				}
			else
				{
				if (helper.comesBefore (x, p + offm1, ca + offm1))
					{
					helper.swap (x, p + offm1, ca + offm1);
					p = ca;
					}
				else
					{
					break;
					}
				}
			ca = p << 1;
			cb = ca + 1;
			}
		}

	/**
	 * Sort the given array of type <TT>char[]</TT>. The array is sorted into
	 * ascending order. An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort
	 * algorithm is used.
	 *
	 * @param  x  Array to be sorted.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 */
	public static char[] sort
		(char[] x)
		{
		return sort (x, 0, x.length, Sorting.Character.DEFAULT);
		}

	/**
	 * Sort a portion of the given array of type <TT>char[]</TT>. Elements
	 * <TT>off</TT> through <TT>off+len-1</TT> are sorted in place. The array
	 * portion is sorted into ascending order. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  off     Index of first element to be sorted.
	 * @param  len     Number of elements to be sorted.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>x.length</TT>.
	 */
	public static char[] sort
		(char[] x,
		 int off,
		 int len)
		{
		return sort (x, off, len, Sorting.Character.DEFAULT);
		}

	/**
	 * Sort the given array of type <TT>char[]</TT>. The given helper object is
	 * used to determine the desired ordering of the array elements and to swap
	 * the array elements. An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort
	 * algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 */
	public static char[] sort
		(char[] x,
		 Sorting.Character helper)
		{
		return sort (x, 0, x.length, helper);
		}

	/**
	 * Sort a portion of the given array of type <TT>char[]</TT>. Elements
	 * <TT>off</TT> through <TT>off+len-1</TT> are sorted in place. The given
	 * helper object is used to determine the desired ordering of the array
	 * elements and to swap the array elements. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  off     Index of first element to be sorted.
	 * @param  len     Number of elements to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>x.length</TT>.
	 */
	public static char[] sort
		(char[] x,
		 int off,
		 int len,
		 Sorting.Character helper)
		{
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();
		for (int i = 2; i <= len; ++ i)
			{
			siftUp (x, off - 1, i, helper);
			}
		for (int i = len; i >= 2; -- i)
			{
			helper.swap (x, off, i - 1 + off);
			siftDown (x, off - 1, i - 1, helper);
			}
		return x;
		}

	private static void siftUp
		(char[] x,
		 int offm1, // Offset - 1
		 int c,     // 1-based index
		 Sorting.Character helper)
		{
		int p = c >> 1; // 1-based index
		while (p >= 1)
			{
			if (helper.comesBefore (x, p + offm1, c + offm1))
				{
				helper.swap (x, p + offm1, c + offm1);
				}
			else
				{
				break;
				}
			c = p;
			p = c >> 1;
			}
		}

	private static void siftDown
		(char[] x,
		 int offm1, // Offset - 1
		 int n,     // 1-based index
		 Sorting.Character helper)
		{
		int p  = 1; // 1-based index
		int ca = 2; // 1-based index
		int cb = 3; // 1-based index
		while (ca <= n)
			{
			if (cb <= n && helper.comesBefore (x, ca + offm1, cb + offm1))
				{
				if (helper.comesBefore (x, p + offm1, cb + offm1))
					{
					helper.swap (x, p + offm1, cb + offm1);
					p = cb;
					}
				else
					{
					break;
					}
				}
			else
				{
				if (helper.comesBefore (x, p + offm1, ca + offm1))
					{
					helper.swap (x, p + offm1, ca + offm1);
					p = ca;
					}
				else
					{
					break;
					}
				}
			ca = p << 1;
			cb = ca + 1;
			}
		}

	/**
	 * Sort the given array of type <TT>short[]</TT>. The array is sorted into
	 * ascending order. An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort
	 * algorithm is used.
	 *
	 * @param  x  Array to be sorted.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 */
	public static short[] sort
		(short[] x)
		{
		return sort (x, 0, x.length, Sorting.Short.DEFAULT);
		}

	/**
	 * Sort a portion of the given array of type <TT>short[]</TT>. Elements
	 * <TT>off</TT> through <TT>off+len-1</TT> are sorted in place. The array
	 * portion is sorted into ascending order. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  off     Index of first element to be sorted.
	 * @param  len     Number of elements to be sorted.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>x.length</TT>.
	 */
	public static short[] sort
		(short[] x,
		 int off,
		 int len)
		{
		return sort (x, off, len, Sorting.Short.DEFAULT);
		}

	/**
	 * Sort the given array of type <TT>short[]</TT>. The given helper object is
	 * used to determine the desired ordering of the array elements and to swap
	 * the array elements. An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort
	 * algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 */
	public static short[] sort
		(short[] x,
		 Sorting.Short helper)
		{
		return sort (x, 0, x.length, helper);
		}

	/**
	 * Sort a portion of the given array of type <TT>short[]</TT>. Elements
	 * <TT>off</TT> through <TT>off+len-1</TT> are sorted in place. The given
	 * helper object is used to determine the desired ordering of the array
	 * elements and to swap the array elements. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  off     Index of first element to be sorted.
	 * @param  len     Number of elements to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>x.length</TT>.
	 */
	public static short[] sort
		(short[] x,
		 int off,
		 int len,
		 Sorting.Short helper)
		{
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();
		for (int i = 2; i <= len; ++ i)
			{
			siftUp (x, off - 1, i, helper);
			}
		for (int i = len; i >= 2; -- i)
			{
			helper.swap (x, off, i - 1 + off);
			siftDown (x, off - 1, i - 1, helper);
			}
		return x;
		}

	private static void siftUp
		(short[] x,
		 int offm1, // Offset - 1
		 int c,     // 1-based index
		 Sorting.Short helper)
		{
		int p = c >> 1; // 1-based index
		while (p >= 1)
			{
			if (helper.comesBefore (x, p + offm1, c + offm1))
				{
				helper.swap (x, p + offm1, c + offm1);
				}
			else
				{
				break;
				}
			c = p;
			p = c >> 1;
			}
		}

	private static void siftDown
		(short[] x,
		 int offm1, // Offset - 1
		 int n,     // 1-based index
		 Sorting.Short helper)
		{
		int p  = 1; // 1-based index
		int ca = 2; // 1-based index
		int cb = 3; // 1-based index
		while (ca <= n)
			{
			if (cb <= n && helper.comesBefore (x, ca + offm1, cb + offm1))
				{
				if (helper.comesBefore (x, p + offm1, cb + offm1))
					{
					helper.swap (x, p + offm1, cb + offm1);
					p = cb;
					}
				else
					{
					break;
					}
				}
			else
				{
				if (helper.comesBefore (x, p + offm1, ca + offm1))
					{
					helper.swap (x, p + offm1, ca + offm1);
					p = ca;
					}
				else
					{
					break;
					}
				}
			ca = p << 1;
			cb = ca + 1;
			}
		}

	/**
	 * Sort the given array of type <TT>int[]</TT>. The array is sorted into
	 * ascending order. An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort
	 * algorithm is used.
	 *
	 * @param  x  Array to be sorted.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 */
	public static int[] sort
		(int[] x)
		{
		return sort (x, 0, x.length, Sorting.Integer.DEFAULT);
		}

	/**
	 * Sort a portion of the given array of type <TT>int[]</TT>. Elements
	 * <TT>off</TT> through <TT>off+len-1</TT> are sorted in place. The array
	 * portion is sorted into ascending order. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  off     Index of first element to be sorted.
	 * @param  len     Number of elements to be sorted.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int[] sort
		(int[] x,
		 int off,
		 int len)
		{
		return sort (x, off, len, Sorting.Integer.DEFAULT);
		}

	/**
	 * Sort the given array of type <TT>int[]</TT>. The given helper object is
	 * used to determine the desired ordering of the array elements and to swap
	 * the array elements. An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort
	 * algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 */
	public static int[] sort
		(int[] x,
		 Sorting.Integer helper)
		{
		return sort (x, 0, x.length, helper);
		}

	/**
	 * Sort a portion of the given array of type <TT>int[]</TT>. Elements
	 * <TT>off</TT> through <TT>off+len-1</TT> are sorted in place. The given
	 * helper object is used to determine the desired ordering of the array
	 * elements and to swap the array elements. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  off     Index of first element to be sorted.
	 * @param  len     Number of elements to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>x.length</TT>.
	 */
	public static int[] sort
		(int[] x,
		 int off,
		 int len,
		 Sorting.Integer helper)
		{
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();
		for (int i = 2; i <= len; ++ i)
			{
			siftUp (x, off - 1, i, helper);
			}
		for (int i = len; i >= 2; -- i)
			{
			helper.swap (x, off, i - 1 + off);
			siftDown (x, off - 1, i - 1, helper);
			}
		return x;
		}

	private static void siftUp
		(int[] x,
		 int offm1, // Offset - 1
		 int c,     // 1-based index
		 Sorting.Integer helper)
		{
		int p = c >> 1; // 1-based index
		while (p >= 1)
			{
			if (helper.comesBefore (x, p + offm1, c + offm1))
				{
				helper.swap (x, p + offm1, c + offm1);
				}
			else
				{
				break;
				}
			c = p;
			p = c >> 1;
			}
		}

	private static void siftDown
		(int[] x,
		 int offm1, // Offset - 1
		 int n,     // 1-based index
		 Sorting.Integer helper)
		{
		int p  = 1; // 1-based index
		int ca = 2; // 1-based index
		int cb = 3; // 1-based index
		while (ca <= n)
			{
			if (cb <= n && helper.comesBefore (x, ca + offm1, cb + offm1))
				{
				if (helper.comesBefore (x, p + offm1, cb + offm1))
					{
					helper.swap (x, p + offm1, cb + offm1);
					p = cb;
					}
				else
					{
					break;
					}
				}
			else
				{
				if (helper.comesBefore (x, p + offm1, ca + offm1))
					{
					helper.swap (x, p + offm1, ca + offm1);
					p = ca;
					}
				else
					{
					break;
					}
				}
			ca = p << 1;
			cb = ca + 1;
			}
		}

	/**
	 * Sort the given integer list. The list is sorted into ascending order. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x  List to be sorted.
	 *
	 * @return  The list that was sorted (<TT>x</TT>).
	 */
	public static IntList sort
		(IntList x)
		{
		sort (x.item, 0, x.size, Sorting.Integer.DEFAULT);
		return x;
		}

	/**
	 * Sort the given integer list. The given helper object is used to determine
	 * the desired ordering of the list elements and to swap the list elements.
	 * An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       List to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The list that was sorted (<TT>x</TT>).
	 */
	public static IntList sort
		(IntList x,
		 Sorting.Integer helper)
		{
		sort (x.item, 0, x.size, helper);
		return x;
		}

	/**
	 * Sort the given array of type <TT>long[]</TT>. The array is sorted into
	 * ascending order. An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort
	 * algorithm is used.
	 *
	 * @param  x  Array to be sorted.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 */
	public static long[] sort
		(long[] x)
		{
		return sort (x, 0, x.length, Sorting.Long.DEFAULT);
		}

	/**
	 * Sort a portion of the given array of type <TT>long[]</TT>. Elements
	 * <TT>off</TT> through <TT>off+len-1</TT> are sorted in place. The array
	 * portion is sorted into ascending order. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  off     Index of first element to be sorted.
	 * @param  len     Number of elements to be sorted.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>x.length</TT>.
	 */
	public static long[] sort
		(long[] x,
		 int off,
		 int len)
		{
		return sort (x, off, len, Sorting.Long.DEFAULT);
		}

	/**
	 * Sort the given array of type <TT>long[]</TT>. The given helper object is
	 * used to determine the desired ordering of the array elements and to swap
	 * the array elements. An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort
	 * algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 */
	public static long[] sort
		(long[] x,
		 Sorting.Long helper)
		{
		return sort (x, 0, x.length, helper);
		}

	/**
	 * Sort a portion of the given array of type <TT>long[]</TT>. Elements
	 * <TT>off</TT> through <TT>off+len-1</TT> are sorted in place. The given
	 * helper object is used to determine the desired ordering of the array
	 * elements and to swap the array elements. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  off     Index of first element to be sorted.
	 * @param  len     Number of elements to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>x.length</TT>.
	 */
	public static long[] sort
		(long[] x,
		 int off,
		 int len,
		 Sorting.Long helper)
		{
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();
		for (int i = 2; i <= len; ++ i)
			{
			siftUp (x, off - 1, i, helper);
			}
		for (int i = len; i >= 2; -- i)
			{
			helper.swap (x, off, i - 1 + off);
			siftDown (x, off - 1, i - 1, helper);
			}
		return x;
		}

	private static void siftUp
		(long[] x,
		 int offm1, // Offset - 1
		 int c,     // 1-based index
		 Sorting.Long helper)
		{
		int p = c >> 1; // 1-based index
		while (p >= 1)
			{
			if (helper.comesBefore (x, p + offm1, c + offm1))
				{
				helper.swap (x, p + offm1, c + offm1);
				}
			else
				{
				break;
				}
			c = p;
			p = c >> 1;
			}
		}

	private static void siftDown
		(long[] x,
		 int offm1, // Offset - 1
		 int n,     // 1-based index
		 Sorting.Long helper)
		{
		int p  = 1; // 1-based index
		int ca = 2; // 1-based index
		int cb = 3; // 1-based index
		while (ca <= n)
			{
			if (cb <= n && helper.comesBefore (x, ca + offm1, cb + offm1))
				{
				if (helper.comesBefore (x, p + offm1, cb + offm1))
					{
					helper.swap (x, p + offm1, cb + offm1);
					p = cb;
					}
				else
					{
					break;
					}
				}
			else
				{
				if (helper.comesBefore (x, p + offm1, ca + offm1))
					{
					helper.swap (x, p + offm1, ca + offm1);
					p = ca;
					}
				else
					{
					break;
					}
				}
			ca = p << 1;
			cb = ca + 1;
			}
		}

	/**
	 * Sort the given long integer list. The list is sorted into ascending
	 * order. An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is
	 * used.
	 *
	 * @param  x  List to be sorted.
	 *
	 * @return  The list that was sorted (<TT>x</TT>).
	 */
	public static LongList sort
		(LongList x)
		{
		sort (x.item, 0, x.size, Sorting.Long.DEFAULT);
		return x;
		}

	/**
	 * Sort the given long integer list. The given helper object is used to
	 * determine the desired ordering of the list elements and to swap the list
	 * elements. An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm
	 * is used.
	 *
	 * @param  x       List to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The list that was sorted (<TT>x</TT>).
	 */
	public static LongList sort
		(LongList x,
		 Sorting.Long helper)
		{
		sort (x.item, 0, x.size, helper);
		return x;
		}

	/**
	 * Sort the given array of type <TT>float[]</TT>. The array is sorted into
	 * ascending order. An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort
	 * algorithm is used.
	 *
	 * @param  x  Array to be sorted.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 */
	public static float[] sort
		(float[] x)
		{
		return sort (x, 0, x.length, Sorting.Float.DEFAULT);
		}

	/**
	 * Sort a portion of the given array of type <TT>float[]</TT>. Elements
	 * <TT>off</TT> through <TT>off+len-1</TT> are sorted in place. The array
	 * portion is sorted into ascending order. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  off     Index of first element to be sorted.
	 * @param  len     Number of elements to be sorted.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>x.length</TT>.
	 */
	public static float[] sort
		(float[] x,
		 int off,
		 int len)
		{
		return sort (x, off, len, Sorting.Float.DEFAULT);
		}

	/**
	 * Sort the given array of type <TT>float[]</TT>. The given helper object is
	 * used to determine the desired ordering of the array elements and to swap
	 * the array elements. An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort
	 * algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 */
	public static float[] sort
		(float[] x,
		 Sorting.Float helper)
		{
		return sort (x, 0, x.length, helper);
		}

	/**
	 * Sort a portion of the given array of type <TT>float[]</TT>. Elements
	 * <TT>off</TT> through <TT>off+len-1</TT> are sorted in place. The given
	 * helper object is used to determine the desired ordering of the array
	 * elements and to swap the array elements. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  off     Index of first element to be sorted.
	 * @param  len     Number of elements to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>x.length</TT>.
	 */
	public static float[] sort
		(float[] x,
		 int off,
		 int len,
		 Sorting.Float helper)
		{
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();
		for (int i = 2; i <= len; ++ i)
			{
			siftUp (x, off - 1, i, helper);
			}
		for (int i = len; i >= 2; -- i)
			{
			helper.swap (x, off, i - 1 + off);
			siftDown (x, off - 1, i - 1, helper);
			}
		return x;
		}

	private static void siftUp
		(float[] x,
		 int offm1, // Offset - 1
		 int c,     // 1-based index
		 Sorting.Float helper)
		{
		int p = c >> 1; // 1-based index
		while (p >= 1)
			{
			if (helper.comesBefore (x, p + offm1, c + offm1))
				{
				helper.swap (x, p + offm1, c + offm1);
				}
			else
				{
				break;
				}
			c = p;
			p = c >> 1;
			}
		}

	private static void siftDown
		(float[] x,
		 int offm1, // Offset - 1
		 int n,     // 1-based index
		 Sorting.Float helper)
		{
		int p  = 1; // 1-based index
		int ca = 2; // 1-based index
		int cb = 3; // 1-based index
		while (ca <= n)
			{
			if (cb <= n && helper.comesBefore (x, ca + offm1, cb + offm1))
				{
				if (helper.comesBefore (x, p + offm1, cb + offm1))
					{
					helper.swap (x, p + offm1, cb + offm1);
					p = cb;
					}
				else
					{
					break;
					}
				}
			else
				{
				if (helper.comesBefore (x, p + offm1, ca + offm1))
					{
					helper.swap (x, p + offm1, ca + offm1);
					p = ca;
					}
				else
					{
					break;
					}
				}
			ca = p << 1;
			cb = ca + 1;
			}
		}

	/**
	 * Sort the given float list. The list is sorted into ascending order. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x  List to be sorted.
	 *
	 * @return  The list that was sorted (<TT>x</TT>).
	 */
	public static FloatList sort
		(FloatList x)
		{
		sort (x.item, 0, x.size, Sorting.Float.DEFAULT);
		return x;
		}

	/**
	 * Sort the given float list. The given helper object is used to determine
	 * the desired ordering of the list elements and to swap the list elements.
	 * An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       List to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The list that was sorted (<TT>x</TT>).
	 */
	public static FloatList sort
		(FloatList x,
		 Sorting.Float helper)
		{
		sort (x.item, 0, x.size, helper);
		return x;
		}

	/**
	 * Sort the given array of type <TT>double[]</TT>. The array is sorted into
	 * ascending order. An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort
	 * algorithm is used.
	 *
	 * @param  x  Array to be sorted.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 */
	public static double[] sort
		(double[] x)
		{
		return sort (x, 0, x.length, Sorting.Double.DEFAULT);
		}

	/**
	 * Sort a portion of the given array of type <TT>double[]</TT>. Elements
	 * <TT>off</TT> through <TT>off+len-1</TT> are sorted in place. The array
	 * portion is sorted into ascending order. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  off     Index of first element to be sorted.
	 * @param  len     Number of elements to be sorted.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>x.length</TT>.
	 */
	public static double[] sort
		(double[] x,
		 int off,
		 int len)
		{
		return sort (x, off, len, Sorting.Double.DEFAULT);
		}

	/**
	 * Sort the given array of type <TT>double[]</TT>. The given helper object
	 * is used to determine the desired ordering of the array elements and to
	 * swap the array elements. An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>)
	 * heapsort algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 */
	public static double[] sort
		(double[] x,
		 Sorting.Double helper)
		{
		return sort (x, 0, x.length, helper);
		}

	/**
	 * Sort a portion of the given array of type <TT>double[]</TT>. Elements
	 * <TT>off</TT> through <TT>off+len-1</TT> are sorted in place. The given
	 * helper object is used to determine the desired ordering of the array
	 * elements and to swap the array elements. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  off     Index of first element to be sorted.
	 * @param  len     Number of elements to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>x.length</TT>.
	 */
	public static double[] sort
		(double[] x,
		 int off,
		 int len,
		 Sorting.Double helper)
		{
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();
		for (int i = 2; i <= len; ++ i)
			{
			siftUp (x, off - 1, i, helper);
			}
		for (int i = len; i >= 2; -- i)
			{
			helper.swap (x, off, i - 1 + off);
			siftDown (x, off - 1, i - 1, helper);
			}
		return x;
		}

	private static void siftUp
		(double[] x,
		 int offm1, // Offset - 1
		 int c,     // 1-based index
		 Sorting.Double helper)
		{
		int p = c >> 1; // 1-based index
		while (p >= 1)
			{
			if (helper.comesBefore (x, p + offm1, c + offm1))
				{
				helper.swap (x, p + offm1, c + offm1);
				}
			else
				{
				break;
				}
			c = p;
			p = c >> 1;
			}
		}

	private static void siftDown
		(double[] x,
		 int offm1, // Offset - 1
		 int n,     // 1-based index
		 Sorting.Double helper)
		{
		int p  = 1; // 1-based index
		int ca = 2; // 1-based index
		int cb = 3; // 1-based index
		while (ca <= n)
			{
			if (cb <= n && helper.comesBefore (x, ca + offm1, cb + offm1))
				{
				if (helper.comesBefore (x, p + offm1, cb + offm1))
					{
					helper.swap (x, p + offm1, cb + offm1);
					p = cb;
					}
				else
					{
					break;
					}
				}
			else
				{
				if (helper.comesBefore (x, p + offm1, ca + offm1))
					{
					helper.swap (x, p + offm1, ca + offm1);
					p = ca;
					}
				else
					{
					break;
					}
				}
			ca = p << 1;
			cb = ca + 1;
			}
		}

	/**
	 * Sort the given double list. The list is sorted into ascending order. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x  List to be sorted.
	 *
	 * @return  The list that was sorted (<TT>x</TT>).
	 */
	public static DoubleList sort
		(DoubleList x)
		{
		sort (x.item, 0, x.size, Sorting.Double.DEFAULT);
		return x;
		}

	/**
	 * Sort the given double list. The given helper object is used to determine
	 * the desired ordering of the list elements and to swap the list elements.
	 * An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       List to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The list that was sorted (<TT>x</TT>).
	 */
	public static DoubleList sort
		(DoubleList x,
		 Sorting.Double helper)
		{
		sort (x.item, 0, x.size, helper);
		return x;
		}

	/**
	 * Sort the given comparable object array of type <TT>T[]</TT>. The array is
	 * sorted into the element's natural ordering as determined by the element's
	 * {@link Comparable#compareTo(Object) compareTo()} method. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  <T>  Data type of the array elements.
	 * @param  x    Array to be sorted.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 */
	public static <T extends Comparable<T>> T[] sort
		(T[] x)
		{
		return sort (x, 0, x.length, new Sorting.Sortable<T>());
		}

	/**
	 * Sort a portion of the given comparable object array of type <TT>T[]</TT>.
	 * Elements <TT>off</TT> through <TT>off+len-1</TT> are sorted in place. The
	 * array portion is sorted into the element's natural ordering as determined
	 * by the element's {@link Comparable#compareTo(Object) compareTo()} method.
	 * An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  <T>  Data type of the array elements.
	 * @param  x    Array to be sorted.
	 * @param  off  Index of first element to be sorted.
	 * @param  len  Number of elements to be sorted.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>x.length</TT>.
	 */
	public static <T extends Comparable<T>> T[] sort
		(T[] x,
		 int off,
		 int len)
		{
		return sort (x, off, len, new Sorting.Sortable<T>());
		}

	/**
	 * Sort the given object array of type <TT>T[]</TT>. The given helper object
	 * is used to determine the desired ordering of the array elements and to
	 * swap the array elements. An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>)
	 * heapsort algorithm is used.
	 *
	 * @param  <T>     Data type of the array elements.
	 * @param  x       Array to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 */
	public static <T> T[] sort
		(T[] x,
		 Sorting.Object<T> helper)
		{
		return sort (x, 0, x.length, helper);
		}

	/**
	 * Sort a portion of the given object array of type <TT>T[]</TT>. Elements
	 * <TT>off</TT> through <TT>off+len-1</TT> are sorted in place. The given
	 * helper object is used to determine the desired ordering of the array
	 * elements and to swap the array elements. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  x       Array to be sorted.
	 * @param  off     Index of first element to be sorted.
	 * @param  len     Number of elements to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The array that was sorted (<TT>x</TT>).
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>x.length</TT>.
	 */
	public static <T> T[] sort
		(T[] x,
		 int off,
		 int len,
		 Sorting.Object<T> helper)
		{
		if (off < 0 || len < 0 || off + len > x.length)
			throw new IndexOutOfBoundsException();
		for (int i = 2; i <= len; ++ i)
			{
			siftUp (x, off - 1, i, helper);
			}
		for (int i = len; i >= 2; -- i)
			{
			helper.swap (x, off, i - 1 + off);
			siftDown (x, off - 1, i - 1, helper);
			}
		return x;
		}

	private static <T> void siftUp
		(T[] x,
		 int offm1, // Offset - 1
		 int c,     // 1-based index
		 Sorting.Object<T> helper)
		{
		int p = c >> 1; // 1-based index
		while (p >= 1)
			{
			if (helper.comesBefore (x, p + offm1, c + offm1))
				{
				helper.swap (x, p + offm1, c + offm1);
				}
			else
				{
				break;
				}
			c = p;
			p = c >> 1;
			}
		}

	private static <T> void siftDown
		(T[] x,
		 int offm1, // Offset - 1
		 int n,     // 1-based index
		 Sorting.Object<T> helper)
		{
		int p  = 1; // 1-based index
		int ca = 2; // 1-based index
		int cb = 3; // 1-based index
		while (ca <= n)
			{
			if (cb <= n && helper.comesBefore (x, ca + offm1, cb + offm1))
				{
				if (helper.comesBefore (x, p + offm1, cb + offm1))
					{
					helper.swap (x, p + offm1, cb + offm1);
					p = cb;
					}
				else
					{
					break;
					}
				}
			else
				{
				if (helper.comesBefore (x, p + offm1, ca + offm1))
					{
					helper.swap (x, p + offm1, ca + offm1);
					p = ca;
					}
				else
					{
					break;
					}
				}
			ca = p << 1;
			cb = ca + 1;
			}
		}

	/**
	 * Sort the given comparable object list. The list is sorted into the
	 * element's natural ordering as determined by the element's {@link
	 * Comparable#compareTo(Object) compareTo()} method. An
	 * <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  <T>  Data type of the list elements.
	 * @param  x    List to be sorted.
	 *
	 * @return  The list that was sorted (<TT>x</TT>).
	 */
	public static <T extends Comparable<T>> AList<T> sort
		(AList<T> x)
		{
		sort (x.item, 0, x.size, new Sorting.Sortable<T>());
		return x;
		}

	/**
	 * Sort the given object list. The given helper object is used to determine
	 * the desired ordering of the list elements and to swap the list elements.
	 * An <I>O</I>(<I>n</I>&nbsp;log&nbsp;<I>n</I>) heapsort algorithm is used.
	 *
	 * @param  <T>     Data type of the list elements.
	 * @param  x       List to be sorted.
	 * @param  helper  Helper object.
	 *
	 * @return  The list that was sorted (<TT>x</TT>).
	 */
	public static <T> AList<T> sort
		(AList<T> x,
		 Sorting.Object<T> helper)
		{
		sort (x.item, 0, x.size, helper);
		return x;
		}

// Unit test main program.

//	/**
//	 * Unit test main program.
//	 * <P>
//	 * Usage: java edu.rit.util.Sorting <I>n</I> <I>off</I> <I>len</I>
//	 * <I>seed</I>
//	 * <BR><I>n</I> = Array length
//	 * <BR><I>off</I> = Index of first element to be sorted
//	 * <BR><I>len</I> = Number of elements to be sorted
//	 * <BR><I>seed</I> = Random seed
//	 */
//	public static void main
//		(String[] args)
//		{
//		if (args.length != 4) usage();
//		int n = java.lang.Integer.parseInt (args[0]);
//		int off = java.lang.Integer.parseInt (args[1]);
//		int len = java.lang.Integer.parseInt (args[2]);
//		long seed = java.lang.Long.parseLong (args[3]);
//		byte[] x = new byte [n];
//		Random prng = new Random (seed);
//		for (int i = 0; i < n; ++ i) x[i] = prng.nextByte();
//		for (int i = 0; i < n; ++ i) System.out.printf ("%d ", x[i]);
//		System.out.println();
//		Sorting.sort (x, off, len);
//		for (int i = 0; i < n; ++ i) System.out.printf ("%d ", x[i]);
//		System.out.println();
//		}
//
//	/**
//	 * Print a usage message and exit.
//	 */
//	private static void usage()
//		{
//		System.err.println ("Usage: java edu.rit.util.Sorting <n> <off> <len> <seed>");
//		System.err.println ("<n> = Array length");
//		System.err.println ("<off> = Index of first element to be sorted");
//		System.err.println ("<len> = Number of elements to be sorted");
//		System.err.println ("<seed> = Random seed");
//		System.exit (1);
//		}

	}
