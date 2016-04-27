//******************************************************************************
//
// File:    CCACell.java
// Package: edu.rit.pj2.example
// Unit:    Class edu.rit.pj2.example.CCACell
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

package edu.rit.pj2.example;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import java.io.IOException;
import java.util.Arrays;
import java.util.Formatter;

/**
 * Class CCACell provides one cell in a continuous cellular automaton. It is a
 * high precision fixed point number with a 32-bit integer part and a 1024-bit
 * fractional part.
 * <P>
 * <I>Note:</I> Class CCACell is not multiple thread safe.
 *
 * @author  Alan Kaminsky
 * @version 28-Jun-2014
 */
public class CCACell
	implements Streamable
	{

// Hidden data members.

	// Number of words in fractional part.
	private static final int FW = 32;

	// Number of words in integer and fractional part.
	private static final int W = FW + 1;

	// Value. Words 0 .. FW-1 are the fractional part in little-endian order.
	// Word FW is the integer part.
	private int[] value = new int [W];

	// Scratch space for the multiply() method.
	private int[] scratch = new int [2*W];

// Exported constructors.

	/**
	 * Construct a new CCA cell. Its initial value is 0.
	 */
	public CCACell()
		{
		}

// Exported operations.

	/**
	 * Assign this cell the given integer value.
	 *
	 * @param  v  Value (&ge; 0).
	 *
	 * @return  This cell, with its value set to <TT>v</TT>.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>v</TT> &lt; 0.
	 */
	public CCACell assign
		(int v)
		{
		if (v < 0)
			throw new IllegalArgumentException (String.format
				("CCACell.assign(): v = %d illegal", v));
		Arrays.fill (value, 0);
		value[FW] = v;
		return this;
		}

	/**
	 * Assign this cell the given rational value.
	 *
	 * @param  n  Numerator (&ge; 0).
	 * @param  d  Denominator (&gt; 0).
	 *
	 * @return  This cell, with its value set to <TT>n/d</TT>.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>n</TT> &lt; 0. Thrown if
	 *     <TT>d</TT> &le; 0.
	 */
	public CCACell assign
		(int n,
		 int d)
		{
		return this.assign (n) .divide (d);
		}

	/**
	 * Assign this cell the value parsed from the given string. The string must
	 * be of the form <TT>"<I>n</I>"</TT> or <TT>"<I>n</I>/<I>d</I>"</TT>, where
	 * <TT><I>n</I></TT> is an integer &ge; 0 and <TT><I>d</I></TT> is an
	 * integer &gt; 0. No whitespace is allowed.
	 *
	 * @param  s  Value string.
	 *
	 * @return  This cell, with its value parsed from <TT>s</TT>.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>s</TT> is illegal.
	 */
	public CCACell assign
		(String s)
		{
		try
			{
			String[] part = s.split ("/", 2);
			if (part.length == 1)
				return this.assign (Integer.parseInt (part[0]));
			else if (part.length == 2)
				return this.assign (Integer.parseInt (part[0]),
					Integer.parseInt (part[1]));
			}
		catch (Throwable exc)
			{
			}
		throw new IllegalArgumentException (String.format
			("CCACell.assign(): s = \"%s\" illegal", s));
		}

	/**
	 * Assign the given cell to this cell.
	 *
	 * @param  cell  Cell to assign.
	 *
	 * @return  This cell, with its value set to <TT>cell</TT>.
	 */
	public CCACell assign
		(CCACell cell)
		{
		System.arraycopy (cell.value, 0, this.value, 0, W);
		return this;
		}

	/**
	 * Add the given cell to this cell.
	 *
	 * @param  cell  Cell to add.
	 *
	 * @return  This cell, with its value set to <TT>this</TT> + <TT>cell</TT>.
	 */
	public CCACell add
		(CCACell cell)
		{
		long acc = 0L;
		for (int i = 0; i < W; ++ i)
			{
			acc = acc + longOf (this.value[i]) + longOf (cell.value[i]);
			this.value[i] = (int) acc;
			acc >>>= 32;
			}
		return this;
		}

	/**
	 * Multiply this cell by the given cell.
	 *
	 * @param  cell  Cell to multiply by.
	 *
	 * @return  This cell, with its value set to <TT>this</TT> &times;
	 *          <TT>cell</TT>.
	 */
	public CCACell multiply
		(CCACell cell)
		{
		long acc, cell_i;
		Arrays.fill (scratch, 0);
		scratch[FW-1] = 0x80000000; // To round the product
		for (int i = 0; i < W; ++ i) // Loop over words of cell
			{
			acc = 0L;
			cell_i = longOf (cell.value[i]);
			for (int j = 0; j < W; ++ j) // Loop over words of this
				{
				acc = acc + longOf (scratch[i+j]) +
					cell_i * longOf (this.value[j]);
				scratch[i+j] = (int) acc;
				acc >>>= 32;
				}
			scratch[i+W] = (int) acc;
			}
		System.arraycopy (scratch, FW, this.value, 0, W);
		return this;
		}

	/**
	 * Divide this cell by the given value.
	 *
	 * @param  v  Divisor (&gt; 0).
	 *
	 * @return  This cell, with its value set to <TT>this</TT> &div; <TT>v</TT>.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>v</TT> &le; 0.
	 */
	public CCACell divide
		(int v)
		{
		if (v <= 0)
			throw new IllegalArgumentException (String.format
				("CCACell.divide(): v = %d illegal", v));
		long vv = longOf (v);
		long acc = 0L;
		for (int i = W - 1; i >= 0; -- i)
			{
			acc = (acc << 32) | longOf (this.value[i]);
			this.value[i] = (int)(acc / vv);
			acc = acc % vv;
			}
		return this;
		}

	/**
	 * Discard the integer part of this cell and keep the fractional part.
	 *
	 * @return  This cell, with the integer part set to 0.
	 */
	public CCACell fracPart()
		{
		this.value[FW] = 0;
		return this;
		}

	/**
	 * Returns this cell's value as type float.
	 *
	 * @return  Float value.
	 */
	public float floatValue()
		{
		return (float)
			(this.value[W-1] +
			 longOf (this.value[W-2]) / 4294967296.0 +
			 longOf (this.value[W-3]) / 18446744073709551616.0);
		}

	/**
	 * Write this cell to the given out stream.
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
		out.writeUnsignedIntArray (this.value);
		}

	/**
	 * Read this cell from the given in stream.
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
		in.readUnsignedIntArray (this.value);
		}

	/**
	 * Returns a string version of this cell. The string is in hexadecimal
	 * format.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		StringBuilder b = new StringBuilder();
		Formatter f = new Formatter (b);
		f.format ("%x.", this.value[FW]);
		for (int i = FW - 1; i >= 0; -- i)
			f.format ("%08x", this.value[i]);
		return b.toString();
		}

// Hidden operations.

	/**
	 * Convert the given unsigned integer to a long.
	 */
	private static long longOf
		(int x)
		{
		return x & 0x00000000FFFFFFFFL;
		}

// Unit test main program.

//	/**
//	 * Unit test main program.
//	 * <P>
//	 * Usage: <TT>java edu.rit.pj2.example.CCACell <I>A</I> <I>B</I>
//	 * <I>C</I></TT>
//	 */
//	public static void main
//		(String[] args)
//		{
//		if (args.length != 3) usage();
//		CCACell A = new CCACell() .assign (args[0]);
//		CCACell B = new CCACell() .assign (args[1]);
//		int C = Integer.parseInt (args[2]);
//		CCACell X = new CCACell();
//		print ("A  ", A);
//		print ("B  ", B);
//		System.out.printf ("C   = 0x%x%n", C);
//		print ("A+B", X.assign (A) .add (B));
//		print ("A*B", X.assign (A) .multiply (B));
//		print ("A/C", X.assign (A) .divide (C));
//		}
//
//	/**
//	 * Print the given cell in hexadecimal and in floating point formats.
//	 */
//	private static void print
//		(String label,
//		 CCACell cell)
//		{
//		System.out.printf ("%s = 0x%s = %.6f%n",
//			label, cell, cell.floatValue());
//		}
//
//	/**
//	 * Print a usage message and exit.
//	 */
//	private static void usage()
//		{
//		System.err.println ("Usage: java edu.rit.pj2.example.CCACell <A> <B> <C>");
//		System.exit (1);
//		}

	}
