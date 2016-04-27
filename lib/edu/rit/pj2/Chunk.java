//******************************************************************************
//
// File:    Chunk.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.Chunk
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

package edu.rit.pj2;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import java.io.IOException;

/**
 * Class Chunk is a {@link Tuple} specifying the iterations of a loop with an
 * index of type <TT>int</TT>. A chunk has lower bound, upper bound, and stride
 * fields. Use a chunk to control a for loop as follows. Note that the loop
 * index goes from the lower bound to the upper bound <I>inclusive.</I>
 * <P><PRE>
 *     Chunk chunk = ...;
 *     for (int i = chunk.lb(); i &lt;= chunk.ub(); i += chunk.stride())
 *        {
 *        ...
 *        }</PRE>
 * <P>
 * If the stride is known to be 1 (the default), the for loop can be coded this
 * way:
 * <P><PRE>
 *     Chunk chunk = ...;
 *     for (int i = chunk.lb(); i &lt;= chunk.ub(); ++ i)
 *        {
 *        ...
 *        }</PRE>
 * <P>
 * If a chunk's lower bound is greater than the upper bound, it represents a
 * loop with no iterations. If a chunk's lower bound is less than or equal to
 * the upper bound, it represents a loop with one or more iterations. The stride
 * must be positive. This restricts the possible loops to those whose loop index
 * increases from the lower bound to the upper bound. You can obtain other loop
 * index patterns by <I>transforming</I> an increasing loop index. For example,
 * here is a code fragment that counts from 10 down to 0:
 * <P><PRE>
 *     Chunk chunk = new Chunk() .lb (0) .ub (10);
 *     for (int i = chunk.lb(); i &lt;= chunk.ub(); ++ i)
 *        System.out.printf ("%d%n", chunk.ub() - i);</PRE>
 * <P>
 * A chunk also has a rank field for use in a cluster parallel program. If a
 * chunk must be processed by a specific worker task, the rank field is set to
 * the worker task's rank (&ge; 0). If a chunk can be processed by any worker
 * task, the rank field is set to {@link #ANY ANY}.
 *
 * @author  Alan Kaminsky
 * @version 20-Jan-2014
 */
public class Chunk
	extends Tuple
	{

// Exported constants.

	/**
	 * Designates that the chunk may be processed by any worker task.
	 */
	public static final int ANY = -1;

// Hidden data members.

	private int rank = ANY;
	private int lb = 0;
	private int ub = 0;
	private int stride = 1;

// Exported constructors.

	/**
	 * Construct a new chunk. The rank is {@link #ANY}. The lower bound, upper
	 * bound, and stride are 0, 0, and 1, respectively.
	 */
	public Chunk()
		{
		}

	/**
	 * Construct a new chunk that is a copy of the given chunk.
	 *
	 * @param  chunk  Chunk to copy.
	 */
	public Chunk
		(Chunk chunk)
		{
		copy (chunk);
		}

// Exported operations.

	/**
	 * Set this chunk's rank.
	 *
	 * @param  rank  Worker task rank (&ge; 0), or {@link #ANY}.
	 *
	 * @return  This chunk.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>rank</TT> is illegal.
	 */
	public Chunk rank
		(int rank)
		{
		if (rank < ANY)
			throw new IllegalArgumentException (String.format
				("Chunk.rank(): rank = %d illegal", rank));
		this.rank = rank;
		return this;
		}

	/**
	 * Get this chunk's rank.
	 *
	 * @return  Worker task rank (&ge; 0), or {@link #ANY}.
	 */
	public int rank()
		{
		return this.rank;
		}

	/**
	 * Set this chunk's lower bound.
	 *
	 * @param  lb  Lower bound.
	 *
	 * @return  This chunk.
	 */
	public Chunk lb
		(int lb)
		{
		this.lb = lb;
		return this;
		}

	/**
	 * Get this chunk's lower bound.
	 *
	 * @return  Lower bound.
	 */
	public int lb()
		{
		return this.lb;
		}

	/**
	 * Set this chunk's upper bound.
	 *
	 * @param  ub  Upper bound.
	 *
	 * @return  This chunk.
	 */
	public Chunk ub
		(int ub)
		{
		this.ub = ub;
		return this;
		}

	/**
	 * Get this chunk's upper bound.
	 *
	 * @return  Upper bound.
	 */
	public int ub()
		{
		return this.ub;
		}

	/**
	 * Set this chunk's stride.
	 *
	 * @param  stride  Stride (&ge; 1).
	 *
	 * @return  This chunk.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>stride</TT> is illegal.
	 */
	public Chunk stride
		(int stride)
		{
		if (stride < 1)
			throw new IllegalArgumentException (String.format
				("Chunk.stride(): stride = %d illegal", stride));
		this.stride = stride;
		return this;
		}

	/**
	 * Get this chunk's stride.
	 *
	 * @return  Stride (&ge; 1).
	 */
	public int stride()
		{
		return this.stride;
		}

	/**
	 * Determine if this chunk is empty. If <TT>lb</TT> &le; <TT>ub</TT>, then
	 * this chunk is not empty; otherwise, this chunk is empty.
	 *
	 * @return  True if this chunk is empty, false if it isn't.
	 */
	public boolean isEmpty()
		{
		return lb > ub;
		}

	/**
	 * Get this chunk's length. The length is the number of loop iterations
	 * specified by this chunk. If <TT>lb</TT> &le; <TT>ub</TT>, the length is
	 * <TT>ub-lb+1</TT>; otherwise, the length is 0. (The stride does not affect
	 * the length.) The return value is type <TT>long</TT> because a chunk's
	 * length can be larger than the largest possible <TT>int</TT> value.
	 *
	 * @return  Length.
	 */
	public long length()
		{
		return lb <= ub ? (long)ub - (long)lb + 1L : 0L;
		}

	/**
	 * Determine if this chunk contains the given value. This is so if
	 * <TT>lb</TT> &le; <TT>val</TT> &le; <TT>ub</TT>. (The stride does not
	 * affect the answer.)
	 *
	 * @param  val  Value.
	 *
	 * @return  True if this chunk contains <TT>val</TT>, false otherwise.
	 */
	public boolean contains
		(int val)
		{
		return lb <= val && val <= ub;
		}

	/**
	 * Set this chunk to a copy of the given chunk.
	 *
	 * @param  chunk  Chunk to copy.
	 *
	 * @return  This chunk.
	 */
	public Chunk copy
		(Chunk chunk)
		{
		this.rank = chunk.rank;
		this.lb = chunk.lb;
		this.ub = chunk.ub;
		this.stride = chunk.stride;
		return this;
		}

	/**
	 * Determine if the given target tuple's content matches this template's
	 * content. The target tuple is assumed to be an instance of this template's
	 * matching class or a subclass thereof.
	 * <P>
	 * There is a match if any of these conditions is true:
	 * <UL>
	 * <LI>
	 * The template's rank is {@link #ANY}.
	 * <LI>
	 * The target's rank is {@link #ANY}.
	 * <LI>
	 * The template's rank is equal to the target's rank.
	 * </UL>
	 *
	 * @param  target  Target tuple.
	 *
	 * @return  True if the target tuple's content matches this template tuple's
	 *          content, false otherwise.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>target</TT> is null.
	 */
	public boolean matchContent
		(Tuple target)
		{
		int targetRank = ((Chunk)target).rank;
		return this.rank == ANY || targetRank == ANY || this.rank == targetRank;
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
		out.writeInt (rank);
		out.writeInt (lb);
		out.writeInt (ub);
		out.writeInt (stride);
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
		rank = in.readInt();
		lb = in.readInt();
		ub = in.readInt();
		stride = in.readInt();
		}

	/**
	 * Returns a string version of this chunk.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		return String.format ("Chunk(rank=%d,lb=%d,ub=%d,stride=%d)",
			rank, lb, ub, stride);
		}

	/**
	 * Partition the given loop index range into a series of chunks, and return
	 * all the chunks. Each chunk is the same size (except possibly the last
	 * chunk).
	 *
	 * @param  lb  Loop index lower bound.
	 * @param  ub  Loop index upper bound.
	 * @param  n   Number of chunks (&ge; 1).
	 *
	 * @return  An <TT>n</TT>-element array of chunks. The rank of each chunk in
	 *          the array is equal to the array index.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>n</TT> is illegal.
	 */
	public static Chunk[] partitions
		(int lb,
		 int ub,
		 int n)
		{
		if (n < 1)
			throw new IllegalArgumentException (String.format
				("Chunk.partitions(): n = %d illegal", n));
		long blb = lb;
		long bub = ub;
		long bn = n;
		long chunkSize = Math.max ((bub - blb + bn)/bn, 1L);
		long chunklb = blb;
		long chunkub = 0L;
		Chunk[] c = new Chunk [n];
		for (int r = 0; r < n; ++ r)
			{
			chunkub = Math.min (chunklb + chunkSize - 1L, bub);
			c[r] = new Chunk() .rank (r) .lb ((int)chunklb) .ub ((int)chunkub);
			chunklb = chunkub + 1L;
			}
		return c;
		}

	/**
	 * Partition the given loop index range into a series of chunks, and return
	 * one of the chunks. Each chunk is the same size (except possibly the last
	 * chunk). The chunk with rank <TT>r</TT> is returned.
	 *
	 * @param  lb  Loop index lower bound.
	 * @param  ub  Loop index upper bound.
	 * @param  n   Number of chunks (&ge; 1).
	 * @param  r   Rank of the desired chunk (0 &le; <TT>r</TT> &lt;
	 *             <TT>n</TT>).
	 *
	 * @return  An <TT>n</TT>-element array of chunks. The rank of each chunk in
	 *          the array is equal to the array index.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>n</TT> or <TT>r</TT> is illegal.
	 */
	public static Chunk partition
		(int lb,
		 int ub,
		 int n,
		 int r)
		{
		if (n < 1)
			throw new IllegalArgumentException (String.format
				("Chunk.partition(): n = %d illegal", n));
		if (0 > r || r >= n)
			throw new IllegalArgumentException (String.format
				("Chunk.partition(): r = %d illegal", r));
		long blb = lb;
		long bub = ub;
		long bn = n;
		long chunkSize = Math.max ((bub - blb + bn)/bn, 1L);
		long chunklb = r*chunkSize + blb;
		long chunkub = Math.min (chunklb + chunkSize - 1L, bub);
		return new Chunk() .rank (r) .lb ((int)chunklb) .ub ((int)chunkub);
		}

// Unit test main program.

	/**
	 * Unit test main program.
	 */
	public static void main
		(String[] args)
		{
		if (args.length != 3) usage();
		int lb = Integer.parseInt (args[0]);
		int ub = Integer.parseInt (args[1]);
		int n  = Integer.parseInt (args[2]);

		System.out.printf ("Chunk.partitions (%d, %d, %d)%n", lb, ub, n);
		Chunk[] c = Chunk.partitions (lb, ub, n);
		for (int r = 0; r < n; ++ r)
			System.out.printf ("[%d] = %s length %d%n", r, c[r], c[r].length());

		for (int r = 0; r < n; ++ r)
			{
			Chunk part = Chunk.partition (lb, ub, n, r);
			System.out.printf
				("Chunk.partition (%d, %d, %d, %d) = %s length %d%n",
				 lb, ub, n, r, part, part.length());
			}
		}

	private static void usage()
		{
		System.err.println ("Usage: java edu.rit.pj2.Chunk <lb> <ub> <n>");
		System.exit (1);
		}

	}
