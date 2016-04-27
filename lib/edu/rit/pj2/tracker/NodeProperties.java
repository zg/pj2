//******************************************************************************
//
// File:    NodeProperties.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.NodeProperties
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

package edu.rit.pj2.tracker;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import java.io.IOException;
import java.util.Scanner;

/**
 * Class NodeProperties encapsulates the capabilities a computational node must
 * have in order to run a certain PJ2 {@linkplain edu.rit.pj2.Task Task} as part
 * of a PJ2 {@linkplain edu.rit.pj2.Job Job}. The following capabilities may be
 * specified:
 * <UL>
 * <LI><TT>nodeName</TT> &mdash; The name of the node needed.
 * <LI><TT>cores</TT> &mdash; The number of CPU cores needed.
 * <LI><TT>gpus</TT> &mdash; The number of GPU accelerators needed.
 * </UL>
 *
 * @author  Alan Kaminsky
 * @version 30-May-2014
 */
public class NodeProperties
	implements Streamable
	{

// Exported constants.

	/**
	 * Indicates that the <TT>nodeName</TT> property is defaulted.
	 */
	public static final String DEFAULT_NODE_NAME = null;

	/**
	 * Indicates that the <TT>cores</TT> property is defaulted.
	 */
	public static final int DEFAULT_CORES = -2;

	/**
	 * Indicates that the <TT>gpus</TT> property is defaulted.
	 */
	public static final int DEFAULT_GPUS = -2;

	/**
	 * Indicates that the task can run on any node of the cluster.
	 */
	public static final String ANY_NODE_NAME = "";

	/**
	 * Indicates that the task requires all the cores on the node.
	 */
	public static final int ALL_CORES = -1;

	/**
	 * Indicates that the task requires all the GPU accelerators on the node.
	 */
	public static final int ALL_GPUS = -1;

// Hidden data members.

	String nodeName = DEFAULT_NODE_NAME;
	int cores = DEFAULT_CORES;
	int gpus = DEFAULT_GPUS;

// Exported constructors.

	/**
	 * Construct a new node properties object. All settings are defaulted.
	 */
	public NodeProperties()
		{
		}

	/**
	 * Construct a new node properties object with the given settings.
	 *
	 * @param  nodeName  Node name on which the task must execute, {@link
	 *                   #ANY_NODE_NAME}, or {@link #DEFAULT_NODE_NAME}.
	 * @param  cores     Number of CPU cores the task requires, {@link
	 *                   #ALL_CORES}, or {@link #DEFAULT_CORES}.
	 * @param  gpus      Number of GPU accelerators the task requires, {@link
	 *                   #ALL_GPUS}, or {@link #DEFAULT_GPUS}.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>cores</TT> or <TT>gpus</TT> is
	 *     illegal.
	 */
	public NodeProperties
		(String nodeName,
		 int cores,
		 int gpus)
		{
		this.nodeName (nodeName);
		this.cores (cores);
		this.gpus (gpus);
		}

	/**
	 * Construct a new node properties object that is a copy of the given node
	 * properties object.
	 *
	 * @param  node  Node properties object to copy.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>node</TT> is null.
	 */
	public NodeProperties
		(NodeProperties node)
		{
		this.nodeName (node.nodeName);
		this.cores (node.cores);
		this.gpus (node.gpus);
		}

	/**
	 * Construct a new node properties object with capabilities specified by
	 * the given string. The <TT>capabilities</TT> string must be in the format
	 * <TT>"<I>name,cores,gpus</I>"</TT>, where <TT><I>name</I></TT> is the name
	 * of the node, <TT><I>cores</I></TT> is the number of CPU cores in the node
	 * (&ge; 1), and <TT><I>gpus</I></TT> is the number of GPU accelerators in
	 * the node (&ge; 0).
	 *
	 * @param  capabilities  Capabilities string.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>capabilities</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>capabilities</TT> is illegal.
	 */
	public NodeProperties
		(String capabilities)
		{
		Scanner s = new Scanner (capabilities) .useDelimiter (",");

		if (! s.hasNext())
			throw new IllegalArgumentException
				("NodeProperties(): Node name missing");
		nodeName = s.next();
		if (nodeName.length() == 0)
			throw new IllegalArgumentException
				("NodeProperties(): Node name illegal");

		if (! s.hasNextInt())
			throw new IllegalArgumentException
				("NodeProperties(): CPU cores missing");
		cores = s.nextInt();
		if (cores < 1)
			throw new IllegalArgumentException
				("NodeProperties(): CPU cores illegal");

		if (! s.hasNextInt())
			throw new IllegalArgumentException
				("NodeProperties(): GPU accelerators missing");
		gpus = s.nextInt();
		if (gpus < 0)
			throw new IllegalArgumentException
				("NodeProperties(): GPU accelerators illegal");
		}

// Exported operations.

	/**
	 * Set the <TT>nodeName</TT> property. The <TT>nodeName</TT> property
	 * specifies the name of the cluster node on which the task must run.
	 *
	 * @param  nodeName  Node name, {@link #ANY_NODE_NAME}, or {@link
	 *                   #DEFAULT_NODE_NAME}.
	 *
	 * @return  This node properties object.
	 *
	 * @see  #nodeName()
	 */
	public NodeProperties nodeName
		(String nodeName)
		{
		this.nodeName = nodeName;
		return this;
		}

	/**
	 * Get the <TT>nodeName</TT> property. The <TT>nodeName</TT> property
	 * specifies the name of the cluster node on which the task must run. If the
	 * <TT>nodeName</TT> property is defaulted, {@link #ANY_NODE_NAME} is
	 * returned, indicating that the task can run on any node of the cluster.
	 *
	 * @return  Node name, or {@link #ANY_NODE_NAME}.
	 *
	 * @see  #nodeName(String)
	 */
	public String nodeName()
		{
		return nodeName == DEFAULT_NODE_NAME ? ANY_NODE_NAME : nodeName;
		}

	/**
	 * Set the <TT>cores</TT> property. The <TT>cores</TT> property specifies
	 * the number of CPU cores the task requires.
	 *
	 * @param  cores  Number of cores (&ge; 1), {@link #ALL_CORES}, or {@link
	 *                #DEFAULT_CORES}.
	 *
	 * @return  This node properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>cores</TT> is illegal.
	 *
	 * @see  #cores()
	 */
	public NodeProperties cores
		(int cores)
		{
		if (cores < DEFAULT_CORES || cores == 0)
			throw new IllegalArgumentException (String.format
				("NodeProperties.cores(): cores = %d illegal", cores));
		this.cores = cores;
		return this;
		}

	/**
	 * Get the <TT>cores</TT> property. The <TT>cores</TT> property specifies
	 * the number of CPU cores the task requires. If the <TT>cores</TT> property
	 * is defaulted, {@link #ALL_CORES} is returned, indicating that the task
	 * requires all the cores on the node.
	 *
	 * @return  Number of cores (&ge; 1), or {@link #ALL_CORES}.
	 *
	 * @see  #cores(int)
	 */
	public int cores()
		{
		return cores == DEFAULT_CORES ? ALL_CORES : cores;
		}

	/**
	 * Set the <TT>gpus</TT> property. The <TT>gpus</TT> property specifies the
	 * number of GPU accelerators the task requires.
	 *
	 * @param  gpus  Number of GPUs (&ge; 0), {@link #ALL_GPUS}, or {@link
	 *               #DEFAULT_GPUS}.
	 *
	 * @return  This node properties object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>gpus</TT> is illegal.
	 *
	 * @see  #gpus()
	 */
	public NodeProperties gpus
		(int gpus)
		{
		if (gpus < DEFAULT_GPUS)
			throw new IllegalArgumentException (String.format
				("NodeProperties.gpus(): gpus = %d illegal", gpus));
		this.gpus = gpus;
		return this;
		}

	/**
	 * Get the <TT>gpus</TT> property. The <TT>gpus</TT> property specifies the
	 * number of GPU accelerators the task requires. If the <TT>gpus</TT>
	 * property is defaulted, 0 is returned, indicating that the task requires
	 * no GPU accelerators.
	 *
	 * @return  Number of GPUs (&ge; 0).
	 *
	 * @see  #gpus(int)
	 */
	public int gpus()
		{
		return gpus == DEFAULT_GPUS ? 0 : gpus;
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
		out.writeString (nodeName);
		out.writeInt (cores);
		out.writeInt (gpus);
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
		nodeName = in.readString();
		cores = in.readInt();
		gpus = in.readInt();
		}

	/**
	 * Returns a string version of this node capabilities object.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		return String.format
			("NodeProperties(nodeName=\"%s\",cores=%d,gpus=%d)",
			 nodeName(), cores(), gpus());
		}

	}
