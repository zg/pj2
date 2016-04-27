//******************************************************************************
//
// File:    ScatterFiles.java
// Package: edu.rit.pjmr.util
// Unit:    Class edu.rit.pjmr.util.ScatterFiles
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

package edu.rit.pjmr.util;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.pj2.Job;
import edu.rit.pj2.Task;
import edu.rit.util.AList;
import edu.rit.util.Sorting;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;

/**
 * Class ScatterFiles is a cluster parallel program for scattering the files in
 * a directory among the nodes of a cluster parallel computer.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pjmr.util.ScatterFiles <I>nodes</I> <I>host</I>
 * <I>port</I> <I>directory</I></TT>
 * <BR><TT><I>nodes</I></TT> = List of one or more comma-separated backend node
 * names
 * <BR><TT><I>host</I></TT> = Frontend node host name or IP address
 * <BR><TT><I>port</I></TT> = Frontend node port number
 * <BR><TT><I>directory</I></TT> = Directory full pathname
 * <P>
 * The program reads files from the given <I>directory</I> on the cluster's
 * frontend node. The program scatters those files among the given backend
 * <I>nodes</I>. Each backend node stores its files in the given
 * <I>directory</I> also. The program distributes the files to the backend nodes
 * such that the files occupy about the same amount of storage on each backend
 * node. The program chooses the backend node to receive each frontend file in
 * an unspecified manner; any file could end up on any backend.
 * <P>
 * The program scatters only the files that reside in the given directory. The
 * program does not descend recursively into any subdirectories.
 * <P>
 * The files are transferred via a direct socket connection from the frontend
 * node to each backend node. The frontend node listens for connections on the
 * given <I>host</I> and <I>port</I>. The backend nodes connect to that host and
 * port.
 *
 * @author  Alan Kaminsky
 * @version 29-Oct-2015
 */
public class ScatterFiles
	extends Job
	{

	/**
	 * Job main program.
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Parse command line arguments.
		if (args.length != 4) usage();
		String[] nodes = args[0].split (",");
		String host = args[1];
		int port = Integer.parseInt (args[2]);
		File directory = new File (args[3]);
		if (! directory.isDirectory())
			{
			System.err.printf ("ScatterFiles: %s is not a directory%n",
				directory);
			usage();
			}
		if (! directory.isAbsolute())
			{
			System.err.printf ("ScatterFiles: %s is not a full pathname%n",
				directory);
			usage();
			}

		// Set up writer task on frontend node.
		rule() .task (WriterTask.class) .args (args) .runInJobProcess();

		// Set up reader tasks on backend nodes.
		for (String node : nodes)
			rule() .task (ReaderTask.class) .args (node, host, ""+port)
				.nodeName (node);
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pjmr.util.ScatterFiles <nodes> <host> <port> <directory>");
		System.err.println ("<nodes> = List of one or more comma-separated backend node names");
		System.err.println ("<host> = Frontend node host name or IP address");
		System.err.println ("<port> = Frontend node port number");
		System.err.println ("<directory> = Directory full pathname");
		throw new IllegalArgumentException();
		}

	/**
	 * Writer task.
	 */
	private static class WriterTask
		extends Task
		{
		// Writer task main program.
		public void main
			(String[] args)
			throws Exception
			{
			// Parse command line arguments.
			String[] nodes = args[0].split (",");
			int B = nodes.length;
			String host = args[1];
			int port = Integer.parseInt (args[2]);
			File directory = new File (args[3]);

			// Accept socket connections from backend nodes.
			AList<Socket> socketList = new AList<Socket>();
			ServerSocket ss = new ServerSocket();
			ss.bind (new InetSocketAddress (host, port));
			for (int i = 0; i < B; ++ i)
				socketList.addLast (ss.accept());

			// List frontend files and sizes, compute total size.
			final File[] files = directory.listFiles (new FileFilter()
				{
				public boolean accept (File pathname)
					{
					return pathname.isFile();
					}
				});
			final long[] fileSize = new long [files.length];
			long totalSize = 0L;
			for (int i = 0; i < files.length; ++ i)
				{
				fileSize[i] = files[i].length();
				totalSize += fileSize[i];
				}

			// Sort files into descending order of length.
			Sorting.sort (fileSize, new Sorting.Long()
				{
				public boolean comesBefore (long[] x, int a, int b)
					{
					return x[a] > x[b];
					}
				public void swap (long[] x, int a, int b)
					{
					super.swap (x, a, b);
					File f = files[a]; files[a] = files[b]; files[b] = f;
					}
				});

			// Set up array of total file size on each node.
			long[] nodeSize = new long [B];

			// Set up list of files for each node.
			AList<File>[] filesForNode = (AList<File>[]) new AList [B];
			for (int i = 0; i < B; ++ i)
				filesForNode[i] = new AList<File>();

			// Allocate each file to the node with the smallest total file size.
			for (File file : files)
				{
				int mini = 0;
				long minTotalSize = Long.MAX_VALUE;
				for (int i = 0; i < B; ++ i)
					if (nodeSize[i] < minTotalSize)
						{
						mini = i;
						minTotalSize = nodeSize[i];
						}
				nodeSize[mini] += file.length();
				filesForNode[mini].addLast (file);
				}

			// Read files from frontend directory, distribute to backend nodes.
			long fefc = 0L;
			long febc = 0L;
			for (int i = 0; i < B; ++ i)
				{
				Socket socket = socketList.get (i);
				OutStream out = new OutStream (socket.getOutputStream());
				InputStream in;
				for (int j = 0; j < filesForNode[i].size(); ++ j)
					{
					File file = filesForNode[i].get (j);
					long size = file.length();
					++ fefc;
					System.out.printf ("Node %d of %d, file %d of %d, %s%n",
						i + 1, B, fefc, files.length, file);
					System.out.flush();
					out.writeObject (file);
					out.writeLong (size);
					in = new BufferedInputStream (new FileInputStream (file));
					for (long k = 0; k < size; ++ k)
						{
						out.writeByte ((byte)in.read());
						++ febc;
						}
					in.close();
					}
				out.writeObject (null);
				out.close();
				socket.close();
				}

			// Print results.
			System.out.printf ("Frontend: %d files, %d bytes%n", fefc, febc);
			System.out.flush();
			}

		// The writer task requires one core.
		protected static int coresRequired()
			{
			return 1;
			}
		}

	/**
	 * Reader task.
	 */
	private static class ReaderTask
		extends Task
		{
		// Reader task main program.
		public void main
			(String[] args)
			throws Exception
			{
			// Parse command line arguments.
			String node = args[0];
			String host = args[1];
			int port = Integer.parseInt (args[2]);

			// Set up socket connection to frontend node.
			Socket socket = new Socket();
			socket.connect (new InetSocketAddress (host, port));
			InStream in = new InStream (socket.getInputStream());

			// Read files, write to backend directory.
			long befc = 0L;
			long bebc = 0L;
			OutputStream out;
			File file;
			while ((file = (File) in.readObject()) != null)
				{
				++ befc;
				out = new BufferedOutputStream (new FileOutputStream (file));
				long size = in.readLong();
				for (long i = 0; i < size; ++ i)
					{
					out.write (in.readByte());
					++ bebc;
					}
				out.close();
				file.setReadable (true, false);
				file.setWritable (true, false);
				}

			// Print results.
			System.out.printf ("Backend %s: %d files, %d bytes%n",
				node, befc, bebc);
			System.out.flush();
			}

		// The reader task requires one core.
		protected static int coresRequired()
			{
			return 1;
			}
		}

	}
