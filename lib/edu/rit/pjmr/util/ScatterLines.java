//******************************************************************************
//
// File:    ScatterLines.java
// Package: edu.rit.pjmr.util
// Unit:    Class edu.rit.pjmr.util.ScatterLines
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

import edu.rit.pj2.Job;
import edu.rit.pj2.Task;
import edu.rit.util.AList;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;

/**
 * Class ScatterLines is a cluster parallel program for scattering the lines in
 * one or more text files among the nodes of a cluster parallel computer.
 * <P>
 * Usage: <TT>java pj2 edu.rit.pjmr.util.ScatterLines <I>nodes</I> <I>host</I>
 * <I>port</I> <I>backendfile</I> <I>frontendfile</I> [ <I>frontendfile</I>
 * . . . ]</TT>
 * <BR><TT><I>nodes</I></TT> = List of one or more comma-separated backend node
 * names
 * <BR><TT><I>host</I></TT> = Frontend node host name or IP address
 * <BR><TT><I>port</I></TT> = Frontend node port number
 * <BR><TT><I>backendfile</I></TT> = Backend node file name
 * <BR><TT><I>frontendfile</I></TT> = Frontend node file name
 * <P>
 * The program reads lines from the <I>frontendfile</I> or files on the
 * cluster's frontend node. The program scatters those lines among the given
 * backend <I>nodes</I>. Each backend node stores its lines in the given
 * <I>backendfile</I>. The program distributes the lines to the backend nodes
 * such that all the backend files are about the same size. The program chooses
 * the backend node to receive each frontend file line in an unspecified manner;
 * any line could end up on any backend.
 * <P>
 * The file data is transferred via a direct socket connection from the frontend
 * node to each backend node. The frontend node listens for connections on the
 * given <I>host</I> and <I>port</I>. The backend nodes connect to that host and
 * port.
 *
 * @author  Alan Kaminsky
 * @version 29-Jul-2015
 */
public class ScatterLines
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
		if (args.length < 5) usage();
		String[] nodes = args[0].split (",");
		String host = args[1];
		int port = Integer.parseInt (args[2]);
		String backendFile = args[3];

		// Set up writer task on frontend node.
		rule() .task (WriterTask.class) .args (args) .runInJobProcess();

		// Set up reader tasks on backend nodes.
		for (String node : nodes)
			rule() .task (ReaderTask.class)
				.args (node, host, ""+port, backendFile)
				.nodeName (node);
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java pj2 edu.rit.pjmr.util.ScatterLines <nodes> <host> <port> <backendfile> <frontendfile> [<frontendfile> ...]");
		System.err.println ("<nodes> = List of one or more comma-separated backend node names");
		System.err.println ("<host> = Frontend node host name or IP address");
		System.err.println ("<port> = Frontend node port number");
		System.err.println ("<backendfile> = Backend node file name");
		System.err.println ("<frontendfile> = Frontend node file name");
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
			int N = args.length - 4;
			File[] frontendFile = new File [N];
			for (int i = 0; i < N; ++ i)
				frontendFile[i] = new File (args[i+4]);

			// Accept socket connections from backend nodes.
			AList<Socket> socketList = new AList<Socket>();
			ServerSocket ss = new ServerSocket();
			ss.bind (new InetSocketAddress (host, port));
			for (int i = 0; i < B; ++ i)
				socketList.addLast (ss.accept());

			// Compute total size of all frontend files.
			long totalSize = 0L;
			for (int i = 0; i < N; ++ i)
				if (frontendFile[i].isFile())
					totalSize += frontendFile[i].length();
				else
					{
					System.err.printf ("ScatterLines: \"%s\" is not a file%n",
						frontendFile[i]);
					throw new IllegalArgumentException();
					}

			// Read bytes from frontend files, distribute to backend nodes.
			long felc = 0L;
			long febc = 0L;
			long threshold = totalSize/B;
			int be = 0;
			long bebc = 0L;
			Socket socket = socketList.get (be);
			OutputStream out = new BufferedOutputStream
				(socket.getOutputStream());
			InputStream in;
			int data;
			for (int i = 0; i < N; ++ i)
				{
				System.out.printf ("Scattering %s%n", frontendFile[i]);
				System.out.flush();
				in = new BufferedInputStream
					(new FileInputStream (frontendFile[i]));
				while ((data = in.read()) != -1)
					{
					out.write (data);
					++ febc;
					++ bebc;
					if (data == '\n')
						{
						++ felc;
						if (be < B - 1 && bebc >= threshold)
							{
							out.close();
							socket.close();
							++ be;
							bebc = 0L;
							socket = socketList.get (be);
							out = new BufferedOutputStream
								(socket.getOutputStream());
							}
						}
					}
				in.close();
				}
			out.close();
			socket.close();

			// Print results.
			System.out.printf ("Frontend: %d lines, %d bytes%n", felc, febc);
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
			File backendFile = new File (args[3]);

			// Set up socket connection to frontend node.
			Socket socket = new Socket();
			socket.connect (new InetSocketAddress (host, port));
			InputStream in = new BufferedInputStream (socket.getInputStream());

			// Open backend file for writing.
			OutputStream out = new BufferedOutputStream
				(new FileOutputStream (backendFile));

			// Read bytes, write to backend file.
			long belc = 0L;
			long bebc = 0L;
			int data;
			while ((data = in.read()) != -1)
				{
				out.write (data);
				++ bebc;
				if (data == '\n') ++ belc;
				}

			// Close backend file, make it world readable and writeable.
			out.close();
			backendFile.setReadable (true, false);
			backendFile.setWritable (true, false);

			// Print results.
			System.out.printf ("Backend %s: %d lines, %d bytes%n",
				node, belc, bebc);
			}

		// The reader task requires one core.
		protected static int coresRequired()
			{
			return 1;
			}
		}

	}
