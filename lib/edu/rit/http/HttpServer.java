//******************************************************************************
//
// File:    HttpServer.java
// Package: edu.rit.http
// Unit:    Class edu.rit.http.HttpServer
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

package edu.rit.http;

import edu.rit.util.Logger;
import edu.rit.util.PrintStreamLogger;

import java.io.IOException;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

// For unit test main program
// import edu.rit.util.Action;
// import edu.rit.util.Pair;
// import java.io.PrintWriter;

/**
 * Class HttpServer provides a lightweight HTTP/1.0 server. The HTTP server is
 * designed to be embedded inside another application.
 * <P>
 * When constructed, the HTTP server starts a thread listening for connections
 * to a given host and port. When a web browser sets up a connection, the HTTP
 * server calls the {@link #process(HttpRequest,HttpResponse) process()} method
 * to process the request. This is an abstract method that must be overridden in
 * a subclass. The {@link #process(HttpRequest,HttpResponse) process()} method's
 * arguments are an {@linkplain HttpRequest} object from which the method reads
 * the HTTP request message and an {@linkplain HttpResponse} object to which the
 * method writes the HTTP response message. Each incoming HTTP request is
 * processed in a separate thread obtained from an internal thread pool.
 * <P>
 * When a client opens a socket connection to the HTTP server, the server places
 * a two-second timeout on reading the socket. If the client does not send an
 * HTTP request message before the timeout, the HTTP server closes the socket
 * without processing anything.
 *
 * @author  Alan Kaminsky
 * @version 08-Jan-2015
 */
public abstract class HttpServer
	{

// Hidden data members.

	private ServerSocket myServerSocket;
	private ExecutorService myPool;
	private AcceptorThread myAcceptorThread;
	private Logger myLogger;

// Hidden helper classes.

	/**
	 * Class for accepting incoming socket connections.
	 */
	private class AcceptorThread
		extends Thread
		{
		public void run()
			{
			try
				{
				for (;;)
					{
					myPool.execute
						(new ConnectionProcessor
							(myServerSocket.accept()));
					}
				}
			catch (Throwable exc)
				{
				// Any exception while accepting a connection: Terminate thread.
				if (! myServerSocket.isClosed())
					{
					myLogger.log
						("Exception while accepting HTTP connection",
						 exc);
					}
				}
			finally
				{
				myLogger.log ("HTTP server terminating");
				myPool.shutdown();
				}
			}
		}

	/**
	 * Class for processing one socket connection.
	 */
	private class ConnectionProcessor
		implements Runnable
		{
		private Socket socket;

		public ConnectionProcessor
			(Socket socket)
			{
			this.socket = socket;
			}

		public void run()
			{
			HttpRequest request = null;
			HttpResponse response = null;

			// Set up to process the socket connection.
			try
				{
				socket.setSoTimeout (2000);
				request = new HttpRequest (socket);
				response = new HttpResponse (socket);
				}
			catch (Throwable exc)
				{
				}

			// Process the socket connection.
			try
				{
				if (request != null && response != null)
					process (request, response);
				}
			catch (Throwable exc)
				{
				myLogger.log
					("Exception while processing HTTP request from " +
						toIPAddress (socket.getInetAddress()) +
						": " + request,
					 exc);
				}

			// Close the socket connection.
			try
				{
				if (! socket.isClosed()) socket.close();
				}
			catch (Throwable exc)
				{
				}
			}
		}

// Exported constructors.

	/**
	 * Construct a new HTTP server. The HTTP server will print error messages
	 * on the standard error.
	 *
	 * @param  address  Host and port to which the HTTP server will listen for
	 *                  connections.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public HttpServer
		(InetSocketAddress address)
		throws IOException
		{
		this (address, null);
		}

	/**
	 * Construct a new HTTP server. The HTTP server will print error messages
	 * using the given logger.
	 *
	 * @param  address  Host and port to which the HTTP server will listen for
	 *                  connections.
	 * @param  logger   Error message logger. If null, the HTTP server will
	 *                  print error messages on the standard error.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public HttpServer
		(InetSocketAddress address,
		 Logger logger)
		throws IOException
		{
		myLogger = logger == null ? new PrintStreamLogger() : logger;
		myServerSocket = new ServerSocket();
		myServerSocket.bind (address);
		myPool = Executors.newCachedThreadPool();
		myAcceptorThread = new AcceptorThread();
		myAcceptorThread.setDaemon (true);
		myAcceptorThread.start();
		}

// Exported operations.

	/**
	 * Obtain the host and port to which this HTTP server is listening for
	 * connections.
	 *
	 * @return  Host and port.
	 */
	public InetSocketAddress getAddress()
		{
		return (InetSocketAddress) myServerSocket.getLocalSocketAddress();
		}

	/**
	 * Close this HTTP server.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void close()
		throws IOException
		{
		myServerSocket.close();
		}

// Hidden operations.

	/**
	 * Process the given HTTP request. The <TT>process()</TT> method must be
	 * overridden in a subclass to read the HTTP request from
	 * <TT>theRequest</TT> and write the HTTP response to <TT>theResponse</TT>.
	 *
	 * @param  theRequest   HTTP request.
	 * @param  theResponse  HTTP response.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	protected abstract void process
		(HttpRequest theRequest,
		 HttpResponse theResponse)
		throws IOException;

	/**
	 * Convert the given Internet address to a dotted decimal IP address string.
	 */
	private static String toIPAddress
		(InetAddress addr)
		{
		byte[] ip = addr.getAddress();
		return String.format ("%d.%d.%d.%d",
			ip[0] & 0xff, ip[1] & 0xff, ip[2] & 0xff, ip[3] & 0xff);
		}

// Unit test main program.

//	/**
//	 * Unit test main program. The program listens for connections to
//	 * localhost:8080. The program reads each HTTP request from a web browser
//	 * and merely echoes the request data back to the browser.
//	 * <P>
//	 * Usage: java edu.rit.http.HttpServer
//	 */
//	public static void main
//		(String[] args)
//		throws Exception
//		{
//		HttpServer server =
//			new HttpServer (new InetSocketAddress ("localhost", 8080))
//				{
//				protected void process
//					(HttpRequest request,
//					 HttpResponse response)
//					throws IOException
//					{
//					if (request.isValid())
//						{
//						final PrintWriter out = response.getPrintWriter();
//						out.println ("<HTML>");
//						out.println ("<HEAD>");
//						out.println ("</HEAD>");
//						out.println ("<BODY>");
//						out.println ("<UL>");
//						out.println ("<LI>");
//						out.print   ("Method = <TT>\"");
//						out.print   (request.getMethod());
//						out.println ("\"</TT>");
//						out.println ("<LI>");
//						out.print   ("URI = <TT>\"");
//						out.print   (request.getUri());
//						out.println ("\"</TT>");
//						out.println ("<LI>");
//						out.print   ("Version = <TT>\"");
//						out.print   (request.getHttpVersion());
//						out.println ("\"</TT>");
//						request.forEachHeaderDo (new Action<Pair<String,String>>()
//							{
//							public void run (Pair<String,String> pair)
//								{
//								out.println ("<LI>");
//								out.print   ("Header name = <TT>\"");
//								out.print   (pair.key());
//								out.print   ("\"</TT>, value = <TT>\"");
//								out.print   (pair.value());
//								out.println ("\"</TT>");
//								}
//							});
//						out.println ("</UL>");
//						out.println ("</BODY>");
//						out.println ("</HTML>");
//						}
//					else
//						{
//						response.setStatusCode
//							(HttpResponse.Status.STATUS_400_BAD_REQUEST);
//						PrintWriter out = response.getPrintWriter();
//						out.println ("<HTML>");
//						out.println ("<HEAD>");
//						out.println ("</HEAD>");
//						out.println ("<BODY>");
//						out.println ("<P>");
//						out.println ("400 Bad Request");
//						out.println ("<P>");
//						out.println ("You idiot.");
//						out.println ("</BODY>");
//						out.println ("</HTML>");
//						}
//					response.close();
//					}
//				};
//		
//		Thread.currentThread().join();
//		}

	}
