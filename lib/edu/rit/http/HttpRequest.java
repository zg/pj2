//******************************************************************************
//
// File:    HttpRequest.java
// Package: edu.rit.http
// Unit:    Class edu.rit.http.HttpRequest
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

import edu.rit.util.Action;
import edu.rit.util.Map;
import edu.rit.util.Pair;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import java.net.Socket;

import java.util.Scanner;

// For unit test main program
// import java.io.PrintWriter;
// import java.net.InetSocketAddress;
// import java.net.ServerSocket;
// import java.nio.charset.Charset;

/**
 * Class HttpRequest encapsulates an HTTP request received from a web browser.
 * <P>
 * HTTP/1.0 and HTTP/1.1 requests are supported. The obsolete HTTP/0.9 requests
 * are <I>not</I> supported.
 * <P>
 * This class provides methods for examining the request line and the headers.
 * This class does <I>not</I> support reading the entity body if any.
 * <P>
 * To receive an HTTP request message:
 * <OL TYPE=1>
 * <P><LI>
 * Call the {@link #isValid() isValid()} method.
 * <P><LI>
 * If {@link #isValid() isValid()} returns false, send an HTTP response message
 * indicating the error.
 * <P><LI>
 * If {@link #isValid() isValid()} returns true, call the other methods to
 * examine the contents of the HTTP request message, and send an appropriate
 * HTTP response message.
 * </OL>
 *
 * @author  Alan Kaminsky
 * @version 08-Jan-2015
 */
public class HttpRequest
	{

// Exported constants.

	/**
	 * The GET method string, <TT>"GET"</TT>.
	 */
	public static final String GET_METHOD = "GET";

	/**
	 * The HEAD method string, <TT>"HEAD"</TT>.
	 */
	public static final String HEAD_METHOD = "HEAD";

	/**
	 * The POST method string, <TT>"POST"</TT>.
	 */
	public static final String POST_METHOD = "POST";

	/**
	 * The HTTP/1.0 version string <TT>"HTTP/1.0"</TT>.
	 */
	public static final String HTTP_1_0_VERSION = "HTTP/1.0";

	/**
	 * The HTTP/1.1 version string, <TT>"HTTP/1.1"</TT>.
	 */
	public static final String HTTP_1_1_VERSION = "HTTP/1.1";

// Hidden data members.

	private String myMethod;
	private String myUri;
	private String myHttpVersion;

	private Map<String,String> myHeaderMap = new Map<String,String>();

	private boolean iamValid;

// Exported constructors.

	/**
	 * Construct a new HTTP request. The request is read from the input stream
	 * of the given socket.
	 *
	 * @param  theSocket  Socket.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>theSocket</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred while reading the socket input
	 *     stream.
	 */
	public HttpRequest
		(Socket theSocket)
		throws IOException
		{
		if (theSocket == null)
			throw new NullPointerException
				("HttpRequest(): theSocket is null");
		parse (theSocket);
		}

// Exported operations.

	/**
	 * Determine if this HTTP request is valid. If the data read from the input
	 * stream of the socket given to the constructor represents a valid HTTP
	 * request message, true is returned, otherwise false is returned.
	 *
	 * @return  True if this HTTP request is valid, false otherwise.
	 */
	public boolean isValid()
		{
		return iamValid;
		}

	/**
	 * Obtain this HTTP request's method.
	 *
	 * @return  Method string, e.g. <TT>"GET"</TT>, <TT>"POST"</TT>.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this HTTP request is invalid.
	 */
	public String getMethod()
		{
		if (! isValid())
			throw new IllegalStateException ("HTTP request is invalid");
		return myMethod;
		}

	/**
	 * Obtain this HTTP request's URI.
	 *
	 * @return  URI string.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this HTTP request is invalid.
	 */
	public String getUri()
		{
		if (! isValid())
			throw new IllegalStateException ("HTTP request is invalid");
		return myUri;
		}

	/**
	 * Obtain this HTTP request's version.
	 *
	 * @return  HTTP version string, e.g. <TT>"HTTP/1.0"</TT>,
	 *          <TT>"HTTP/1.1"</TT>.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this HTTP request is invalid.
	 */
	public String getHttpVersion()
		{
		if (! isValid())
			throw new IllegalStateException ("HTTP request is invalid");
		return myHttpVersion;
		}

	/**
	 * Obtain the value of the given header in this HTTP request.
	 *
	 * @param  theHeaderName  Header name.
	 *
	 * @return  Header value, or null if there is no header for
	 *          <TT>theHeaderName</TT>.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if this HTTP request is invalid.
	 */
	public String getHeader
		(String theHeaderName)
		{
		if (! isValid())
			throw new IllegalStateException ("HTTP request is invalid");
		return myHeaderMap.get (theHeaderName);
		}

	/**
	 * Perform the given action on each header in this HTTP request. For each
	 * header in an unspecified order, the given <TT>action</TT>'s
	 * <TT>run()</TT> method is called, passing in a pair containing key =
	 * header name and value = header value.
	 *
	 * @param  action  Action.
	 */
	public void forEachHeaderDo
		(Action<Pair<String,String>> action)
		{
		myHeaderMap.forEachItemDo (action);
		}

	/**
	 * Returns a string version of this HTTP request. The string consists of the
	 * HTTP method, the URI, and the HTTP version.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		if (isValid())
			return myMethod + " " + myUri + " " + myHttpVersion;
		else
			return "Invalid HTTP request";
		}

// Hidden operations.

	/**
	 * Parse the input data read from the given socket.
	 *
	 * @param  theSocket  Socket.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred while reading the socket input
	 *     stream.
	 */
	private void parse
		(Socket theSocket)
		throws IOException
		{
		// Assume the request is invalid.
		iamValid = false;
		myMethod = null;
		myUri = null;
		myHttpVersion = null;

		// Set up to read lines from the socket input stream.
		BufferedReader reader =
			new BufferedReader
				(new InputStreamReader
					(theSocket.getInputStream()));

		// Parse the first line.
		String line = reader.readLine();
		Scanner linescanner = new Scanner (line);
		if (! linescanner.hasNext()) return;
		String method = linescanner.next();
		if (! linescanner.hasNext()) return;
		String uri = linescanner.next();
		if (! linescanner.hasNext()) return;
		String httpVersion = linescanner.next();
		if (linescanner.hasNext()) return;

		// Read and parse remaining lines until an empty line.
		String headerName = null;
		String headerValue = "";
		for (;;)
			{
			line = reader.readLine();
			if (line.length() == 0) break;

			// Check whether line is starting or continuing a header.
			if (Character.isWhitespace (line.charAt (0)))
				{
				// Continuing previous header.
				if (headerName == null) return;
				headerValue += line;
				}
			else
				{
				// Starting new header. Record previous header if any.
				if (headerName != null)
					{
					myHeaderMap.put (headerName, headerValue);
					headerName = null;
					headerValue = "";
					}

				// Parse header name and value.
				int i = line.indexOf (':');
				if (i <= 0) return;
				if (i >= line.length()-1) return;
				if (! Character.isWhitespace (line.charAt (i+1))) return;
				headerName = line.substring (0, i);
				headerValue += line.substring (i+2);
				}
			}

		// If we get here, all is well. Record final header if any.
		if (headerName != null)
			{
			myHeaderMap.put (headerName, headerValue);
			}

		// Record method, URI, and HTTP version.
		myMethod = method;
		myUri = uri;
		myHttpVersion = httpVersion;

		// Mark it valid.
		iamValid = true;
		}

// Unit test main program.

//	/**
//	 * Unit test main program. The program listens for connections to
//	 * localhost:8080. The program reads each HTTP request from a web browser
//	 * and merely echoes the request data back to the browser.
//	 * <P>
//	 * Usage: java edu.rit.http.HttpRequest
//	 */
//	public static void main
//		(String[] args)
//		throws Exception
//		{
//		ServerSocket serversocket = new ServerSocket();
//		serversocket.bind (new InetSocketAddress ("localhost", 8080));
//		for (;;)
//			{
//			Socket socket = serversocket.accept();
//			HttpRequest request = new HttpRequest (socket);
//			final PrintWriter out = new PrintWriter (socket.getOutputStream());
//			out.print ("HTTP/1.0 200 OK\r\n");
//			out.print ("Content-Type: text/plain; charset=");
//			out.print (Charset.defaultCharset() + "\r\n");
//			out.print ("\r\n");
//			if (request.isValid())
//				{
//				out.print ("Method = \"" + request.getMethod() + "\"\r\n");
//				out.print ("URI = \"" + request.getUri() + "\"\r\n");
//				out.print ("Version = \"" + request.getHttpVersion() + "\"\r\n");
//				request.forEachHeaderDo (new Action<Pair<String,String>>()
//					{
//					public void run (Pair<String,String> pair)
//						{
//						out.print ("Header name = \"" + pair.key());
//						out.print ("\", value = \"" + pair.value() + "\"\r\n");
//						}
//					});
//				}
//			else
//				{
//				out.print ("Invalid request\r\n");
//				}
//			out.close();
//			socket.close();
//			}
//		}

	}
