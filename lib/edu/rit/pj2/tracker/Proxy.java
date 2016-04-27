//******************************************************************************
//
// File:    Proxy.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.Proxy
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
import java.io.IOException;
import java.net.InetAddress;
import java.net.Socket;

/**
 * Class Proxy provides an object that sends messages to and receives messages
 * from a certain far end. Messages are sent by a {@linkplain Sender Sender}.
 * Messages are received by a {@linkplain Receiver Receiver}.
 *
 * @author  Alan Kaminsky
 * @version 01-Jul-2014
 */
public class Proxy
	{

// Hidden data members.

	private Socket socket;
	private Sender sender;
	private Receiver receiver;

	private OutStream out;
	private InStream in;

// Exported constructors.

	/**
	 * Construct a new proxy and connect it to the given host and port; call
	 * this constructor to initiate an outgoing connection and attach a proxy to
	 * it. The given sender will be used to send messages. The given receiver
	 * will be used to receive messages.
	 *
	 * @param  host      Host name.
	 * @param  port      Port number.
	 * @param  sender    Sender.
	 * @param  receiver  Receiver.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public Proxy
		(String host,
		 int port,
		 Sender sender,
		 Receiver receiver)
		throws IOException
		{
		this (new Socket (InetAddress.getByName (host), port),
			sender, receiver);
		}

	/**
	 * Construct a new proxy using the given socket; call this constructor to
	 * attach a proxy to an existing outgoing connection. The given sender will
	 * be used to send messages. The given receiver will be used to receive
	 * messages.
	 *
	 * @param  socket    Socket.
	 * @param  sender    Sender.
	 * @param  receiver  Receiver.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public Proxy
		(Socket socket,
		 Sender sender,
		 Receiver receiver)
		throws IOException
		{
		this.socket = socket;
		this.sender = sender;
		this.receiver = receiver;

		// Enable TCP_NODELAY. This lets TCP send short messages immediately.
		this.socket.setTcpNoDelay (true);

		out = new OutStream (socket.getOutputStream());
		out.writeByte (receiver.opcode);
		out.flush();

		in = new InStream (socket.getInputStream());
		byte opcode = in.readByte();
		if (opcode != sender.opcode)
			throw new IOException (String.format
				("Proxy(): Remote type opcode = %d does not match local type opcode = %d",
				 opcode, sender.opcode));

		sender.proxy = this;
		sender.out = this.out;

		receiver.proxy = this;
		receiver.sender = sender;
		receiver.in = this.in;
		receiver.start();
		}

	/**
	 * Construct a new proxy using the given socket; call this constructor to
	 * attach a proxy to an incoming connection that was accepted. The given
	 * receiver will be used to receive messages. The sender is set
	 * automatically to match the entity at the far end.
	 *
	 * @param  socket    Socket.
	 * @param  receiver  Receiver.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public Proxy
		(Socket socket,
		 Receiver receiver)
		throws IOException
		{
		this.socket = socket;
		this.receiver = receiver;

		// Enable TCP_NODELAY. This lets TCP send short messages immediately.
		this.socket.setTcpNoDelay (true);

		out = new OutStream (socket.getOutputStream());
		out.writeByte (receiver.opcode);
		out.flush();

		in = new InStream (socket.getInputStream());
		byte opcode = in.readByte();
		sender = null;
		switch (opcode)
			{
			case Opcode.JOB:
				sender = new JobSender();
				break;
			case Opcode.TRACKER:
				sender = new TrackerSender();
				break;
			case Opcode.LAUNCHER:
				sender = new LauncherSender();
				break;
			case Opcode.BACKEND:
				sender = new BackendSender();
				break;
			default:
				throw new IllegalArgumentException (String.format
					("Proxy(): Remote type opcode = %d illegal", opcode));
			}

		sender.proxy = this;
		sender.out = this.out;

		receiver.proxy = this;
		receiver.sender = sender;
		receiver.in = this.in;
		receiver.start();
		}

// Exported operations.

	/**
	 * Returns the remote host to which this proxy is connected.
	 *
	 * @return  Host name.
	 */
	public String host()
		{
		return socket.getInetAddress().getHostName();
		}

	/**
	 * Returns the remote port to which this proxy is connected.
	 *
	 * @return  Port number.
	 */
	public int port()
		{
		return socket.getPort();
		}

	/**
	 * Returns the local host to which this proxy is connected.
	 *
	 * @return  Host name.
	 */
	public String localHost()
		{
		return socket.getLocalAddress().getHostName();
		}

	/**
	 * Returns the local port to which this proxy is connected.
	 *
	 * @return  Port number.
	 */
	public int localPort()
		{
		return socket.getLocalPort();
		}

	/**
	 * Set the context class loader of this proxy's receiver thread.
	 *
	 * @param  classLoader  Class loader.
	 */
	public void setContextClassLoader
		(ClassLoader classLoader)
		{
		receiver.setContextClassLoader (classLoader);
		}

	/**
	 * Flag stating whether far end is shut down.
	 */
	private boolean farEndShutdown = false;

	/**
	 * Gracefully shut down communication with the far end. The
	 * <TT>shutdown()</TT> method blocks until the far end acknowledges that
	 * communication was shut down.
	 */
	public synchronized void shutdown()
		{
		try
			{
			out.writeByte (Opcode.SHUTDOWN);
			out.flush();
			}
		catch (IOException exc)
			{
			}
		long t = System.currentTimeMillis();
		while (! farEndShutdown && System.currentTimeMillis() < t + 2000L)
			{
			try { wait (2000L); }
				catch (InterruptedException exc) {}
			}
		terminate();
		}

	/**
	 * Report that the far end is shut down.
	 */
	synchronized void farEndShutdown()
		{
		farEndShutdown = true;
		notifyAll();
		}

	/**
	 * Forcibly terminate communication with the far end. The
	 * <TT>terminate()</TT> method returns immediately, without waiting for an
	 * acknowledgment from the far end.
	 */
	public void terminate()
		{
		try { socket.close(); }
			catch (IOException exc) {}
		}

	}
